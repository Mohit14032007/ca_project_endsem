#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/memory_config.h"
#include "dram/mmap_controller.h"
#include "hardware/cluster.h"
#include "hardware/device.h"
#include "hardware/hardware_config.h"
#include "hardware/stat.h"
#include "model/model.h"
#include "model/model_config.h"
#include "module/tensor.h"
#include "prefetch/live_prediction_engine.h"
#include "prefetch/prefetch_config.h"
#include "prefetch/prefetch_metrics_collector.h"
#include "prefetch/predictor_interface.h"
#include "prefetch/predictor_registry.h"
#include "scheduler/scheduler.h"

using namespace llm_system;

static size_t count_file_lines(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) return 0;
  size_t lines = 0;
  std::string line;
  while (std::getline(file, line)) {
    lines++;
  }
  return lines;
}

int main(int argc, char* argv[]) {
  bool validate_only = false;
  std::string trace_file_cli = "";
  bool warm_hbm_cli = false;
  std::string config_path = "reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/configs/deepseekv3_b1_full.yaml";

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--validate-config" || arg == "--check-only") {
      validate_only = true;
    } else if (arg == "--trace-file" && i + 1 < argc) {
      trace_file_cli = argv[++i];
    } else if (arg == "--warm-hbm" || arg == "--warm-replay") {
      warm_hbm_cli = true;
    } else if (arg.rfind("--", 0) != 0) {
      config_path = arg;
    }
  }

  std::cout << "======================================================================\n";
  std::cout << "DS-6C-2: DeepSeek-V3 Mixtral-Style Experiment Harness\n";
  std::cout << "Executable: run_deepseekv3_mixtral_style\n";
  std::cout << "Config Path: " << config_path << "\n";
  if (validate_only) {
    std::cout << "Mode: CONFIGURATION VALIDATION ONLY (No simulation will be run)\n";
  }
  std::cout << "======================================================================\n" << std::flush;

  if (!std::filesystem::exists(config_path)) {
    std::cerr << "[ERROR] Configuration file does not exist: " << config_path << "\n";
    return 1;
  }

  YAML::Node config = YAML::LoadFile(config_path);

  // 1. Model Section
  std::string model_name = config["model"]["model_name"].as<std::string>();
  if (model_name != "deepseekV3" && model_name != "debugV3") {
    std::cerr << "[ERROR] Expected model_name 'deepseekV3' or 'debugV3', got: " << model_name << "\n";
    return 2;
  }

  // 2. System Section
  std::string gpu_gen = config["system"]["gpu_gen"].as<std::string>();
  std::string processor_type = config["system"]["processor_type"].as<std::string>();
  int num_node = config["system"]["num_node"].as<int>();
  int num_device = config["system"]["num_device"].as<int>();
  int nvlink_gen = 3;
  if (config["system"]["nvlink_gen"]) {
    nvlink_gen = config["system"]["nvlink_gen"].as<int>();
  }

  int e_tp_dg = config["system"]["distribution"]["expert_tensor_degree"].as<int>();
  int ne_tp_dg = config["system"]["distribution"]["none_expert_tensor_degree"].as<int>();

  bool use_ramulator = config["system"]["optimization"]["use_ramulator"].as<bool>();
  std::string offchip_dram_cfg_path = config["system"]["optimization"]["offchip_dram_cfg_path"].as<std::string>();
  
  long long cache_capacity = 80000000000LL; // 80 GB standard
  if (config["system"]["optimization"]["expert_cache_capacity_bytes"]) {
    cache_capacity = config["system"]["optimization"]["expert_cache_capacity_bytes"].as<long long>();
  } else if (std::getenv("DEBUG_CACHE_CAPACITY")) {
    cache_capacity = std::stoll(std::getenv("DEBUG_CACHE_CAPACITY"));
  }

  bool use_hitting_queue = false;
  if (config["system"]["optimization"]["use_hitting_queue"]) {
    use_hitting_queue = config["system"]["optimization"]["use_hitting_queue"].as<bool>();
  }

  // 3. Serving Section
  int max_batch_size = config["serving"]["max_batch_size"].as<int>();
  int max_process_token = config["serving"]["max_process_token"].as<int>();
  if (max_process_token == 0) {
    max_process_token = 65536 * 8;
  }

  // 4. Simulation Section
  std::string data_name = config["simulation"]["data"].as<std::string>();
  int input_len = config["simulation"]["input_len"].as<int>();
  int output_len = config["simulation"]["output_len"].as<int>();
  int precision_byte = config["simulation"]["precision_byte"].as<int>();
  int iter = config["simulation"]["iter"].as<int>();
  double skewness = config["simulation"]["skewness"].as<double>();
  bool exit_out_of_memory = config["simulation"]["exit_out_of_memory"].as<bool>();
  bool mem_cap_limit = config["simulation"]["mem_cap_limit"].as<bool>();

  std::string trace_file = trace_file_cli;
  if (trace_file.empty() && config["simulation"]["trace_file"]) {
    trace_file = config["simulation"]["trace_file"].as<std::string>();
  }
  bool warm_hbm_preload = warm_hbm_cli;
  if (!warm_hbm_preload && config["simulation"]["warm_hbm_preload"]) {
    warm_hbm_preload = config["simulation"]["warm_hbm_preload"].as<bool>();
  }

  // 4b. Prefetching Configuration (Phase 1)
  PrefetchConfig prefetch_config;
  if (config["simulation"]["prefetching"]) {
    prefetch_config = PrefetchConfig::parse_yaml(config["simulation"]["prefetching"]);
  }
  try {
    prefetch_config.validate();
  } catch (const std::exception& e) {
    std::cerr << "[CONFIGURATION ERROR] Prefetching validation failed:\n  " << e.what() << "\n";
    return 3;
  }

  // 5. Log Section
  std::string output_path = config["log"]["output_directory"].as<std::string>();
  std::string gantt_dir = config["log"]["gantt_directory"].as<std::string>();
  bool export_gantt = config["log"]["export_gantt"].as<bool>();
  bool print_log = config["log"]["print_log"].as<bool>();

  std::string trace_dir = gantt_dir;
  if (config["log"]["traces_directory"]) {
    trace_dir = config["log"]["traces_directory"].as<std::string>();
  } else {
    size_t raw_pos = trace_dir.find("/raw/");
    if (raw_pos != std::string::npos) {
      trace_dir.replace(raw_pos, 5, "/traces/");
    } else {
      trace_dir = output_path + "/../traces/";
    }
  }

  // Set trace and memory environment variables early
  std::filesystem::create_directories(output_path);
  std::filesystem::create_directories(gantt_dir);
  std::filesystem::create_directories(trace_dir);
  setenv("TRACE_DIR", trace_dir.c_str(), 1);
  setenv("OFFCHIP_DRAM_CFG", offchip_dram_cfg_path.c_str(), 1);

  // Validate Memory Configuration Files
  if (!std::filesystem::exists(offchip_dram_cfg_path)) {
    std::cerr << "[ERROR] Off-chip DDR5 config file not found: " << offchip_dram_cfg_path << "\n";
    return 3;
  }
  std::string hbm_cfg_path = "./dram_config_HBM3_80GB.yaml";
  if (config["system"]["optimization"]["hbm_dram_cfg_path"]) {
    hbm_cfg_path = config["system"]["optimization"]["hbm_dram_cfg_path"].as<std::string>();
  }
  if (!std::filesystem::exists(hbm_cfg_path)) {
    std::cerr << "[ERROR] On-chip HBM3 config file not found: " << hbm_cfg_path << "\n";
    return 4;
  }

  // Hardware System Config
  SystemConfig system_config;
  if (gpu_gen == "A100") {
    system_config = A100;
  } else if (gpu_gen == "H100") {
    system_config = H100;
  }
  system_config.num_node = num_node;
  system_config.num_device = num_device;
  system_config.use_ramulator = use_ramulator;
  system_config.offchip_dram_cfg_path = offchip_dram_cfg_path;
  system_config.hbm_dram_cfg_path = hbm_cfg_path;
  system_config.processor_type = {ProcessorType::GPU};
  system_config.high_processor_type = ProcessorType::GPU;
  system_config.low_processor_type = ProcessorType::LOGIC;
  system_config.decode_mode = true;
  system_config.prefill_mode = false;
  system_config.exit_out_of_memory = exit_out_of_memory;
  system_config.mem_cap_limit = mem_cap_limit;

  if (config["system"]["optimization"]["bypass_offchip_execution_cache"]) {
    std::string val_str = config["system"]["optimization"]["bypass_offchip_execution_cache"].as<std::string>();
    std::string s = "";
    for (char c : val_str) {
      s += std::tolower(c);
    }
    if (s == "1" || s == "true" || s == "on" || s == "yes") {
      system_config.bypass_offchip_execution_cache = true;
    } else if (s == "0" || s == "false" || s == "off" || s == "no") {
      system_config.bypass_offchip_execution_cache = false;
    }
  }
  double compute_multiplier = 1.0;
  if (config["simulation"] && config["simulation"]["optimization"] && config["simulation"]["optimization"]["compute_time_multiplier"]) {
    compute_multiplier = config["simulation"]["optimization"]["compute_time_multiplier"].as<double>();
  } else if (config["system"] && config["system"]["optimization"] && config["system"]["optimization"]["compute_time_multiplier"]) {
    compute_multiplier = config["system"]["optimization"]["compute_time_multiplier"].as<double>();
  } else if (config["simulation"] && config["simulation"]["compute_time_multiplier"]) {
    compute_multiplier = config["simulation"]["compute_time_multiplier"].as<double>();
  }
  system_config.compute_time_multiplier = compute_multiplier;
  system_config.apply_environment_overrides();
  system_config.prefetch_config = prefetch_config;

  if (nvlink_gen == 3) {
    system_config.device_ict_bandwidth = 150.0 * 1000 * 1000 * 1000;     // 150 GB/s NVLink 3 (A100)
    system_config.device_ict_latency = 3.0 * 1000;                        // 3000 ns
  }

  // Model Config: DeepSeek-V3 exact architecture
  ModelConfig model_config = deepseekV3;
  if (model_name == "debugV3") {
    model_config = debugV3;
  }
  if (config["model"]["num_layers"]) {
    model_config.num_layers = config["model"]["num_layers"].as<int>();
  }
  model_config.e_tp_dg = e_tp_dg;
  model_config.ne_tp_dg = ne_tp_dg;
  model_config.precision_byte = precision_byte;
  model_config.input_len = input_len;
  model_config.output_len = output_len;
  model_config.skewness = skewness;
  model_config.dataset = data_name;

  if (model_config.precision_byte == 1) {
    system_config.compute_peak_flops *= 2;
  }

  // Preflight Mapping Check: Verify 2.75 TB DDR5 geometry covers routed experts
  MemoryConfig ddr5_cfg = ddr5_32gb_x16_2ch;
  long long divisor_before_row = (long long)ddr5_cfg.granul * ddr5_cfg.num_cube * ddr5_cfg.num_channel *
                                 ddr5_cfg.num_rank * ddr5_cfg.num_bankgroup * ddr5_cfg.num_bank * ddr5_cfg.num_col;
  long long total_capacity = divisor_before_row * ddr5_cfg.num_row;
  int moe_layers = model_config.num_layers - model_config.first_k_dense;
  long long total_routed_experts = (long long)moe_layers * model_config.num_routed_expert;
  long long max_expert_addr = total_routed_experts * 88080384LL;
  long long max_row_gpu = max_expert_addr / divisor_before_row;
  double row_utilization = (double)max_row_gpu / (double)ddr5_cfg.num_row * 100.0;

  std::cout << "\n[TOPOLOGY & GEOMETRY VALIDATION]\n";
  std::cout << "  Model: DeepSeek-V3 (" << model_config.num_layers << " layers: " 
            << model_config.first_k_dense << " dense, " 
            << (model_config.num_layers - model_config.first_k_dense) << " MoE)\n";
  std::cout << "  Experts: " << model_config.num_routed_expert << " routed, top-" 
            << model_config.top_k << ", " << model_config.num_shared_expert << " shared\n";
  std::cout << "  Expert Footprint: 88,080,384 bytes (~88.08 MB FP16)\n";
  std::cout << "  Total Expert Inventory: 14,592 experts = " << max_expert_addr << " bytes (~1.285 TB)\n";
  std::cout << "  DDR5 Backing Capacity: " << total_capacity << " bytes (~2.75 TB / 2.5 TiB across 10,485,760 rows)\n";
  std::cout << "  Max Production Row Index: " << max_row_gpu << " (Row Utilization: " 
            << std::fixed << std::setprecision(2) << row_utilization << "% < 100%)\n";
  std::cout << "  HBM Local Cache Budget: " << cache_capacity << " bytes (" 
            << (cache_capacity / (1024.0 * 1024.0 * 1024.0)) << " GB)\n";
  std::cout << "  Trace Output Directory: " << trace_dir << "\n";
  std::cout << "  Summary CSV Directory: " << output_path << "\n";
  std::cout << "  Workload: input_len=" << input_len << ", output_len=" << output_len 
            << ", iter=" << iter << ", batch_size=" << max_batch_size << "\n";
  std::string prefetch_str = "DISABLED";
  if (prefetch_config.enabled) {
    prefetch_str = "ENABLED";
  }
  std::cout << "  Prefetching: " << prefetch_str
            << " (mode=" << to_string(prefetch_config.mode)
            << ", predictor=" << to_string(prefetch_config.predictor_type)
            << ", budget=" << prefetch_config.budget
            << ", horizon=" << prefetch_config.prediction_horizon
            << ", transfer_model=" << to_string(prefetch_config.transfer_model) << ")\n";
  std::cout << "  Eviction Policy: " << to_string(prefetch_config.eviction_policy);
  if (prefetch_config.eviction_policy == EvictionPolicy::FA_LRU) {
    std::cout << " (fa_lru_window_size=" << prefetch_config.fa_lru_window_size << ")";
  }
  std::cout << "\n";

  IPredictor::Ptr predictor_instance = nullptr;
  if (prefetch_config.enabled && prefetch_config.mode != PrefetchMode::NONE) {
    try {
      predictor_instance = PredictorRegistry::create_predictor(
          prefetch_config.predictor_type, prefetch_config,
          moe_layers, model_config.num_routed_expert, model_config.top_k);
      std::string impl_str = "no";
      if (predictor_instance->is_implemented()) {
        impl_str = "yes";
      }
      std::string eval_str = "no";
      if (predictor_instance->is_evaluation_only()) {
        eval_str = "yes";
      }
      std::cout << "  Predictor Status: instantiated (" << predictor_instance->get_name()
                << ", implemented=" << impl_str
                << ", eval_only=" << eval_str << ")\n";
    } catch (const std::exception& e) {
      std::cerr << "[CONFIGURATION ERROR] Failed to instantiate predictor:\n  " << e.what() << "\n";
      return 5;
    }
  }

  assert(max_row_gpu < ddr5_cfg.num_row);

  if (validate_only) {
    std::cout << "\n======================================================================\n";
    std::cout << "[SUCCESS] ALL ARCHITECTURAL, TOPOLOGY, AND GEOMETRY INVARIANTS VALIDATED!\n";
    std::cout << "Configuration file is 100% syntactically and structurally sound.\n";
    std::cout << "======================================================================\n" << std::flush;
    return 0;
  }

  // Instantiate Simulator Objects
  Scheduler::Ptr scheduler = Scheduler::Create(system_config, model_config, "none", max_batch_size, 8192, max_process_token);
  Cluster::Ptr cluster = Cluster::Create(system_config, scheduler);
  Model model(model_config, cluster, scheduler);
  cluster->set_dependency();

  Device::Ptr device = cluster->get_device(0);
  device->cache_capacity_bytes = cache_capacity;
  if (device->config.prefetch_config.pabi_speculative_buffer_capacity_bytes <= 0 && device->config.prefetch_config.pabi_speculative_fraction > 0.0) {
    device->speculative_buffer_capacity_bytes = static_cast<long long>(device->cache_capacity_bytes * device->config.prefetch_config.pabi_speculative_fraction);
  }
  device->reset_expert_cache();
  if (scheduler != nullptr) {
    scheduler->set_device(device);
  }

  PrefetchMetricsCollector::getInstance().reset();
  PrefetchMetricsCollector::getInstance().set_enabled(true);
  std::string pred_name = "none";
  if (prefetch_config.enabled) {
    if (prefetch_config.predictor_type == PredictorType::ST_MOE) {
      pred_name = "ST_MOE";
    } else if (prefetch_config.predictor_type == PredictorType::CSTP) {
      pred_name = "CSTP";
    }
  }
  int eff_horizon = 0;
  int eff_budget = 0;
  if (prefetch_config.enabled) {
    eff_horizon = prefetch_config.prediction_horizon;
    eff_budget = prefetch_config.budget;
  }
  PrefetchMetricsCollector::getInstance().set_experiment_metadata(
      data_name, max_batch_size, pred_name, eff_horizon,
      eff_budget, config_path);
  PrefetchMetricsCollector::getInstance().set_output_dir(trace_dir);

  std::cout << "\n[INITIALIZATION] Scheduler, Cluster, Device, and Model Graph created.\n" << std::flush;

  // Clean trace files in target trace directory before starting
  std::filesystem::remove(trace_dir + "/expert_routing.csv");
  std::filesystem::remove(trace_dir + "/expert_load.csv");
  std::filesystem::remove(trace_dir + "/expert_cache.csv");

  int ne_dp_dg = system_config.num_device * num_node / ne_tp_dg;
  std::string summary_csv_filename = output_path + "/" + model_name + "_" + data_name +
                                     "_" + std::to_string(input_len) + "_" +
                                     std::to_string(output_len) + "_" + processor_type +
                                     "_N" + std::to_string(num_node) + "_D" +
                                     std::to_string(num_device) + "_TP" +
                                     std::to_string(ne_tp_dg) + "_DP" +
                                     std::to_string(ne_dp_dg) + "_maxbatch" +
                                     std::to_string(max_batch_size) + "_maxprocess" +
                                     std::to_string(max_process_token) + "_iter" +
                                     std::to_string(iter) +
                                     "_skew" + std::to_string(int(model_config.skewness * 10)) +
                                     "_precision_byte" + std::to_string(precision_byte) +
                                     "_parallel_execution" + std::to_string(system_config.parallel_execution) +
                                     "_ramul_decode.csv";

  auto run_host_start = std::chrono::high_resolution_clock::now();

  if (use_hitting_queue) {
    // Standard cluster iteration execution
    std::cout << "\n--- Executing via Cluster::runIteration (with hittingQueue warmup) ---\n" << std::flush;
    scheduler->getActualArrivalTime(iter);
    cluster->runIteration(iter, summary_csv_filename);
  } else {
    // Controlled Sequential Decoding Execution (Profile B)
    std::cout << "\n--- Executing Controlled Sequential Decoding Loop (" << iter << " steps) ---\n" << std::flush;
    scheduler->clear();
    scheduler->initRunningQueue();

    int num_moe_layers = model_config.num_layers;
    if (model_name == "deepseekV3" || model_name == "debugV3") {
      num_moe_layers = model_config.num_layers - model_config.first_k_dense;
    } else if (model_config.expert_freq != 0) {
      num_moe_layers = model_config.num_layers / model_config.expert_freq;
    }

    bool is_real_trace_data = false;
    if (data_name == "moe_anywhere" || data_name.rfind("moe_suite", 0) == 0 || !trace_file.empty()) {
      is_real_trace_data = true;
    }
    if (data_name == "synthetic") {
      is_real_trace_data = false;
      trace_file = ""; // Strict isolation: never use real trace file in synthetic mode
    }

    if (is_real_trace_data) {
      if (trace_file.empty()) {
        throw std::runtime_error("[ERROR] Real trace dataset '" + data_name + "' specified but trace_file is empty! Zero fallback to random allowed.");
      }
      std::cout << "\n[REAL TRACE REPLAY MODE] Ingesting trace: " << trace_file << "\n";
      scheduler->loadRealTrace(trace_file, max_batch_size, num_moe_layers, model_config.num_routed_expert, model_config.top_k, iter);
      for (int b = 0; b < max_batch_size; b++) {
        int seq_out_len = output_len;
        if (seq_out_len < iter) {
          seq_out_len = iter;
        }
        Sequence::Ptr new_seq = Sequence::Create(b, input_len, seq_out_len);
        if (system_config.decode_mode) {
          new_seq->current_len = input_len;
        }
        new_seq->get_expert_from_list = true;
        scheduler->sequence_queue.push_back(new_seq);
      }
    } else {
      std::cout << "\n[SYNTHETIC ROUTING MODE] Using synthetic generator with 4-tuple cache isolation.\n";
      for (int b = 0; b < max_batch_size; b++) {
        scheduler->pushDummySeq(input_len, output_len);
      }
    }
    scheduler->fillRunningQueue();

    if (warm_hbm_preload) {
      std::cout << "[WARM HBM PRELOAD] Preloading replay working set into HBM cache...\n";
      long long total_expert_size = 88080384LL;
      if (!trace_file.empty()) {
        for (int s = 0; s < max_batch_size; s++) {
          for (int t = 0; t < iter; t++) {
            for (int l = 0; l < num_moe_layers; l++) {
              size_t table_idx = (size_t)t * num_moe_layers + l;
              int target_layer = (l + 1) % num_moe_layers;
              for (int e_id : scheduler->sequences_info[s]->expert_list[table_idx]) {
                if (!device->is_expert_resident(target_layer, e_id)) {
                  device->mark_expert_resident(target_layer, e_id, total_expert_size);
                }
              }
            }
          }
        }
      }
      std::cout << "[WARM HBM PRELOAD] Resident experts in HBM: " << device->lru_expert_list.size() 
                << " (" << (device->resident_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB)\n" << std::flush;
    }

    std::ofstream summary_csv(summary_csv_filename);
    summary_csv << "iter_info,split,type,time,latency,queueing_delay,arrival_time,seq_queue_size,"
                << "input_len,output_len,num_sum_iter,mixed,batchsize,numtoken,num_sum_seq,num_gen_seq,seqlen,"
                << "sum_attention_opb,qkvgen,q_down_proj,kv_down_proj,kr_proj,q_up_proj,qr_proj,kv_up_proj,"
                << "tr_k_up_proj,v_up_proj,atten_sum,atten_gen,o_proj,ffn,expert_ffn,communication,rope,layernorm,"
                << "residual,act_energy,read_energy,write_energy,all_act_energy,all_read_energy,all_write_energy,"
                << "mac_energy,total_energy,fc_dram,fc_comp,attn_dram,attn_comp,moe_dram,moe_comp,OOM\n";

    time_ns cumulative_device_time = 0;

    time_ns req_start_time = device->status.device_time;
    for (int s = 0; s < max_batch_size; s++) {
      PrefetchMetricsCollector::getInstance().record_request_start(
          0, s, input_len, 0.0, static_cast<double>(req_start_time));
    }

    for (int step = 0; step < iter; step++) {
      auto metadata = scheduler->setMetadata();
      if (metadata.empty()) {
        std::cerr << "[WARN] No active sequences in metadata at step " << step << ". Terminating early.\n";
        break;
      }

      time_ns step_start_dev_time = device->status.device_time;
      for (int s = 0; s < max_batch_size; s++) {
        PrefetchMetricsCollector::getInstance().record_token_start(
            0, s, input_len + step, step, static_cast<double>(step_start_dev_time));
      }

      auto step_host_start = std::chrono::high_resolution_clock::now();
      double ddr5_sim_before = 0.0;
      if (device->offchip_dram_interface != nullptr) {
        ddr5_sim_before = device->offchip_dram_interface->time;
      }
      double hbm_sim_before = 0.0;
      if (device->dram_interface != nullptr) {
        hbm_sim_before = device->dram_interface->time;
      }

      cluster->run(metadata);

      time_ns step_end_dev_time = device->status.device_time;
      for (int s = 0; s < max_batch_size; s++) {
        PrefetchMetricsCollector::getInstance().record_token_end(
            0, s, input_len + step, step, static_cast<double>(step_end_dev_time));
      }

      auto step_host_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> step_host_sec = step_host_end - step_host_start;

      double ddr5_sim_after = 0.0;
      if (device->offchip_dram_interface != nullptr) {
        ddr5_sim_after = device->offchip_dram_interface->time;
      }
      double hbm_sim_after = 0.0;
      if (device->dram_interface != nullptr) {
        hbm_sim_after = device->dram_interface->time;
      }
      double ddr5_delta_ns = ddr5_sim_after - ddr5_sim_before;
      double hbm_delta_ns = hbm_sim_after - hbm_sim_before;
      time_ns step_device_time = device->status.device_time - step_start_dev_time;
      if (system_config.clock_continuity) {
        cumulative_device_time = device->status.device_time;
      } else {
        cumulative_device_time += step_device_time;
      }

      // Populate standard Stat object with measured simulation breakdowns and energies
      Stat stat;
      stat.iter_info = 1;
      stat.type = "t2t";
      stat.time = cumulative_device_time;
      stat.latency = step_device_time;
      stat.input_len = input_len;
      stat.output_len = output_len;
      stat.average_seq_len = input_len + step + 1;
      stat.batchsize = max_batch_size;
      stat.process_token = max_batch_size;
      stat.seq_queue_size = scheduler->sequence_queue.size();

      std::vector<energy_nJ> total_energy = cluster->getTotalEnergy();
      stat.act_energy = total_energy[0];
      stat.read_energy = total_energy[1];
      stat.write_energy = total_energy[2];
      stat.all_act_energy = total_energy[3];
      stat.all_read_energy = total_energy[4];
      stat.all_write_energy = total_energy[5];
      stat.mac_energy = total_energy[6];
      stat.total_energy = total_energy[7];

      cluster->setStat(stat);
      stat.latency = step_device_time;
      stat.time = cumulative_device_time;
      cluster->setTimeBreakDown(stat);

      std::cout << "  [Step " << step << "] Token SeqLen=" << (input_len + step) 
                << " | Host: " << std::fixed << std::setprecision(2) << step_host_sec.count() << "s"
                << " | Simulated DDR5: " << (ddr5_delta_ns / 1e6) << " ms"
                << " | Resident Experts: " << device->lru_expert_list.size()
                << " (" << (device->resident_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB)\n" << std::flush;

      // Advance scheduler for the next sequential decoding token
      scheduler->updateScheduler(step_device_time);
      if (!is_real_trace_data) {
        scheduler->fillSequenceQueue(step_device_time, cumulative_device_time);
      }
      scheduler->fillRunningQueue();
      if (is_real_trace_data && scheduler->running_queue.empty() && step + 1 < iter) {
        throw std::runtime_error("[ERROR] Running queue unexpectedly empty during real trace replay at step " + std::to_string(step));
      }

      // Export step row to summary CSV using genuine Stat values
      summary_csv << stat.iter_info << ","
                  << stat.split << ","
                  << stat.type << ","
                  << stat.time << ","
                  << stat.latency << ","
                  << stat.queueing_delay << ","
                  << stat.arrival_time << ","
                  << stat.seq_queue_size << ","
                  << stat.input_len << ","
                  << stat.output_len << ","
                  << stat.num_sum_iter << ","
                  << stat.is_mixed << ","
                  << stat.batchsize << ","
                  << stat.process_token << ","
                  << stat.sum_seq << ","
                  << stat.gen_seq << ","
                  << stat.average_seq_len << ","
                  << stat.sum_attention_opb << ","
                  << stat.qkv_gen << ","
                  << stat.q_down_proj << ","
                  << stat.kv_down_proj << ","
                  << stat.kr_proj << ","
                  << stat.q_up_proj << ","
                  << stat.qr_proj << ","
                  << stat.kv_up_proj << ","
                  << stat.tr_k_up_proj << ","
                  << stat.v_up_proj << ","
                  << stat.atten_sum << ","
                  << stat.atten_gen << ","
                  << stat.o_proj << ","
                  << stat.ffn << ","
                  << stat.expert_ffn << ","
                  << stat.communication << ","
                  << stat.rope << ","
                  << stat.layernorm << ","
                  << stat.residual << ","
                  << stat.act_energy << ","
                  << stat.read_energy << ","
                  << stat.write_energy << ","
                  << stat.all_act_energy << ","
                  << stat.all_read_energy << ","
                  << stat.all_write_energy << ","
                  << stat.mac_energy << ","
                  << stat.total_energy << ","
                  << stat.FC_DRAM_energy << ","
                  << stat.FC_COMP_energy << ","
                  << stat.Attn_DRAM_energy << ","
                  << stat.Attn_COMP_energy << ","
                  << stat.MoE_DRAM_energy << ","
                  << stat.MoE_COMP_energy << ","
                  << stat.isOOM << "\n";
      summary_csv.flush();
    }
    summary_csv.close();

    time_ns req_end_time = device->status.device_time;
    for (int s = 0; s < max_batch_size; s++) {
      PrefetchMetricsCollector::getInstance().record_request_end(
          0, s, iter, static_cast<double>(req_end_time));
    }
  }

  auto run_host_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_host_sec = run_host_end - run_host_start;

  if (scheduler != nullptr && scheduler->prediction_engine != nullptr) {
    scheduler->prediction_engine->finalize(total_host_sec.count());
  }

  if (device != nullptr) {
    device->dump_prefetch_effectiveness(trace_dir + "/prefetch_effectiveness.csv");
    device->dump_fa_lru_diagnostics(trace_dir + "/fa_lru_diagnostics.json");
    PrefetchMetricsCollector::getInstance().set_hbm_metrics(
        device->cache_capacity_bytes, to_string(device->eviction_policy),
        device->fa_lru_window_size, device->reloaded_experts_count);
    PrefetchMetricsCollector::getInstance().finalize(static_cast<double>(device->get_time()), 0);
    PrefetchMetricsCollector::getInstance().dump_artifacts();
  }

  if (export_gantt) {
    cluster->exportGantt(gantt_dir);
  }

  size_t routing_lines = count_file_lines(trace_dir + "/expert_routing.csv");
  size_t load_lines = count_file_lines(trace_dir + "/expert_load.csv");
  size_t cache_lines = count_file_lines(trace_dir + "/expert_cache.csv");

  std::cout << "\n======================================================================\n";
  std::cout << "EXECUTION COMPLETE\n";
  std::cout << "  Total Host Runtime: " << total_host_sec.count() << " seconds\n";
  std::cout << "  Final Resident Experts: " << device->lru_expert_list.size() << "\n";
  std::cout << "  Final Resident Memory:  " << device->resident_bytes << " bytes (" 
            << (device->resident_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB)\n";
  std::cout << "  Routing Records: " << routing_lines << " lines\n";
  std::cout << "  Load Records:    " << load_lines << " lines\n";
  std::cout << "  Cache Records:   " << cache_lines << " lines\n";
  std::cout << "======================================================================\n" << std::flush;

  // Write machine-readable completion status
  std::string status_path = output_path + "/../logs/completion_status.txt";
  std::ofstream status_file(status_path);
  if (status_file.is_open()) {
    status_file << "STATUS=SUCCESS\n"
                << "EXIT_CODE=0\n"
                << "MODEL=deepseekV3\n"
                << "CONFIG_PATH=" << config_path << "\n"
                << "ITERATIONS=" << iter << "\n"
                << "BATCH_SIZE=" << max_batch_size << "\n"
                << "PRECISION_BYTE=" << precision_byte << "\n"
                << "TOTAL_HOST_SECONDS=" << std::fixed << std::setprecision(4) << total_host_sec.count() << "\n"
                << "RESIDENT_EXPERTS=" << device->lru_expert_list.size() << "\n"
                << "RESIDENT_BYTES=" << device->resident_bytes << "\n"
                << "ROUTING_RECORDS=" << routing_lines << "\n"
                << "LOAD_RECORDS=" << load_lines << "\n"
                << "CACHE_RECORDS=" << cache_lines << "\n"
                << "CANDIDATES_EVALUATED=" << device->prefetch_metrics.candidates_evaluated << "\n"
                << "CANDIDATES_DISPATCHED=" << device->prefetch_metrics.candidates_dispatched << "\n"
                << "PREFETCH_TRANSFERS_SUBMITTED=" << device->prefetch_metrics.prefetch_transfers_submitted << "\n"
                << "PREFETCH_TRANSFERS_COMPLETED=" << device->prefetch_metrics.prefetch_transfers_completed << "\n"
                << "PREFETCHES_INSERTED_HBM=" << device->prefetch_metrics.prefetches_inserted_hbm << "\n"
                << "RESIDENT_PREFETCHES_SKIPPED=" << device->prefetch_metrics.resident_prefetches_skipped << "\n"
                << "DUPLICATE_PREFETCHES_PREVENTED=" << device->prefetch_metrics.duplicate_prefetches_prevented << "\n"
                << "PREFETCH_USEFUL_HITS=" << device->prefetch_metrics.prefetch_useful_hits << "\n"
                << "DEMAND_INFLIGHT_ATTACHMENTS=" << device->prefetch_metrics.demand_inflight_attachments << "\n"
                << "SPECULATIVE_INSERTIONS=" << device->pabi_metrics.speculative_insertions << "\n"
                << "SPECULATIVE_PROMOTIONS=" << device->pabi_metrics.speculative_promotions << "\n"
                << "SPECULATIVE_EVICTIONS=" << device->pabi_metrics.speculative_evictions << "\n"
                << "UNPINNED_SPECULATIVE_EVICTIONS=" << device->pabi_metrics.unpinned_speculative_evictions << "\n"
                << "PINNED_EVICTION_SKIPPED=" << device->pabi_metrics.pinned_eviction_skipped << "\n"
                << "PINS_CLEARED_ON_PROMOTION=" << device->pabi_metrics.pins_cleared_on_promotion << "\n"
                << "PIN_CANDIDATES_CONSIDERED=" << device->pabi_metrics.pin_candidates_considered << "\n"
                << "PIN_CANDIDATES_SELECTED=" << device->pabi_metrics.pin_candidates_selected << "\n"
                << "PIN_INTENTS_CREATED=" << device->pabi_metrics.pin_intents_created << "\n"
                << "PIN_EXPIRATIONS=" << device->pabi_metrics.pin_expirations << "\n"
                << "RESIDENT_EXPERTS_PINNED=" << device->pabi_metrics.resident_experts_pinned << "\n"
                << "CANCELLED_PREFETCHES=" << device->prefetch_metrics.cancelled_prefetches << "\n"
                << "DEMAND_INSERTIONS=" << device->pabi_metrics.demand_insertions << "\n"
                << "DEMAND_EVICTIONS=" << device->pabi_metrics.demand_evictions << "\n"
                << "PINNED_EXPERT_DEMAND_HITS=" << device->pabi_metrics.pinned_expert_demand_hits << "\n"
                << "SPECULATIVE_REJECT_PINNED_CAPACITY=" << device->pabi_metrics.speculative_reject_pinned_capacity << "\n";
    status_file.close();
  }

  return 0;
}
