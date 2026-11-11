#include "hardware/device.h"

#include <random>
#include <string>
#include <stdexcept>

#include "common/assert.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "dram/mmap_controller.h"
#include "dram/pimkernel/pim_kernel.h"
#include "module/module_graph.h"
#include "module/tensor.h"
#include "prefetch/prefetch_candidate_queue.h"
#include "prefetch/prefetch_metrics_collector.h"

namespace llm_system {

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dis(0, 5);

Device::Device(SystemConfig config, int device_total_rank, Cluster_ptr cluster)
    : config(config),
      device_total_rank(device_total_rank),
      cluster(cluster),
      status() {
  compute_peak_flops = config.compute_peak_flops;
  if (config.compute_time_multiplier > 0.0 && config.compute_time_multiplier != 1.0) {
    this->config.compute_peak_flops = this->config.compute_peak_flops / config.compute_time_multiplier;
    compute_peak_flops = this->config.compute_peak_flops;
  }
  memory_bandwidth = config.memory_bandwidth;
  memory_capacity = config.memory_capacity;
  device_local_rank = device_total_rank % config.num_device;

  top_module_graph = TopModuleGraph::Create(status);

  std::string dram_cfg_path;
  if (!config.hbm_dram_cfg_path.empty()) {
    dram_cfg_path = config.hbm_dram_cfg_path;
  }
  else if(config.gpu_gen == "H100" || config.gpu_gen == "A100"){
    dram_cfg_path = "./dram_config_HBM3_80GB.yaml";
  }
  else if(config.gpu_gen == "B100" || config.gpu_gen == "B200"){
    dram_cfg_path = "./dram_config_HBM3E_192GB.yaml";
  }
  YAML::Node cfg = YAML::LoadFile(dram_cfg_path);

  double memory_scale_factor = 0;
  MemoryConfig memory_config = MemoryConfig(config.num_cube, config.num_logic_cube);
  if(config.gpu_gen == "H100" || config.gpu_gen == "A100"){
    // H100, HBM3 80GB, 5.2Gbps
    memory_scale_factor = 0.76923;
    memory_config = hbm3_80GB;
    memory_config.num_cube = config.num_cube;
    memory_config.num_logic_cube = config.num_logic_cube;
  }
  else if((config.gpu_gen == "B100") || (config.gpu_gen == "B200")){
    // B100, HBM3E 192GB, 8Gbps
    memory_scale_factor = 0.5; // 8.0Gbps pin rate's ideal bandwidth = 8000, tCK = 2000, 1 / 2GHz = 0.5
    memory_config = hbm3e_192GB;
    memory_config.num_cube = config.num_cube;
    memory_config.num_logic_cube = config.num_logic_cube;
  }

  dram_interface = DRAMInterface::Create(dram_cfg_path, memory_scale_factor);
  mmap_controller = MMapController::Create(memory_config);
  // Phase 4: Instantiate second memory tier for off-chip DRAM.
  std::string offchip_cfg = config.offchip_dram_cfg_path;
  if (offchip_cfg.empty()) {
      offchip_cfg = dram_cfg_path; // fallback to HBM placeholder if not provided
  }
  double offchip_scale_factor = memory_scale_factor;
  MemoryConfig offchip_memory_config = memory_config;
  if (offchip_cfg.find("DDR5") != std::string::npos) {
      offchip_scale_factor = 0.625;
      offchip_memory_config = ddr5_32gb_x16_2ch;
  }
  offchip_dram_interface = DRAMInterface::Create(offchip_cfg, offchip_scale_factor);
  offchip_mmap_controller = MMapController::Create(offchip_memory_config);

  // Phase 4 Validation Only: Synthetic test request to Ramulator B (Disabled for production)
  /*
  std::cout << "[PHASE 4 VALIDATION] Submitting synthetic read to Off-chip DRAM (Ramulator B)..." << std::endl;
  auto mem_obj = MemoryObject::Create(MMap::ALL_CHANNEL, 0, 64, offchip_mmap_controller);
  auto req = DRAMRequest::Create(DRAMRequestType::kRead);
  req->AddOperand(mem_obj, PIMOperandType::kDRAM);
  std::list<DRAMRequest::Ptr> req_list;
  req_list.push_back(req);
  offchip_dram_interface->HandleRequest(req_list, 0);
  std::cout << "[PHASE 4 VALIDATION] Synthetic request completed." << std::endl;
  std::cout << "[PHASE 4 VALIDATION] Off-chip Memory duration: " 
            << offchip_dram_interface->getExecStatus().memory_duration << " ns" << std::endl;
  */

  use_ramulator = config.use_ramulator;
  perform_execution = false;

  const char* debug_cap = std::getenv("DEBUG_CACHE_CAPACITY");
  if (debug_cap != nullptr) {
    try {
      cache_capacity_bytes = std::stoll(debug_cap);
    } catch (...) {
    }
  }

  eviction_policy = config.prefetch_config.eviction_policy;
  fa_lru_window_size = config.prefetch_config.fa_lru_window_size;
  pabi_quarantine_enabled = config.prefetch_config.pabi_quarantine_enabled;
  if (config.prefetch_config.pabi_speculative_buffer_capacity_bytes > 0) {
    speculative_buffer_capacity_bytes = config.prefetch_config.pabi_speculative_buffer_capacity_bytes;
  } else if (config.prefetch_config.pabi_speculative_fraction > 0.0) {
    speculative_buffer_capacity_bytes = static_cast<long long>(cache_capacity_bytes * config.prefetch_config.pabi_speculative_fraction);
  } else {
    speculative_buffer_capacity_bytes = static_cast<long long>(cache_capacity_bytes * 0.20);
  }

  const char* env_eviction = std::getenv("EVICTION_POLICY");
  if (env_eviction != nullptr) {
    try {
      eviction_policy = parse_eviction_policy(env_eviction);
    } catch (...) {
    }
  }

  const char* env_window = std::getenv("FA_LRU_WINDOW_SIZE");
  if (env_window != nullptr) {
    try {
      int w = std::stoi(env_window);
      if (w >= 1) {
        fa_lru_window_size = w;
      }
    } catch (...) {
    }
  }

  const char* env_pabi_quar = std::getenv("PABI_QUARANTINE_ENABLED");
  if (env_pabi_quar != nullptr) {
    std::string s(env_pabi_quar);
    if (s == "1" || s == "true" || s == "on" || s == "yes") {
      pabi_quarantine_enabled = true;
    } else if (s == "0" || s == "false" || s == "off" || s == "no") {
      pabi_quarantine_enabled = false;
    }
  }

  const char* env_spec_cap = std::getenv("PABI_SPECULATIVE_BUFFER_CAPACITY_BYTES");
  if (env_spec_cap != nullptr) {
    try {
      speculative_buffer_capacity_bytes = std::stoll(env_spec_cap);
    } catch (...) {
    }
  }

  pinning_enabled = config.prefetch_config.pabi_pinning_enabled;
  max_pinned_experts = config.prefetch_config.pabi_max_pinned_experts;
  pin_selection_policy = config.prefetch_config.pabi_pin_selection_policy;
  pin_expiration_policy = config.prefetch_config.pabi_pin_expiration_policy;

  const char* env_pinning = std::getenv("PABI_PINNING_ENABLED");
  if (env_pinning != nullptr) {
    std::string s(env_pinning);
    if (s == "1" || s == "true" || s == "on" || s == "yes") {
      pinning_enabled = true;
    } else if (s == "0" || s == "false" || s == "off" || s == "no") {
      pinning_enabled = false;
    }
  }

  const char* env_max_pinned = std::getenv("PABI_MAX_PINNED_EXPERTS");
  if (env_max_pinned != nullptr) {
    try {
      max_pinned_experts = std::stoi(env_max_pinned);
    } catch (...) {
    }
  }

  if (config.use_event_driven_dram) {
    double bw = config.ddr5_bandwidth_gbps;
    if (bw <= 0.0) {
      bw = DDR5TransferEngine::DEFAULT_BANDWIDTH_BYTES_PER_NS;
    }
    ddr5_transfer_engine = std::make_shared<DDR5TransferEngine>(bw);
  }
}

void Device::set_dependency() { top_module_graph->set_dependency(); }

bool Device::check_module_graph_remain() {
  return top_module_graph->check_module_graph_remain();
};

void Device::run(std::vector<BatchedSequence::Ptr> sequences_metadata_list) {
  int dp_rank = device_total_rank / model_config.ne_tp_dg;
  top_module_graph->run(sequences_metadata_list.at(dp_rank));
}

void Device::restartGraph() { top_module_graph->restart_graph(); }

void Device::connectTopModuleGraph() {
  top_module_graph->connectDevice(get_ptr());
}

void Device::reset_status() { status = StatusBoard(); }
void Device::reset_timeboard() { top_module_graph->reset_timeboard(); }

void Device::add_module(std::string name, Module_ptr module) {
  cluster->add_module(device_total_rank, name, module);
}

void Device::setMemoryObject(Tensor::Ptr tensor) {
  if (tensor->weight_target == MemoryTarget::OFFCHIP_DRAM && offchip_mmap_controller != nullptr) {
    offchip_mmap_controller->setMemoryObject(tensor);
  } else {
    mmap_controller->setMemoryObject(tensor);
  }
}

void Device::addExecutionCache(ExecStatus& exec_status, CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  cache.emplace(key, exec_status);
}

void Device::addExecutionCache(ExecStatus& exec_status, LayerType layer_type,
                               ProcessorType processor_type,
                               DRAMRequestType dram_reqeust_type, long size, MemoryTarget target) {
  CacheKey key =
      std::make_tuple(layer_type, processor_type, dram_reqeust_type, size, target);
  auto& cache = cluster->execution_time_cache;
  cache.emplace(key, exec_status);
}

bool Device::checkExecutionCache(ExecStatus& exec_status, CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  if (const auto& cache_iter = cache.find(key); cache_iter != cache.end()) {
    exec_status = (*cache_iter).second;
    return true;
  } else {
    return false;
  }
}

bool Device::checkExecutionCache(CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  if (const auto& cache_iter = cache.find(key); cache_iter != cache.end()) {
    ExecStatus exec_status = (*cache_iter).second;
    ExecStatus& _status = dram_interface->getExecStatus();
    _status = exec_status;
    return true;
  } else {
    return false;
  }
}

void Device::setExecStatus(ExecStatus& exec_status_) {
  exec_status = exec_status_;
}

ExecStatus Device::getHighExecStatus() {
  ExecStatus return_status = high_exec_status;
  high_exec_status = ExecStatus();
  return return_status;
}

ExecStatus Device::getLowExecStatus() {
  ExecStatus return_status = low_exec_status;
  low_exec_status = ExecStatus();
  return return_status;
}

ExecStatus Device::getExecStatus() {
  ExecStatus return_status = exec_status;
  exec_status = ExecStatus();
  return return_status;
}

// check whether execution must be performed, and ramulator
void Device::execution(LayerType layer_type,
                       const std::vector<Tensor::Ptr>& tensor_list,
                       const BatchedSequence::Ptr sequences_metadata,
                       const LayerInfo layer_info) {
  if (perform_execution) {
    dram_interface->resetCounter();
    cluster->executor.execution(layer_type, tensor_list, sequences_metadata,
                                config.processor_type, layer_info,
                                use_ramulator, get_ptr());
  }
}

void Device::execution_ramulator(LayerType layer_type,
                                 std::vector<Tensor::Ptr> tensor_list) {
  Tensor::Ptr input = tensor_list.at(0);
  Tensor::Ptr weight = tensor_list.at(1);
  Tensor::Ptr output = tensor_list.at(2);

  std::cout << weight->name << std::endl;
}

void Device::run_ramulator(DRAMRequest_Ptr dram_request, MemoryTarget target) {
  std::list<DRAMRequest::Ptr> request;
  request.push_back(dram_request);
  if (target == MemoryTarget::OFFCHIP_DRAM && offchip_dram_interface != nullptr) {
    offchip_dram_interface->HandleRequest(request, 0);
  } else {
    dram_interface->HandleRequest(request, 0);
  }
}

void Device::run_ramulator_batch(const std::list<DRAMRequest_Ptr>& requests, MemoryTarget target) {
  if (target == MemoryTarget::OFFCHIP_DRAM && offchip_dram_interface != nullptr) {
    offchip_dram_interface->HandleRequest(requests, 0);
  } else {
    dram_interface->HandleRequest(requests, 0);
  }
}

void Device::run_ideal(DRAMRequestType dram_request_type, Tensor_Ptr tensor){
  long total_size = tensor->getSize(); // Byte
  if (total_size == 0) {
    return;
  }
  MemoryConfig memory_config = mmap_controller->getConfig();
  int num_cube = memory_config.num_cube;
  int num_channel = memory_config.num_channel; // 32 (not legacy, pCH)
  int num_col = memory_config.num_col;
  int granul = mmap_controller->getGranul();

  long total_read = total_size / granul;
  long rw_cmd_to_cube_0 = 0;
  if (total_read % num_cube == 0) {
    rw_cmd_to_cube_0 = total_read / num_cube;
  } else {
    rw_cmd_to_cube_0 = (total_read / num_cube) + 1;
  }

  long rw_cmd_to_pCH_0 = 0;
  if (rw_cmd_to_cube_0 % num_channel == 0) {
    rw_cmd_to_pCH_0 = rw_cmd_to_cube_0 / num_channel;
  } else {
    rw_cmd_to_pCH_0 = (rw_cmd_to_cube_0 / num_channel) + 1;
  }

  long rw_cmd_to_pCH_1 = 0;
  if (rw_cmd_to_cube_0 % num_channel == 1) {
    rw_cmd_to_pCH_1 = rw_cmd_to_cube_0 / num_channel;
  } else {
    rw_cmd_to_pCH_1 = (rw_cmd_to_cube_0 / num_channel) + 1;
  }

  dram_interface->resetCounter();
  long pch_sum = rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1;
  if (pch_sum < num_col) {
    dram_interface->getExecStatus().act_count = 1;
  } else {
    dram_interface->getExecStatus().act_count = pch_sum / num_col;
  }
  if(dram_request_type == DRAMRequestType::kRead){
    dram_interface->getExecStatus().read_count = (rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1);
  }
  else if(dram_request_type == DRAMRequestType::kWrite){
    dram_interface->getExecStatus().write_count = (rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1);
  }
}

void Device::initializeDRAM(int ProcessorType, DramEnergy dramEnergy) {
  int num_pseudo_ch = 0;
  if (ProcessorType == (int)(ProcessorType::GPU)) {
    num_pseudo_ch = mmap_controller->getConfig().num_cube * mmap_controller->getConfig().num_channel / 2;
  }
  else {
    num_pseudo_ch = mmap_controller->getConfig().num_cube * mmap_controller->getConfig().num_channel;
  }
  dramEnergy.kACT_energy_j_ *= num_pseudo_ch;
  dramEnergy.kREAD_energy_j_ *= num_pseudo_ch;
  dramEnergy.kWRITE_energy_j_ *= num_pseudo_ch;

  dramEnergy.kALL_ACT_energy_j_ *= num_pseudo_ch;
  dramEnergy.kALL_READ_energy_j_ *= num_pseudo_ch;
  dramEnergy.kALL_WRITE_energy_j_ *= num_pseudo_ch;

  top_module_graph->initializeDRAM(ProcessorType, dramEnergy);
}

static void log_cache_event(int layer_id, int expert_id, const std::string& action, long long bytes) {
  static bool init = false;
  static std::ofstream csv;
  if (!init) {
    const char* trace_dir_env = std::getenv("TRACE_DIR");
    std::string path = "expert_cache.csv";
    if (trace_dir_env != nullptr) {
      path = std::string(trace_dir_env) + "/expert_cache.csv";
    }
    csv.open(path, std::ios::app);
    csv.seekp(0, std::ios::end);
    if (csv.tellp() == 0) {
      csv << "layer_id,expert_id,action,resident_bytes\n";
    }
    init = true;
  }
  if (csv.is_open()) {
    csv << layer_id << "," << expert_id << "," << action << "," << bytes << "\n";
    csv.flush();
  }
}

bool Device::is_pinning_active() const {
  if (eviction_policy != EvictionPolicy::PABI) {
    return false;
  }
  if (!pabi_quarantine_enabled) {
    return false;
  }
  if (!pinning_enabled) {
    return false;
  }
  if (max_pinned_experts <= 0) {
    return false;
  }
  return true;
}

std::vector<PrefetchCandidate> Device::select_pin_candidates_for_stage(
    int target_stage_id,
    const std::vector<PrefetchCandidate>& candidates) {
  if (!is_pinning_active()) {
    return {};
  }

  // 1. Filter eligible candidates
  std::vector<PrefetchCandidate> eligible;
  for (const auto& cand : candidates) {
    pabi_metrics.pin_candidates_considered = pabi_metrics.pin_candidates_considered + 1;
    if (cand.target_stage_id < 0) {
      continue;
    }
    if (cand.expert_id < 0) {
      continue;
    }
    if (cand.state == CandidateState::EXPIRED_MISS || cand.state == CandidateState::CANCELLED || cand.state == CandidateState::EVICTED) {
      continue;
    }
    std::pair<int, int> key = {cand.target_stage_id, cand.expert_id};
    auto it = resident_experts.find(key);
    if (it != resident_experts.end()) {
      if (it->second.classification == ExpertClassification::DEMAND_VERIFIED) {
        continue;
      }
    }
    eligible.push_back(cand);
  }

  // 2. Deterministic sort: Rank ascending, then Score descending, then Expert ID ascending
  std::sort(eligible.begin(), eligible.end(), [](const PrefetchCandidate& a, const PrefetchCandidate& b) {
    if (a.prediction_rank != b.prediction_rank) {
      return a.prediction_rank < b.prediction_rank;
    }
    if (std::abs(a.prediction_score - b.prediction_score) > 1e-9) {
      return a.prediction_score > b.prediction_score;
    }
    return a.expert_id < b.expert_id;
  });

  // 3. Deduplicate by expert_id within this stage (best rank/score wins)
  std::vector<PrefetchCandidate> deduplicated;
  std::unordered_set<int> seen_experts;
  for (const auto& cand : eligible) {
    if (seen_experts.find(cand.expert_id) == seen_experts.end()) {
      seen_experts.insert(cand.expert_id);
      deduplicated.push_back(cand);
    }
  }

  // 4. Budget selection up to max_pinned_experts
  std::vector<PrefetchCandidate> selected;
  int count = 0;
  for (const auto& cand : deduplicated) {
    if (count >= max_pinned_experts) {
      break;
    }
    selected.push_back(cand);
    count = count + 1;
    pabi_metrics.pin_candidates_selected = pabi_metrics.pin_candidates_selected + 1;

    std::pair<int, int> key = {cand.target_stage_id, cand.expert_id};

    PinIntent intent;
    intent.layer_id = cand.target_stage_id;
    intent.expert_id = cand.expert_id;
    intent.target_stage_id = target_stage_id;
    intent.expiration_stage_id = target_stage_id;
    intent.prediction_rank = cand.prediction_rank;
    intent.prediction_score = cand.prediction_score;
    intent.candidate_id = cand.candidate_id;
    intent.is_active = true;

    auto existing_it = active_pin_intents.find(key);
    if (existing_it != active_pin_intents.end()) {
      if (existing_it->second.expiration_stage_id > intent.expiration_stage_id) {
        intent.expiration_stage_id = existing_it->second.expiration_stage_id;
      }
      if (existing_it->second.prediction_score > intent.prediction_score) {
        intent.prediction_score = existing_it->second.prediction_score;
        intent.prediction_rank = existing_it->second.prediction_rank;
      }
    }

    active_pin_intents[key] = intent;
    pabi_metrics.pin_intents_created = pabi_metrics.pin_intents_created + 1;
    log_cache_event(cand.target_stage_id, cand.expert_id, "PIN_INTENT_CREATED", cand.prediction_rank);

    // If expert is already resident as speculative, apply pin immediately
    auto res_it = resident_experts.find(key);
    if (res_it != resident_experts.end()) {
      if (res_it->second.classification == ExpertClassification::SPECULATIVE) {
        res_it->second.is_pinned = true;
        res_it->second.pin_target_stage_id = target_stage_id;
        int exp_stage = target_stage_id;
        if (res_it->second.pin_expiration_stage_id > exp_stage) {
          exp_stage = res_it->second.pin_expiration_stage_id;
        }
        res_it->second.pin_expiration_stage_id = exp_stage;
        res_it->second.prediction_rank = cand.prediction_rank;
        res_it->second.prediction_score = cand.prediction_score;
        res_it->second.candidate_id = cand.candidate_id;
        pabi_metrics.resident_experts_pinned = pabi_metrics.resident_experts_pinned + 1;
        log_cache_event(cand.target_stage_id, cand.expert_id, "PIN_APPLIED", cand.prediction_rank);
      }
    }

    // If expert is already in flight, attach pin intent
    auto inf_it = inflight_registry.find(key);
    if (inf_it != inflight_registry.end()) {
      if (inf_it->second.transfer_class == TransferClass::PREFETCH) {
        inf_it->second.has_pin_intent = true;
        inf_it->second.pin_target_stage_id = target_stage_id;
        int exp_stage = target_stage_id;
        if (inf_it->second.pin_expiration_stage_id > exp_stage) {
          exp_stage = inf_it->second.pin_expiration_stage_id;
        }
        inf_it->second.pin_expiration_stage_id = exp_stage;
        inf_it->second.prediction_rank = cand.prediction_rank;
        inf_it->second.prediction_score = cand.prediction_score;
        inf_it->second.candidate_id = cand.candidate_id;
        pabi_metrics.inflight_pin_intents = pabi_metrics.inflight_pin_intents + 1;
      }
    }
  }

  return selected;
}

int Device::expire_pins_for_stage(int target_stage_id) {
  int expired_count = 0;

  // 1. Resident speculative experts
  for (auto& kv : resident_experts) {
    if (kv.second.classification == ExpertClassification::SPECULATIVE) {
      if (kv.second.is_pinned && kv.second.pin_expiration_stage_id <= target_stage_id) {
        kv.second.is_pinned = false;
        kv.second.pin_target_stage_id = -1;
        kv.second.pin_expiration_stage_id = -1;
        pabi_metrics.pin_expirations = pabi_metrics.pin_expirations + 1;
        log_cache_event(kv.first.first, kv.first.second, "PIN_EXPIRED", target_stage_id);
        expired_count = expired_count + 1;
      }
      if (kv.second.is_prefetched && !kv.second.was_demanded && kv.second.pin_target_stage_id <= target_stage_id) {
        if (kv.second.transfer_id != 0) {
          auto tr_it = speculative_transfers.find(kv.second.transfer_id);
          if (tr_it != speculative_transfers.end()) {
            if (tr_it->second.outcome == SpeculativeTransferOutcome::UNDECIDED) {
              tr_it->second.outcome = SpeculativeTransferOutcome::UNUSED;
              prefetch_metrics.unused_transfers = prefetch_metrics.unused_transfers + 1;
            }
          }
        }
      }
    }
  }

  // 2. In-flight transfers
  for (auto& kv : inflight_registry) {
    if (kv.second.has_pin_intent && kv.second.pin_expiration_stage_id <= target_stage_id) {
      kv.second.has_pin_intent = false;
      kv.second.pin_target_stage_id = -1;
      kv.second.pin_expiration_stage_id = -1;
      pabi_metrics.pin_expirations = pabi_metrics.pin_expirations + 1;
      log_cache_event(kv.first.first, kv.first.second, "PIN_EXPIRED", target_stage_id);
    }
  }

  // 3. Active pin intents
  for (auto it = active_pin_intents.begin(); it != active_pin_intents.end();) {
    if (it->second.expiration_stage_id <= target_stage_id) {
      it = active_pin_intents.erase(it);
    } else {
      ++it;
    }
  }

  return expired_count;
}

int Device::cleanup_unexecuted_stage_pins(int stage_id) {
  return expire_pins_for_stage(stage_id);
}

bool Device::promote_speculative_to_demand(int layer_id, int expert_id) {
  std::pair<int, int> key = {layer_id, expert_id};
  auto it = resident_experts.find(key);
  if (it == resident_experts.end()) {
    return false;
  }
  if (it->second.classification != ExpertClassification::SPECULATIVE) {
    return false;
  }

  // Phase 4D.4: Clear pin metadata if pinned
  if (it->second.is_pinned) {
    it->second.is_pinned = false;
    it->second.pin_target_stage_id = -1;
    it->second.pin_expiration_stage_id = -1;
    pabi_metrics.pins_cleared_on_promotion = pabi_metrics.pins_cleared_on_promotion + 1;
    pabi_metrics.pinned_expert_demand_hits = pabi_metrics.pinned_expert_demand_hits + 1;
    log_cache_event(layer_id, expert_id, "PIN_CLEARED_ON_PROMOTION", 0);
  }

  auto pin_it = active_pin_intents.find(key);
  if (pin_it != active_pin_intents.end()) {
    active_pin_intents.erase(pin_it);
  }

  long long expert_size = it->second.size_bytes;
  speculative_lru_list.erase(it->second.list_it);

  demand_lru_list.push_back({layer_id, expert_id, expert_size});
  lru_expert_list.push_back({layer_id, expert_id, expert_size});

  it->second.classification = ExpertClassification::DEMAND_VERIFIED;
  it->second.list_it = std::prev(demand_lru_list.end());
  it->second.was_demanded = true;

  resident_speculative_bytes -= expert_size;
  resident_demand_bytes += expert_size;

  pabi_metrics.speculative_promotions = pabi_metrics.speculative_promotions + 1;
  pabi_metrics.useful_prefetch_hits = pabi_metrics.useful_prefetch_hits + 1;
  prefetch_metrics.prefetch_useful_hits = prefetch_metrics.prefetch_useful_hits + 1;
  prefetch_metrics.timely_resident_hits = prefetch_metrics.timely_resident_hits + 1;
  prefetch_metrics.demanded_transfers = prefetch_metrics.demanded_transfers + 1;
  prefetched_resident_experts.erase(key);

  if (it->second.transfer_id != 0) {
    auto tr_it = speculative_transfers.find(it->second.transfer_id);
    if (tr_it != speculative_transfers.end()) {
      tr_it->second.outcome = SpeculativeTransferOutcome::DEMANDED;
      tr_it->second.actual_demand_ns = get_time();
      tr_it->second.remaining_transfer_at_demand_ns = 0.0;
      tr_it->second.demand_arrival_to_ready_ns = 0.0;
    }
  }

  DemandOutcomeRecord dem_rec;
  dem_rec.layer_id = layer_id;
  dem_rec.expert_id = expert_id;
  dem_rec.demand_time_ns = get_time();
  dem_rec.remaining_transfer_ns = 0.0;
  dem_rec.demand_arrival_to_ready_ns = 0.0;
  dem_rec.outcome = DemandOutcome::TIMELY_RESIDENT;
  demand_outcomes.push_back(dem_rec);

  log_cache_event(layer_id, expert_id, "PROMOTION", resident_bytes);
  return true;
}

bool Device::has_resident_expert(int layer_id, int expert_id) const {
  std::pair<int, int> key = {layer_id, expert_id};
  return resident_experts.find(key) != resident_experts.end();
}

void Device::materialize_lazy_decay(ResidentExpertEntry& entry) {
  if (global_epoch <= entry.last_epoch) {
    return;
  }

  uint32_t epoch_diff = global_epoch - entry.last_epoch;
  if (epoch_diff >= 31) {
    entry.access_frequency = 1;
  } else {
    uint32_t decayed = entry.access_frequency >> epoch_diff;
    if (decayed < 1) {
      entry.access_frequency = 1;
    } else {
      entry.access_frequency = decayed;
    }
  }
  entry.last_epoch = global_epoch;
}

std::list<std::tuple<int, int, long long>>::iterator Device::select_fa_lru_victim() {
  if (lru_expert_list.empty()) {
    return lru_expert_list.end();
  }

  // Find what standard LRU would have chosen (first unpinned candidate)
  auto lru_victim_it = lru_expert_list.end();
  for (auto lit = lru_expert_list.begin(); lit != lru_expert_list.end(); ++lit) {
    int l = std::get<0>(*lit);
    int e = std::get<1>(*lit);
    auto mit = resident_experts.find({l, e});
    if (mit != resident_experts.end()) {
      if (mit->second.is_pinned) {
        continue;
      }
    }
    lru_victim_it = lit;
    break;
  }

  auto best_it = lru_expert_list.end();
  uint32_t min_freq = UINT32_MAX;
  int count = 0;
  int window_size = fa_lru_window_size;
  int list_size = static_cast<int>(lru_expert_list.size());
  if (window_size > list_size) {
    window_size = list_size;
  }

  // Pass 1: Scan bottom-W candidates for unpinned minimum-frequency victim
  for (auto it = lru_expert_list.begin(); it != lru_expert_list.end() && count < window_size; ++it, ++count) {
    int l = std::get<0>(*it);
    int e = std::get<1>(*it);
    auto map_it = resident_experts.find({l, e});
    if (map_it != resident_experts.end()) {
      if (map_it->second.is_pinned) {
        continue;
      }
      materialize_lazy_decay(map_it->second);
      uint32_t freq = map_it->second.access_frequency;
      fa_lru_diag.window_frequency_hist[freq] = fa_lru_diag.window_frequency_hist[freq] + 1;
      if (freq < min_freq) {
        min_freq = freq;
        best_it = it;
      }
    }
  }

  fa_lru_diag.total_eviction_decisions = fa_lru_diag.total_eviction_decisions + 1;
  fa_lru_diag.total_scanned_candidates = fa_lru_diag.total_scanned_candidates + count;
  if ((uint64_t)count > fa_lru_diag.max_scan_depth) {
    fa_lru_diag.max_scan_depth = count;
  }

  // Pass 2: Fallback if all candidates in bottom-W were pinned: search beyond window for first unpinned candidate
  if (best_it == lru_expert_list.end()) {
    fa_lru_diag.fallback_beyond_window_count = fa_lru_diag.fallback_beyond_window_count + 1;
    for (auto it = lru_expert_list.begin(); it != lru_expert_list.end(); ++it) {
      int l = std::get<0>(*it);
      int e = std::get<1>(*it);
      auto map_it = resident_experts.find({l, e});
      if (map_it != resident_experts.end()) {
        if (!map_it->second.is_pinned) {
          best_it = it;
          break;
        }
      }
    }
  }

  if (best_it != lru_expert_list.end()) {
    int bl = std::get<0>(*best_it);
    int be = std::get<1>(*best_it);
    auto bmap = resident_experts.find({bl, be});
    if (bmap != resident_experts.end()) {
      uint32_t bfreq = bmap->second.access_frequency;
      fa_lru_diag.victim_frequency_hist[bfreq] = fa_lru_diag.victim_frequency_hist[bfreq] + 1;
      if (bmap->second.is_prefetched && !bmap->second.was_demanded) {
        fa_lru_diag.unconsumed_speculative_victims = fa_lru_diag.unconsumed_speculative_victims + 1;
      }
    }

    if (best_it != lru_victim_it && lru_victim_it != lru_expert_list.end()) {
      fa_lru_diag.divergent_decisions = fa_lru_diag.divergent_decisions + 1;
      int ll = std::get<0>(*lru_victim_it);
      int le = std::get<1>(*lru_victim_it);
      auto lmap = resident_experts.find({ll, le});
      if (lmap != resident_experts.end()) {
        if (!lmap->second.is_prefetched || lmap->second.was_demanded) {
          fa_lru_diag.protected_demand_experts_count = fa_lru_diag.protected_demand_experts_count + 1;
          fa_lru_diag.protected_experts_set.insert({ll, le});
        }
      }
    }
  }

  return best_it;
}

void Device::dump_fa_lru_diagnostics(const std::string& filepath) {
  std::ofstream f(filepath);
  if (!f.is_open()) {
    return;
  }
  uint64_t total_decisions = fa_lru_diag.total_eviction_decisions;
  double avg_scan = 0.0;
  if (total_decisions > 0) {
    avg_scan = static_cast<double>(fa_lru_diag.total_scanned_candidates) / static_cast<double>(total_decisions);
  }
  uint64_t protected_reused = fa_lru_diag.protected_reused_set.size();
  uint64_t total_protected = fa_lru_diag.protected_experts_set.size();
  uint64_t protected_never_reused = 0;
  if (total_protected > protected_reused) {
    protected_never_reused = total_protected - protected_reused;
  }

  f << "{\n";
  f << "  \"total_eviction_decisions\": " << total_decisions << ",\n";
  f << "  \"divergent_decisions\": " << fa_lru_diag.divergent_decisions << ",\n";
  f << "  \"protected_demand_experts_count\": " << fa_lru_diag.protected_demand_experts_count << ",\n";
  f << "  \"unconsumed_speculative_victims\": " << fa_lru_diag.unconsumed_speculative_victims << ",\n";
  f << "  \"protected_experts_reused\": " << protected_reused << ",\n";
  f << "  \"protected_experts_never_reused\": " << protected_never_reused << ",\n";
  f << "  \"fallback_beyond_window_count\": " << fa_lru_diag.fallback_beyond_window_count << ",\n";
  f << "  \"average_candidate_window_scan_cost\": " << avg_scan << ",\n";
  f << "  \"max_candidate_window_scan_depth\": " << fa_lru_diag.max_scan_depth << ",\n";
  f << "  \"victim_frequency_distribution\": {\n";
  size_t v_idx = 0;
  for (const auto& kv : fa_lru_diag.victim_frequency_hist) {
    f << "    \"" << kv.first << "\": " << kv.second;
    v_idx = v_idx + 1;
    if (v_idx < fa_lru_diag.victim_frequency_hist.size()) {
      f << ",";
    }
    f << "\n";
  }
  f << "  },\n";
  f << "  \"window_frequency_distribution\": {\n";
  size_t w_idx = 0;
  for (const auto& kv : fa_lru_diag.window_frequency_hist) {
    f << "    \"" << kv.first << "\": " << kv.second;
    w_idx = w_idx + 1;
    if (w_idx < fa_lru_diag.window_frequency_hist.size()) {
      f << ",";
    }
    f << "\n";
  }
  f << "  }\n";
  f << "}\n";
}

bool Device::is_expert_resident(int layer_id, int expert_id, bool is_demand_access, bool is_primary_demand) {
  std::pair<int, int> key = {layer_id, expert_id};
  auto it = resident_experts.find(key);
  if (it == resident_experts.end()) {
    if (is_demand_access) {
      log_cache_event(layer_id, expert_id, "MISS", resident_bytes);
    }
    return false;
  }

  // Speculative probing does not promote, touch LRU, or record demand events
  if (!is_demand_access) {
    return true;
  }

  materialize_lazy_decay(it->second);
  if (is_primary_demand) {
    if (!it->second.was_demanded) {
      it->second.was_demanded = true;
      it->second.access_frequency = 2;
    } else {
      it->second.access_frequency = it->second.access_frequency + 1;
    }
    if (fa_lru_diag.protected_experts_set.count(key) > 0) {
      fa_lru_diag.protected_reused_set.insert(key);
    }
  }

  if (eviction_policy == EvictionPolicy::PABI && pabi_quarantine_enabled) {
    if (it->second.classification == ExpertClassification::SPECULATIVE) {
      promote_speculative_to_demand(layer_id, expert_id);
      return true;
    }

    // Hit on DEMAND_VERIFIED expert: Move to MRU of demand_lru_list
    auto node = *(it->second.list_it);
    demand_lru_list.erase(it->second.list_it);
    demand_lru_list.push_back(node);
    it->second.list_it = std::prev(demand_lru_list.end());

    for (auto lit = lru_expert_list.begin(); lit != lru_expert_list.end(); ++lit) {
      if (std::get<0>(*lit) == layer_id && std::get<1>(*lit) == expert_id) {
        auto lnode = *lit;
        lru_expert_list.erase(lit);
        lru_expert_list.push_back(lnode);
        break;
      }
    }

    log_cache_event(layer_id, expert_id, "HIT", resident_bytes);
    return true;
  }

  // Standard LRU and FA-LRU mode
  auto node = *(it->second.list_it);
  lru_expert_list.erase(it->second.list_it);
  lru_expert_list.push_back(node);
  it->second.list_it = std::prev(lru_expert_list.end());

  auto p_it = prefetched_resident_experts.find(key);
  if (p_it != prefetched_resident_experts.end()) {
    it->second.was_demanded = true;
    prefetch_metrics.prefetch_useful_hits = prefetch_metrics.prefetch_useful_hits + 1;
    prefetch_metrics.timely_resident_hits = prefetch_metrics.timely_resident_hits + 1;
    prefetch_metrics.demanded_transfers = prefetch_metrics.demanded_transfers + 1;
    pabi_metrics.useful_prefetch_hits = pabi_metrics.useful_prefetch_hits + 1;
    prefetched_resident_experts.erase(p_it);

    if (it->second.transfer_id != 0) {
      auto tr_it = speculative_transfers.find(it->second.transfer_id);
      if (tr_it != speculative_transfers.end()) {
        tr_it->second.outcome = SpeculativeTransferOutcome::DEMANDED;
        tr_it->second.actual_demand_ns = get_time();
        tr_it->second.remaining_transfer_at_demand_ns = 0.0;
        tr_it->second.demand_arrival_to_ready_ns = 0.0;
      }
    }

    DemandOutcomeRecord dem_rec;
    dem_rec.layer_id = layer_id;
    dem_rec.expert_id = expert_id;
    dem_rec.demand_time_ns = get_time();
    dem_rec.remaining_transfer_ns = 0.0;
    dem_rec.demand_arrival_to_ready_ns = 0.0;
    dem_rec.outcome = DemandOutcome::TIMELY_RESIDENT;
    demand_outcomes.push_back(dem_rec);
  }

  log_cache_event(layer_id, expert_id, "HIT", resident_bytes);
  return true;
}

void Device::mark_expert_resident(int layer_id, int expert_id, long long expert_size,
                                  ExpertClassification classification,
                                  bool is_admitting_prefetch) {
  std::pair<int, int> key = {layer_id, expert_id};
  auto it = resident_experts.find(key);
  if (it != resident_experts.end()) {
    if (it->second.classification == ExpertClassification::SPECULATIVE && classification == ExpertClassification::DEMAND_VERIFIED) {
      promote_speculative_to_demand(layer_id, expert_id);
    }
    return;
  }

  if (eviction_policy != EvictionPolicy::PABI || !pabi_quarantine_enabled) {
    // Standard LRU or FA-LRU Mode (Single list)
    while (resident_bytes + expert_size > cache_capacity_bytes && !lru_expert_list.empty()) {
      std::list<std::tuple<int, int, long long>>::iterator victim_it;
      if (eviction_policy == EvictionPolicy::FA_LRU) {
        victim_it = select_fa_lru_victim();
      } else {
        victim_it = lru_expert_list.end();
        for (auto lit = lru_expert_list.begin(); lit != lru_expert_list.end(); ++lit) {
          int l = std::get<0>(*lit);
          int e = std::get<1>(*lit);
          auto map_it = resident_experts.find({l, e});
          if (map_it != resident_experts.end()) {
            if (map_it->second.is_pinned) {
              continue;
            }
          }
          victim_it = lit;
          break;
        }
      }
      if (victim_it == lru_expert_list.end()) {
        break;
      }

      auto victim_expert = *victim_it;
      lru_expert_list.erase(victim_it);
      int ev_layer = std::get<0>(victim_expert);
      int ev_expert = std::get<1>(victim_expert);
      long long ev_size = std::get<2>(victim_expert);
      std::pair<int, int> ev_key = {ev_layer, ev_expert};
      auto ev_it = resident_experts.find(ev_key);
      std::string ev_reason = "DEMAND_RESIDENT";
      bool was_dem = true;
      if (ev_it != resident_experts.end()) {
        was_dem = ev_it->second.was_demanded;
        if (ev_it->second.is_prefetched && !ev_it->second.was_demanded) {
          ev_reason = "UNCONSUMED_PREFETCH";
          if (ev_it->second.transfer_id != 0) {
            PrefetchMetricsCollector::getInstance().record_transfer_outcome(
                ev_it->second.transfer_id, "EVICTED_BEFORE_DEMAND");
          }
        }
      }
      PrefetchMetricsCollector::getInstance().record_hbm_evicted(
          ev_layer, ev_expert, ev_reason, get_time());
      if (is_admitting_prefetch && was_dem) {
        PrefetchMetricsCollector::getInstance().record_prefetch_induced_eviction(
            ev_layer, ev_expert, get_time());
      }
      if (fa_lru_diag.protected_experts_set.count(ev_key) > 0) {
        if (fa_lru_diag.protected_reused_set.count(ev_key) == 0) {
          fa_lru_diag.protected_unreused_evicted_set.insert(ev_key);
        }
      }
      previously_evicted_experts.insert(ev_key);
      resident_experts.erase(ev_key);
      prefetched_resident_experts.erase(ev_key);
      resident_bytes -= ev_size;
      resident_demand_bytes -= ev_size;
      pabi_metrics.demand_evictions = pabi_metrics.demand_evictions + 1;
      log_cache_event(ev_layer, ev_expert, "EVICT", resident_bytes);
    }

    if (resident_bytes + expert_size > cache_capacity_bytes && is_admitting_prefetch) {
      log_cache_event(layer_id, expert_id, "PREFETCH_REJECT_PINNED_CAPACITY", expert_size);
      return;
    }

    lru_expert_list.push_back({layer_id, expert_id, expert_size});

    ResidentExpertEntry entry;
    entry.layer_id = layer_id;
    entry.expert_id = expert_id;
    entry.size_bytes = expert_size;
    entry.classification = ExpertClassification::DEMAND_VERIFIED;
    entry.list_it = std::prev(lru_expert_list.end());
    entry.access_frequency = 1;
    entry.last_epoch = global_epoch;
    if (is_admitting_prefetch) {
      entry.is_prefetched = true;
      entry.was_demanded = false;
    } else {
      entry.is_prefetched = false;
      entry.was_demanded = true;
    }

    resident_experts[key] = entry;
    resident_bytes += expert_size;
    resident_demand_bytes += expert_size;
    pabi_metrics.demand_insertions = pabi_metrics.demand_insertions + 1;
    PrefetchMetricsCollector::getInstance().record_hbm_resident(
        layer_id, expert_id, 0, "DEMAND", get_time());
    log_cache_event(layer_id, expert_id, "INSERT", resident_bytes);
    return;
  }

  // PABI Mode
  if (expert_size > cache_capacity_bytes) {
    return;
  }

  if (classification == ExpertClassification::SPECULATIVE) {
    if (speculative_buffer_capacity_bytes <= 0 || expert_size > speculative_buffer_capacity_bytes) {
      pabi_metrics.speculative_insertions_rejected = pabi_metrics.speculative_insertions_rejected + 1;
      log_cache_event(layer_id, expert_id, "SPECULATIVE_REJECT", expert_size);
      return;
    }

    // Case A: Speculative buffer capacity enforcement
    while (resident_speculative_bytes + expert_size > speculative_buffer_capacity_bytes && !speculative_lru_list.empty()) {
      auto victim_it = speculative_lru_list.end();
      for (auto sit = speculative_lru_list.begin(); sit != speculative_lru_list.end(); ++sit) {
        int l = std::get<0>(*sit);
        int e = std::get<1>(*sit);
        auto map_it = resident_experts.find({l, e});
        if (map_it != resident_experts.end()) {
          if (map_it->second.is_pinned) {
            pabi_metrics.pinned_eviction_skipped = pabi_metrics.pinned_eviction_skipped + 1;
            log_cache_event(l, e, "PINNED_EVICTION_SKIPPED", 0);
            continue;
          }
        }
        victim_it = sit;
        break;
      }

      if (victim_it == speculative_lru_list.end()) {
        pabi_metrics.speculative_reject_pinned_capacity = pabi_metrics.speculative_reject_pinned_capacity + 1;
        pabi_metrics.speculative_insertions_rejected = pabi_metrics.speculative_insertions_rejected + 1;
        log_cache_event(layer_id, expert_id, "SPECULATIVE_REJECT_PINNED_CAPACITY", expert_size);
        return;
      }

      auto spec_lru = *victim_it;
      speculative_lru_list.erase(victim_it);
      int ev_layer = std::get<0>(spec_lru);
      int ev_expert = std::get<1>(spec_lru);
      long long ev_size = std::get<2>(spec_lru);

      resident_speculative_bytes -= ev_size;
      resident_bytes -= ev_size;

      std::pair<int, int> ev_key = {ev_layer, ev_expert};
      auto ev_it = resident_experts.find(ev_key);
      if (ev_it != resident_experts.end()) {
        if (ev_it->second.is_prefetched && !ev_it->second.was_demanded) {
          if (ev_it->second.transfer_id != 0) {
            auto tr_it = speculative_transfers.find(ev_it->second.transfer_id);
            if (tr_it != speculative_transfers.end()) {
              if (tr_it->second.outcome == SpeculativeTransferOutcome::UNDECIDED) {
                tr_it->second.outcome = SpeculativeTransferOutcome::EVICTED_BEFORE_DEMAND;
                prefetch_metrics.evicted_before_demand_transfers = prefetch_metrics.evicted_before_demand_transfers + 1;
              }
            }
            PrefetchMetricsCollector::getInstance().record_transfer_outcome(
                ev_it->second.transfer_id, "EVICTED_BEFORE_DEMAND");
          }
        }
        std::string ev_reason = "PABI_SPECULATIVE";
        PrefetchMetricsCollector::getInstance().record_hbm_evicted(
            ev_layer, ev_expert, ev_reason, get_time());
      }
      if (fa_lru_diag.protected_experts_set.count(ev_key) > 0) {
        if (fa_lru_diag.protected_reused_set.count(ev_key) == 0) {
          fa_lru_diag.protected_unreused_evicted_set.insert(ev_key);
        }
      }
      previously_evicted_experts.insert(ev_key);
      resident_experts.erase(ev_key);
      prefetched_resident_experts.erase(ev_key);

      pabi_metrics.speculative_evictions = pabi_metrics.speculative_evictions + 1;
      pabi_metrics.unpinned_speculative_evictions = pabi_metrics.unpinned_speculative_evictions + 1;
      log_cache_event(ev_layer, ev_expert, "SPECULATIVE_EVICT", ev_size);
    }

    // Total physical HBM capacity check for speculative insertion
    if (resident_bytes + expert_size > cache_capacity_bytes) {
      while (resident_bytes + expert_size > cache_capacity_bytes && !speculative_lru_list.empty()) {
        auto victim_it = speculative_lru_list.end();
        for (auto sit = speculative_lru_list.begin(); sit != speculative_lru_list.end(); ++sit) {
          int l = std::get<0>(*sit);
          int e = std::get<1>(*sit);
          auto map_it = resident_experts.find({l, e});
          if (map_it != resident_experts.end()) {
            if (map_it->second.is_pinned) {
              pabi_metrics.pinned_eviction_skipped = pabi_metrics.pinned_eviction_skipped + 1;
              log_cache_event(l, e, "PINNED_EVICTION_SKIPPED", 0);
              continue;
            }
          }
          victim_it = sit;
          break;
        }

        if (victim_it == speculative_lru_list.end()) {
          break;
        }

        auto spec_lru = *victim_it;
        speculative_lru_list.erase(victim_it);
        int ev_layer = std::get<0>(spec_lru);
        int ev_expert = std::get<1>(spec_lru);
        long long ev_size = std::get<2>(spec_lru);

        resident_speculative_bytes -= ev_size;
        resident_bytes -= ev_size;

        std::pair<int, int> ev_key = {ev_layer, ev_expert};
        auto ev_it2 = resident_experts.find(ev_key);
        if (ev_it2 != resident_experts.end()) {
          if (ev_it2->second.is_prefetched && !ev_it2->second.was_demanded) {
            if (ev_it2->second.transfer_id != 0) {
              auto tr_it = speculative_transfers.find(ev_it2->second.transfer_id);
              if (tr_it != speculative_transfers.end()) {
                if (tr_it->second.outcome == SpeculativeTransferOutcome::UNDECIDED) {
                  tr_it->second.outcome = SpeculativeTransferOutcome::EVICTED_BEFORE_DEMAND;
                  prefetch_metrics.evicted_before_demand_transfers = prefetch_metrics.evicted_before_demand_transfers + 1;
                }
              }
              PrefetchMetricsCollector::getInstance().record_transfer_outcome(
                  ev_it2->second.transfer_id, "EVICTED_BEFORE_DEMAND");
            }
          }
          std::string ev_reason = "PABI_SPECULATIVE";
          PrefetchMetricsCollector::getInstance().record_hbm_evicted(
              ev_layer, ev_expert, ev_reason, get_time());
        }
        if (fa_lru_diag.protected_experts_set.count(ev_key) > 0) {
        if (fa_lru_diag.protected_reused_set.count(ev_key) == 0) {
          fa_lru_diag.protected_unreused_evicted_set.insert(ev_key);
        }
      }
      previously_evicted_experts.insert(ev_key);
        resident_experts.erase(ev_key);
        prefetched_resident_experts.erase(ev_key);

        pabi_metrics.speculative_evictions = pabi_metrics.speculative_evictions + 1;
        pabi_metrics.unpinned_speculative_evictions = pabi_metrics.unpinned_speculative_evictions + 1;
        log_cache_event(ev_layer, ev_expert, "SPECULATIVE_EVICT", ev_size);
      }

      if (resident_bytes + expert_size > cache_capacity_bytes) {
        // Speculative prefetch cannot fit without evicting demand entries!
        pabi_metrics.speculative_insertions_rejected = pabi_metrics.speculative_insertions_rejected + 1;
        log_cache_event(layer_id, expert_id, "SPECULATIVE_REJECT", expert_size);
        return;
      }
    }

    speculative_lru_list.push_back({layer_id, expert_id, expert_size});
    ResidentExpertEntry entry;
    entry.layer_id = layer_id;
    entry.expert_id = expert_id;
    entry.size_bytes = expert_size;
    entry.classification = ExpertClassification::SPECULATIVE;
    entry.list_it = std::prev(speculative_lru_list.end());

    resident_experts[key] = entry;
    resident_speculative_bytes += expert_size;
    resident_bytes += expert_size;

    pabi_metrics.speculative_insertions = pabi_metrics.speculative_insertions + 1;
    log_cache_event(layer_id, expert_id, "SPECULATIVE_INSERT", resident_bytes);
    return;
  }

  // classification == ExpertClassification::DEMAND_VERIFIED
  while (resident_bytes + expert_size > cache_capacity_bytes) {
    auto victim_spec_it = speculative_lru_list.end();
    for (auto sit = speculative_lru_list.begin(); sit != speculative_lru_list.end(); ++sit) {
      int l = std::get<0>(*sit);
      int e = std::get<1>(*sit);
      auto map_it = resident_experts.find({l, e});
      if (map_it != resident_experts.end()) {
        if (map_it->second.is_pinned) {
          pabi_metrics.pinned_eviction_skipped = pabi_metrics.pinned_eviction_skipped + 1;
          log_cache_event(l, e, "PINNED_EVICTION_SKIPPED", 0);
          continue;
        }
      }
      victim_spec_it = sit;
      break;
    }

    if (victim_spec_it != speculative_lru_list.end()) {
      // Evict UNPINNED speculative entries FIRST!
      auto spec_lru = *victim_spec_it;
      speculative_lru_list.erase(victim_spec_it);
      int ev_layer = std::get<0>(spec_lru);
      int ev_expert = std::get<1>(spec_lru);
      long long ev_size = std::get<2>(spec_lru);

      resident_speculative_bytes -= ev_size;
      resident_bytes -= ev_size;

      std::pair<int, int> ev_key = {ev_layer, ev_expert};
      auto ev_it = resident_experts.find(ev_key);
      std::string ev_reason = "PABI_SPECULATIVE";
      PrefetchMetricsCollector::getInstance().record_hbm_evicted(
          ev_layer, ev_expert, ev_reason, get_time());
      if (fa_lru_diag.protected_experts_set.count(ev_key) > 0) {
        if (fa_lru_diag.protected_reused_set.count(ev_key) == 0) {
          fa_lru_diag.protected_unreused_evicted_set.insert(ev_key);
        }
      }
      previously_evicted_experts.insert(ev_key);
      resident_experts.erase(ev_key);
      prefetched_resident_experts.erase(ev_key);

      pabi_metrics.speculative_evictions = pabi_metrics.speculative_evictions + 1;
      pabi_metrics.unpinned_speculative_evictions = pabi_metrics.unpinned_speculative_evictions + 1;
      log_cache_event(ev_layer, ev_expert, "SPECULATIVE_EVICT", ev_size);
    } else if (!demand_lru_list.empty()) {
      // No unpinned speculative entries remain: evict demand-verified LRU!
      auto dem_lru = demand_lru_list.front();
      demand_lru_list.pop_front();
      int ev_layer = std::get<0>(dem_lru);
      int ev_expert = std::get<1>(dem_lru);
      long long ev_size = std::get<2>(dem_lru);

      resident_demand_bytes -= ev_size;
      resident_bytes -= ev_size;

      std::pair<int, int> ev_key = {ev_layer, ev_expert};
      auto ev_it = resident_experts.find(ev_key);
      std::string ev_reason = "DEMAND_RESIDENT";
      PrefetchMetricsCollector::getInstance().record_hbm_evicted(
          ev_layer, ev_expert, ev_reason, get_time());
      if (fa_lru_diag.protected_experts_set.count(ev_key) > 0) {
        if (fa_lru_diag.protected_reused_set.count(ev_key) == 0) {
          fa_lru_diag.protected_unreused_evicted_set.insert(ev_key);
        }
      }
      previously_evicted_experts.insert(ev_key);
      resident_experts.erase(ev_key);
      prefetched_resident_experts.erase(ev_key);

      for (auto lit = lru_expert_list.begin(); lit != lru_expert_list.end(); ++lit) {
        if (std::get<0>(*lit) == ev_layer && std::get<1>(*lit) == ev_expert) {
          lru_expert_list.erase(lit);
          break;
        }
      }

      pabi_metrics.demand_evictions = pabi_metrics.demand_evictions + 1;
      log_cache_event(ev_layer, ev_expert, "DEMAND_EVICT", ev_size);
    } else {
      break;
    }
  }

  demand_lru_list.push_back({layer_id, expert_id, expert_size});
  lru_expert_list.push_back({layer_id, expert_id, expert_size});

  ResidentExpertEntry entry;
  entry.layer_id = layer_id;
  entry.expert_id = expert_id;
  entry.size_bytes = expert_size;
  entry.classification = ExpertClassification::DEMAND_VERIFIED;
  entry.list_it = std::prev(demand_lru_list.end());

  resident_experts[key] = entry;
  resident_demand_bytes += expert_size;
  resident_bytes += expert_size;
  PrefetchMetricsCollector::getInstance().record_hbm_resident(
      layer_id, expert_id, 0, "DEMAND", get_time());

  pabi_metrics.demand_insertions = pabi_metrics.demand_insertions + 1;
  log_cache_event(layer_id, expert_id, "DEMAND_INSERT", resident_bytes);
}

void Device::mark_expert_speculative(int layer_id, int expert_id, long long expert_size) {
  mark_expert_resident(layer_id, expert_id, expert_size, ExpertClassification::SPECULATIVE);
}

void Device::mark_expert_demand(int layer_id, int expert_id, long long expert_size) {
  mark_expert_resident(layer_id, expert_id, expert_size, ExpertClassification::DEMAND_VERIFIED);
}

bool Device::verify_pabi_invariants() const {
  if (resident_bytes < 0 || resident_demand_bytes < 0 || resident_speculative_bytes < 0) {
    return false;
  }
  if (resident_bytes != resident_demand_bytes + resident_speculative_bytes) {
    return false;
  }
  if (resident_bytes > cache_capacity_bytes) {
    return false;
  }

  if (eviction_policy == EvictionPolicy::PABI && pabi_quarantine_enabled) {
    if (resident_speculative_bytes > speculative_buffer_capacity_bytes) {
      return false;
    }

    long long sum_demand = 0;
    for (const auto& tup : demand_lru_list) {
      sum_demand += std::get<2>(tup);
    }
    if (sum_demand != resident_demand_bytes) {
      return false;
    }

    long long sum_spec = 0;
    for (const auto& tup : speculative_lru_list) {
      sum_spec += std::get<2>(tup);
    }
    if (sum_spec != resident_speculative_bytes) {
      return false;
    }

    if (resident_experts.size() != (demand_lru_list.size() + speculative_lru_list.size())) {
      return false;
    }

    for (const auto& kv : resident_experts) {
      int l = kv.first.first;
      int e = kv.first.second;
      if (std::get<0>(*(kv.second.list_it)) != l || std::get<1>(*(kv.second.list_it)) != e) {
        return false;
      }
      if (kv.second.is_pinned) {
        if (kv.second.classification != ExpertClassification::SPECULATIVE) {
          return false;
        }
      }
    }

    for (const auto& kv : inflight_registry) {
      if (kv.second.transfer_class == TransferClass::DEMAND) {
        if (kv.second.has_pin_intent) {
          return false;
        }
      }
    }

    return true;
  }

  // EvictionPolicy::LRU or quarantine disabled
  long long sum_lru = 0;
  for (const auto& tup : lru_expert_list) {
    sum_lru += std::get<2>(tup);
  }
  if (sum_lru != resident_bytes) {
    return false;
  }

  if (resident_experts.size() != lru_expert_list.size()) {
    return false;
  }

  for (const auto& kv : resident_experts) {
    int l = kv.first.first;
    int e = kv.first.second;
    if (std::get<0>(*(kv.second.list_it)) != l || std::get<1>(*(kv.second.list_it)) != e) {
      return false;
    }
    if (kv.second.is_pinned) {
      return false;
    }
  }

  return true;
}

uint64_t Device::submit_demand_transfer(int layer_id, int expert_id, double current_time_ns, uint64_t bytes, int request_id, int seq_id, int token_id) {
  if (ddr5_transfer_engine == nullptr) {
    double bw = config.ddr5_bandwidth_gbps;
    if (bw <= 0.0) {
      bw = DDR5TransferEngine::DEFAULT_BANDWIDTH_BYTES_PER_NS;
    }
    ddr5_transfer_engine = std::make_shared<DDR5TransferEngine>(bw);
  }
  std::pair<int, int> key = {layer_id, expert_id};
  if (previously_evicted_experts.find(key) != previously_evicted_experts.end()) {
    reloaded_experts_count = reloaded_experts_count + 1;
    previously_evicted_experts.erase(key);
  }
  auto it = inflight_registry.find(key);
  if (it != inflight_registry.end()) {
    prefetch_metrics.demand_inflight_attachments = prefetch_metrics.demand_inflight_attachments + 1;
    prefetch_metrics.late_inflight_attachments = prefetch_metrics.late_inflight_attachments + 1;
    prefetch_metrics.demanded_transfers = prefetch_metrics.demanded_transfers + 1;

    double remaining_ns = 0.0;
    if (ddr5_transfer_engine != nullptr) {
      const auto* tr = ddr5_transfer_engine->get_transfer(it->second.transfer_id);
      if (tr != nullptr) {
        double bw = ddr5_transfer_engine->get_bandwidth_bytes_per_ns();
        if (bw <= 0.0) {
          bw = DDR5TransferEngine::DEFAULT_BANDWIDTH_BYTES_PER_NS;
        }
        if (tr->state == TransferState::ACTIVE) {
          if (tr->completion_time_ns > current_time_ns) {
            remaining_ns = tr->completion_time_ns - current_time_ns;
          } else {
            remaining_ns = tr->remaining_bytes / bw;
          }
        } else if (tr->state == TransferState::QUEUED || tr->state == TransferState::PAUSED) {
          remaining_ns = tr->remaining_bytes / bw;
        } else if (tr->state == TransferState::COMPLETED) {
          remaining_ns = 0.0;
        }
      }
    }

    auto tr_it = speculative_transfers.find(it->second.transfer_id);
    if (tr_it != speculative_transfers.end()) {
      tr_it->second.outcome = SpeculativeTransferOutcome::DEMANDED;
      tr_it->second.actual_demand_ns = current_time_ns;
      tr_it->second.remaining_transfer_at_demand_ns = remaining_ns;
      tr_it->second.demand_arrival_to_ready_ns = remaining_ns;
    }

    DemandOutcomeRecord dem_rec;
    dem_rec.request_id = request_id;
    dem_rec.seq_id = seq_id;
    dem_rec.token_id = token_id;
    dem_rec.layer_id = layer_id;
    dem_rec.expert_id = expert_id;
    dem_rec.demand_time_ns = current_time_ns;
    dem_rec.remaining_transfer_ns = remaining_ns;
    dem_rec.demand_arrival_to_ready_ns = remaining_ns;
    dem_rec.outcome = DemandOutcome::LATE_IN_FLIGHT;
    demand_outcomes.push_back(dem_rec);

    PrefetchMetricsCollector::getInstance().record_transfer_outcome(it->second.transfer_id, "DEMANDED");
    PrefetchMetricsCollector::getInstance().record_expert_demanded(
        0, request_id, seq_id, token_id, layer_id, expert_id, current_time_ns,
        current_time_ns + remaining_ns, remaining_ns, 0.0, "LATE_IN_FLIGHT",
        it->second.transfer_id, it->second.candidate_id);

    if (it->second.transfer_class == TransferClass::PREFETCH) {
      promote_inflight_to_demand(layer_id, expert_id, current_time_ns);
    }
    return it->second.transfer_id;
  }

  auto legacy_it = inflight_experts.find(key);
  if (legacy_it != inflight_experts.end()) {
    return legacy_it->second;
  }

  // Pure demand miss (not prefetched)
  prefetch_metrics.not_prefetched_misses = prefetch_metrics.not_prefetched_misses + 1;

  double exp_dur = 0.0;
  double bw = config.ddr5_bandwidth_gbps;
  if (bw > 0.0) {
    exp_dur = static_cast<double>(bytes) / bw;
  } else {
    exp_dur = static_cast<double>(bytes) / DDR5TransferEngine::DEFAULT_BANDWIDTH_BYTES_PER_NS;
  }

  DemandOutcomeRecord dem_rec;
  dem_rec.request_id = request_id;
  dem_rec.seq_id = seq_id;
  dem_rec.token_id = token_id;
  dem_rec.layer_id = layer_id;
  dem_rec.expert_id = expert_id;
  dem_rec.demand_time_ns = current_time_ns;
  dem_rec.remaining_transfer_ns = exp_dur;
  dem_rec.demand_arrival_to_ready_ns = exp_dur;
  dem_rec.outcome = DemandOutcome::NOT_PREFETCHED;
  demand_outcomes.push_back(dem_rec);

  uint64_t tid = ddr5_transfer_engine->submit_demand_id(request_id, layer_id, expert_id, current_time_ns, bytes, 0, seq_id, token_id);
  if (tid != 0) {
    InflightExpertEntry entry;
    entry.transfer_id = tid;
    entry.transfer_class = TransferClass::DEMAND;
    entry.dispatch_time_ns = current_time_ns;
    entry.layer_id = layer_id;
    entry.expert_id = expert_id;
    entry.request_id = request_id;
    entry.seq_id = seq_id;
    entry.token_id = token_id;

    inflight_registry[key] = entry;
    inflight_experts[key] = tid;

    PrefetchMetricsCollector::getInstance().record_transfer_submitted(
        tid, 0, request_id, seq_id, token_id, layer_id, expert_id, "DEMAND", 1, bytes, current_time_ns);
    PrefetchMetricsCollector::getInstance().record_expert_demanded(
        0, request_id, seq_id, token_id, layer_id, expert_id, current_time_ns,
        current_time_ns + exp_dur, exp_dur, 0.0, "DEMAND_MISS", tid, 0);
  }
  return tid;
}

uint64_t Device::submit_prefetch_transfer(int layer_id, int expert_id, double current_time_ns, uint64_t bytes, int priority, int source_stage, uint64_t candidate_id, int request_id, int seq_id, int token_id) {
  if (ddr5_transfer_engine == nullptr) {
    double bw = config.ddr5_bandwidth_gbps;
    if (bw <= 0.0) {
      bw = DDR5TransferEngine::DEFAULT_BANDWIDTH_BYTES_PER_NS;
    }
    ddr5_transfer_engine = std::make_shared<DDR5TransferEngine>(bw);
  }

  if (has_resident_expert(layer_id, expert_id)) {
    return 0;
  }

  std::pair<int, int> key = {layer_id, expert_id};
  if (inflight_registry.find(key) != inflight_registry.end()) {
    return 0;
  }
  if (inflight_experts.find(key) != inflight_experts.end()) {
    return 0;
  }

  uint64_t tid = ddr5_transfer_engine->submit_prefetch_id(request_id, layer_id, expert_id, current_time_ns, bytes, priority, 0, seq_id, token_id, candidate_id);
  if (tid != 0) {
    InflightExpertEntry entry;
    entry.transfer_id = tid;
    entry.transfer_class = TransferClass::PREFETCH;
    entry.dispatch_time_ns = current_time_ns;
    entry.layer_id = layer_id;
    entry.expert_id = expert_id;
    entry.source_stage = source_stage;
    entry.candidate_id = candidate_id;
    entry.request_id = request_id;
    entry.seq_id = seq_id;
    entry.token_id = token_id;

    SpeculativeTransferRecord tr_rec;
    tr_rec.transfer_id = tid;
    tr_rec.candidate_id = candidate_id;
    tr_rec.request_id = request_id;
    tr_rec.seq_id = seq_id;
    tr_rec.token_id = token_id;
    tr_rec.source_stage = source_stage;
    tr_rec.target_stage = layer_id;
    tr_rec.expert_id = expert_id;
    tr_rec.dispatch_time_ns = current_time_ns;
    tr_rec.transfer_start_ns = -1.0;
    tr_rec.outcome = SpeculativeTransferOutcome::UNDECIDED;

    const auto* tr = ddr5_transfer_engine->get_transfer(tid);
    if (tr != nullptr && tr->service_start_time_ns >= 0.0) {
      tr_rec.transfer_start_ns = tr->service_start_time_ns;
    }

    // Phase 4D.4: Attach pin intent if present
    auto pin_it = active_pin_intents.find(key);
    if (pin_it != active_pin_intents.end()) {
      if (pin_it->second.is_active) {
        entry.has_pin_intent = true;
        entry.pin_target_stage_id = pin_it->second.target_stage_id;
        entry.pin_expiration_stage_id = pin_it->second.expiration_stage_id;
        entry.prediction_rank = pin_it->second.prediction_rank;
        entry.prediction_score = pin_it->second.prediction_score;
        entry.candidate_id = pin_it->second.candidate_id;
        pin_it->second.transfer_id = tid;
        pabi_metrics.inflight_pin_intents = pabi_metrics.inflight_pin_intents + 1;

        tr_rec.candidate_id = pin_it->second.candidate_id;
      }
    }

    speculative_transfers[tid] = tr_rec;
    inflight_registry[key] = entry;
    inflight_experts[key] = tid;

    PrefetchMetricsCollector::getInstance().record_transfer_submitted(
        tid, entry.candidate_id, request_id, seq_id, token_id, layer_id, expert_id, "PREFETCH", priority, bytes, current_time_ns);
    prefetch_metrics.prefetch_transfers_submitted = prefetch_metrics.prefetch_transfers_submitted + 1;
  }
  return tid;
}

bool Device::promote_inflight_to_demand(int layer_id, int expert_id, double current_time_ns) {
  if (ddr5_transfer_engine == nullptr) {
    return false;
  }
  std::pair<int, int> key = {layer_id, expert_id};
  if (previously_evicted_experts.find(key) != previously_evicted_experts.end()) {
    reloaded_experts_count = reloaded_experts_count + 1;
    previously_evicted_experts.erase(key);
  }
  auto it = inflight_registry.find(key);
  if (it == inflight_registry.end()) {
    return false;
  }
  bool ok = ddr5_transfer_engine->promote_transfer_to_demand(it->second.transfer_id, current_time_ns);
  if (ok) {
    it->second.transfer_class = TransferClass::DEMAND;
    if (it->second.has_pin_intent) {
      it->second.has_pin_intent = false;
      it->second.pin_target_stage_id = -1;
      it->second.pin_expiration_stage_id = -1;
      pabi_metrics.pins_cleared_on_promotion = pabi_metrics.pins_cleared_on_promotion + 1;
      log_cache_event(layer_id, expert_id, "PIN_CLEARED_ON_PROMOTION", 0);
    }
    active_pin_intents.erase(key);
    prefetch_metrics.prefetch_promotions_to_demand = prefetch_metrics.prefetch_promotions_to_demand + 1;

    auto tr_it = speculative_transfers.find(it->second.transfer_id);
    if (tr_it != speculative_transfers.end()) {
      tr_it->second.outcome = SpeculativeTransferOutcome::DEMANDED;
      tr_it->second.actual_demand_ns = current_time_ns;
    }
  }
  return ok;
}

void Device::process_completed_prefetches(double current_device_time) {
  if (ddr5_transfer_engine == nullptr) {
    return;
  }

  ddr5_transfer_engine->run_until(current_device_time);

  std::vector<std::pair<int, int>> completed_keys;
  for (const auto& kv : inflight_registry) {
    const auto* tr = ddr5_transfer_engine->get_transfer(kv.second.transfer_id);
    if (tr != nullptr) {
      if (tr->is_completed()) {
        completed_keys.push_back(kv.first);
      } else if (tr->state == TransferState::CANCELLED) {
        completed_keys.push_back(kv.first);
      }
    }
  }

  for (const auto& key : completed_keys) {
    auto it = inflight_registry.find(key);
    if (it != inflight_registry.end()) {
      int layer_id = it->second.layer_id;
      int expert_id = it->second.expert_id;
      const auto* tr = ddr5_transfer_engine->get_transfer(it->second.transfer_id);
      if (tr != nullptr && tr->is_completed()) {
        if (!has_resident_expert(layer_id, expert_id)) {
          if (eviction_policy == EvictionPolicy::PABI && pabi_quarantine_enabled && it->second.transfer_class == TransferClass::PREFETCH) {
            mark_expert_speculative(layer_id, expert_id, tr->total_bytes);
            auto res_it = resident_experts.find(key);
            if (res_it != resident_experts.end()) {
              res_it->second.is_prefetched = true;
              res_it->second.transfer_id = it->second.transfer_id;
              res_it->second.dispatch_time_ns = it->second.dispatch_time_ns;
              res_it->second.completion_time_ns = tr->completion_time_ns;
              res_it->second.was_demanded = false;
            }
            if (is_pinning_active() && it->second.has_pin_intent && it->second.pin_expiration_stage_id >= 0) {
              if (it->second.pin_expiration_stage_id < current_stage_id) {
                // Deadline passed before transfer completed: insert unpinned
                it->second.has_pin_intent = false;
                pabi_metrics.pin_expirations = pabi_metrics.pin_expirations + 1;
                log_cache_event(layer_id, expert_id, "PIN_EXPIRED", it->second.pin_expiration_stage_id);
              } else {
                if (res_it != resident_experts.end()) {
                  if (res_it->second.classification == ExpertClassification::SPECULATIVE) {
                    res_it->second.is_pinned = true;
                    res_it->second.pin_target_stage_id = it->second.pin_target_stage_id;
                    res_it->second.pin_expiration_stage_id = it->second.pin_expiration_stage_id;
                    res_it->second.prediction_rank = it->second.prediction_rank;
                    res_it->second.prediction_score = it->second.prediction_score;
                    res_it->second.candidate_id = it->second.candidate_id;
                    pabi_metrics.resident_experts_pinned = pabi_metrics.resident_experts_pinned + 1;
                    log_cache_event(layer_id, expert_id, "PIN_APPLIED", res_it->second.prediction_rank);
                  }
                }
              }
            }
          } else {
            mark_expert_resident(layer_id, expert_id, tr->total_bytes, ExpertClassification::DEMAND_VERIFIED, true);
            auto res_it = resident_experts.find(key);
            if (res_it != resident_experts.end()) {
              res_it->second.is_prefetched = true;
              res_it->second.transfer_id = it->second.transfer_id;
              res_it->second.dispatch_time_ns = it->second.dispatch_time_ns;
              res_it->second.completion_time_ns = tr->completion_time_ns;
              res_it->second.was_demanded = false;
            }
          }
          prefetched_resident_experts.insert(key);
          prefetch_metrics.prefetches_inserted_hbm = prefetch_metrics.prefetches_inserted_hbm + 1;
        }
        if (it->second.transfer_class == TransferClass::PREFETCH) {
          prefetch_metrics.prefetch_transfers_completed = prefetch_metrics.prefetch_transfers_completed + 1;
          auto tr_it = speculative_transfers.find(it->second.transfer_id);
          if (tr_it != speculative_transfers.end()) {
            if (tr->service_start_time_ns >= 0.0) {
              tr_it->second.transfer_start_ns = tr->service_start_time_ns;
            }
            tr_it->second.transfer_complete_ns = tr->completion_time_ns;
          }
        }
        PrefetchMetricsCollector::getInstance().record_transfer_completed(
            it->second.transfer_id, tr->completion_time_ns);
        std::string res_class = "DEMAND";
        if (it->second.transfer_class == TransferClass::PREFETCH) {
          res_class = "PREFETCH";
        }
        PrefetchMetricsCollector::getInstance().record_hbm_resident(
            layer_id, expert_id, it->second.transfer_id, res_class, tr->completion_time_ns);
      } else if (tr != nullptr && tr->state == TransferState::CANCELLED) {
        prefetch_metrics.cancelled_prefetches = prefetch_metrics.cancelled_prefetches + 1;
        auto tr_it = speculative_transfers.find(it->second.transfer_id);
        if (tr_it != speculative_transfers.end()) {
          tr_it->second.outcome = SpeculativeTransferOutcome::CANCELLED;
        }
        PrefetchMetricsCollector::getInstance().record_transfer_cancelled(
            it->second.transfer_id, current_device_time, tr->transferred_bytes);
        PrefetchMetricsCollector::getInstance().record_transfer_outcome(
            it->second.transfer_id, "CANCELLED");
        active_pin_intents.erase(key);
      }
      inflight_registry.erase(it);
      inflight_experts.erase(key);
    }
  }
}

int Device::dispatch_prefetch_candidates(std::shared_ptr<PrefetchCandidateQueue> queue, double current_time_ns, int max_candidates) {
  if (queue == nullptr || ddr5_transfer_engine == nullptr) {
    return 0;
  }

  process_completed_prefetches(current_time_ns);

  auto candidates = queue->select_ready_candidates(max_candidates);
  int dispatched_count = 0;

  if (is_pinning_active()) {
    std::unordered_map<int, std::vector<PrefetchCandidate>> stage_groups;
    for (const auto& cand : candidates) {
      if (cand.target_stage_id >= 0 && cand.expert_id >= 0) {
        stage_groups[cand.target_stage_id].push_back(cand);
      }
    }
    for (const auto& grp : stage_groups) {
      select_pin_candidates_for_stage(grp.first, grp.second);
    }
  }

  for (const auto& cand : candidates) {
    prefetch_metrics.candidates_evaluated = prefetch_metrics.candidates_evaluated + 1;

    if (cand.expert_id < 0 || cand.target_stage_id < 0) {
      continue;
    }

    if (cand.state != CandidateState::ADMITTED && cand.state != CandidateState::READY) {
      continue;
    }

    if (has_resident_expert(cand.target_stage_id, cand.expert_id)) {
      prefetch_metrics.resident_prefetches_skipped = prefetch_metrics.resident_prefetches_skipped + 1;
      queue->mark_candidate_dispatched(cand.candidate_id, current_time_ns);
      continue;
    }

    if (is_expert_inflight(cand.target_stage_id, cand.expert_id)) {
      prefetch_metrics.duplicate_prefetches_prevented = prefetch_metrics.duplicate_prefetches_prevented + 1;
      queue->mark_candidate_dispatched(cand.candidate_id, current_time_ns);
      continue;
    }

    uint64_t tid = submit_prefetch_transfer(cand.target_stage_id, cand.expert_id, current_time_ns,
                                            DDR5TransferEngine::DEFAULT_EXPERT_BYTES, 0,
                                            cand.source_stage_id, cand.candidate_id,
                                            cand.request_id, cand.seq_id, cand.token_id);
    if (tid != 0) {
      queue->mark_candidate_dispatched(cand.candidate_id, current_time_ns);
      dispatched_count = dispatched_count + 1;
      prefetch_metrics.candidates_dispatched = prefetch_metrics.candidates_dispatched + 1;
    }
  }

  return dispatched_count;
}

bool Device::wait_for_demand_transfer(uint64_t transfer_id) {
  if (ddr5_transfer_engine == nullptr) {
    return false;
  }
  bool ok = ddr5_transfer_engine->run_until_transfer_completed(transfer_id);
  if (ok) {
    const auto* tr = ddr5_transfer_engine->get_transfer(transfer_id);
    if (tr != nullptr) {
      auto tr_it = speculative_transfers.find(transfer_id);
      if (tr_it != speculative_transfers.end()) {
        if (tr_it->second.actual_demand_ns > 0.0 && tr->completion_time_ns >= tr_it->second.actual_demand_ns) {
          tr_it->second.demand_arrival_to_ready_ns = tr->completion_time_ns - tr_it->second.actual_demand_ns;
        }
        if (tr->service_start_time_ns >= 0.0) {
          tr_it->second.transfer_start_ns = tr->service_start_time_ns;
        }
        tr_it->second.transfer_complete_ns = tr->completion_time_ns;
      }
      for (auto& dem : demand_outcomes) {
        if (dem.layer_id == tr->layer_id && dem.expert_id == tr->expert_id) {
          if (dem.demand_time_ns > 0.0 && tr->completion_time_ns >= dem.demand_time_ns) {
            dem.demand_arrival_to_ready_ns = tr->completion_time_ns - dem.demand_time_ns;
          }
        }
      }
      PrefetchMetricsCollector::getInstance().record_transfer_completed(
          transfer_id, tr->completion_time_ns);
    }
  }
  return ok;
}

bool Device::is_expert_inflight(int layer_id, int expert_id) const {
  std::pair<int, int> key = {layer_id, expert_id};
  if (inflight_registry.find(key) != inflight_registry.end()) {
    return true;
  }
  if (inflight_experts.find(key) != inflight_experts.end()) {
    return true;
  }
  return false;
}

void Device::clear_inflight_expert(int layer_id, int expert_id) {
  std::pair<int, int> key = {layer_id, expert_id};
  inflight_registry.erase(key);
  inflight_experts.erase(key);
}

void Device::dump_prefetch_effectiveness(const std::string& filepath) {
  std::string path = filepath;
  if (path.empty()) {
    const char* trace_dir_env = std::getenv("TRACE_DIR");
    if (trace_dir_env != nullptr) {
      path = std::string(trace_dir_env) + "/prefetch_effectiveness.csv";
    } else {
      path = "prefetch_effectiveness.csv";
    }
  }

  // Reconcile outstanding speculative transfers
  for (auto& kv : speculative_transfers) {
    const MemoryTransfer* tr = nullptr;
    if (ddr5_transfer_engine != nullptr) {
      tr = ddr5_transfer_engine->get_transfer(kv.second.transfer_id);
    }
    if (tr != nullptr) {
      if (tr->service_start_time_ns >= 0.0) {
        kv.second.transfer_start_ns = tr->service_start_time_ns;
      }
      if (tr->completion_time_ns >= 0.0) {
        kv.second.transfer_complete_ns = tr->completion_time_ns;
      }
    }
    if (kv.second.outcome == SpeculativeTransferOutcome::UNDECIDED) {
      const MemoryTransfer* tr = nullptr;
      if (ddr5_transfer_engine != nullptr) {
        tr = ddr5_transfer_engine->get_transfer(kv.second.transfer_id);
      }
      if (tr != nullptr) {
        if (tr->state == TransferState::CANCELLED) {
          kv.second.outcome = SpeculativeTransferOutcome::CANCELLED;
        } else if (tr->is_completed()) {
          kv.second.outcome = SpeculativeTransferOutcome::UNUSED;
          prefetch_metrics.unused_transfers = prefetch_metrics.unused_transfers + 1;
        } else {
          kv.second.outcome = SpeculativeTransferOutcome::STILL_IN_FLIGHT_AT_END;
          prefetch_metrics.still_inflight_at_end_transfers = prefetch_metrics.still_inflight_at_end_transfers + 1;
        }
      } else {
        kv.second.outcome = SpeculativeTransferOutcome::STILL_IN_FLIGHT_AT_END;
        prefetch_metrics.still_inflight_at_end_transfers = prefetch_metrics.still_inflight_at_end_transfers + 1;
      }
    }
  }

  std::ofstream out(path);
  if (!out.is_open()) {
    return;
  }

  out << "event_type,transfer_id,candidate_id,request_id,token_id,source_stage,target_stage,expert_id,"
      << "predict_ts,dispatch_ts,transfer_start_ts,transfer_complete_ts,actual_demand_ts,"
      << "remaining_transfer_ns,demand_arrival_to_ready_ns,outcome\n";

  for (const auto& kv : speculative_transfers) {
    const auto& rec = kv.second;
    std::string outcome_str = "UNDECIDED";
    if (rec.outcome == SpeculativeTransferOutcome::DEMANDED) {
      outcome_str = "DEMANDED";
    } else if (rec.outcome == SpeculativeTransferOutcome::UNUSED) {
      outcome_str = "UNUSED";
    } else if (rec.outcome == SpeculativeTransferOutcome::CANCELLED) {
      outcome_str = "CANCELLED";
    } else if (rec.outcome == SpeculativeTransferOutcome::EVICTED_BEFORE_DEMAND) {
      outcome_str = "EVICTED_BEFORE_DEMAND";
    } else if (rec.outcome == SpeculativeTransferOutcome::STILL_IN_FLIGHT_AT_END) {
      outcome_str = "STILL_IN_FLIGHT_AT_END";
    }

    out << "SPECULATIVE_TRANSFER,"
        << rec.transfer_id << ","
        << rec.candidate_id << ","
        << rec.request_id << ","
        << rec.token_id << ","
        << rec.source_stage << ","
        << rec.target_stage << ","
        << rec.expert_id << ","
        << std::fixed << std::setprecision(2) << rec.predict_time_ns << ","
        << rec.dispatch_time_ns << ","
        << rec.transfer_start_ns << ","
        << rec.transfer_complete_ns << ","
        << rec.actual_demand_ns << ","
        << rec.remaining_transfer_at_demand_ns << ","
        << rec.demand_arrival_to_ready_ns << ","
        << outcome_str << "\n";
  }

  for (const auto& dem : demand_outcomes) {
    std::string outcome_str = "NOT_PREFETCHED";
    if (dem.outcome == DemandOutcome::TIMELY_RESIDENT) {
      outcome_str = "TIMELY_RESIDENT";
    } else if (dem.outcome == DemandOutcome::LATE_IN_FLIGHT) {
      outcome_str = "LATE_IN_FLIGHT";
    } else if (dem.outcome == DemandOutcome::NOT_PREFETCHED) {
      outcome_str = "NOT_PREFETCHED";
    }

    out << "DEMAND_ACCESS,0,0,"
        << dem.request_id << ","
        << dem.token_id << ",-1,"
        << dem.layer_id << ","
        << dem.expert_id << ",0.0,0.0,0.0,0.0,"
        << std::fixed << std::setprecision(2) << dem.demand_time_ns << ","
        << dem.remaining_transfer_ns << ","
        << dem.demand_arrival_to_ready_ns << ","
        << outcome_str << "\n";
  }
}

};  // namespace llm_system
