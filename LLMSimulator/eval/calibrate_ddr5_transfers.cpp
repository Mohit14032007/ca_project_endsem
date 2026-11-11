#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/mmap_controller.h"
#include "hardware/cluster.h"
#include "hardware/device.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"

using namespace llm_system;

namespace {

struct StreamSpec {
    int expert_id = 0;
    std::string stream_type = "DEMAND"; // "DEMAND" or "PREFETCH"
    int num_tensors = 3;
    long long tensor_size = 29360128LL;
};

struct ExperimentConfig {
    std::string experiment_id;
    std::string workload_label;
    std::string transfer_type;
    int concurrency_streams = 1;
    int demand_streams = 1;
    int prefetch_streams = 0;
    int tensors_per_stream = 3;
    long long tensor_size = 29360128LL;
    std::vector<StreamSpec> streams;
};

struct CalibrationResult {
    std::string experiment_id;
    std::string workload_label;
    int concurrency_streams = 0;
    std::string transfer_type;
    int demand_streams = 0;
    int prefetch_streams = 0;
    uint64_t total_bytes = 0;
    uint64_t bytes_per_stream = 0;
    double simulated_start_time_ns = 0.0;
    double simulated_completion_time_ns = 0.0;
    double simulated_duration_ns = 0.0;
    double aggregate_effective_bandwidth_gbps = 0.0;
    double per_stream_effective_bandwidth_gbps = 0.0;
    double per_stream_min_latency_ns = 0.0;
    double per_stream_max_latency_ns = 0.0;
    double per_stream_mean_latency_ns = 0.0;
    std::string ddr5_config;
    int repetitions = 1;
    double host_runtime_seconds = 0.0;
};

CalibrationResult run_calibration_experiment(
    const ExperimentConfig& exp,
    const std::string& ddr5_config_path,
    Device::Ptr device,
    Cluster::Ptr cluster) {

    CalibrationResult res;
    res.experiment_id = exp.experiment_id;
    res.workload_label = exp.workload_label;
    res.concurrency_streams = exp.concurrency_streams;
    res.transfer_type = exp.transfer_type;
    res.demand_streams = exp.demand_streams;
    res.prefetch_streams = exp.prefetch_streams;
    res.ddr5_config = ddr5_config_path;
    res.repetitions = 1;

    std::vector<Tensor_Ptr> batch_tensors;
    std::vector<uint64_t> stream_bytes;

    uint64_t total_experiment_bytes = 0;

    for (size_t s = 0; s < exp.streams.size(); ++s) {
        const auto& stream = exp.streams[s];
        long long expert_base_addr = static_cast<long long>(stream.expert_id) * 88080384LL;
        uint64_t this_stream_bytes = 0;

        for (int t = 0; t < stream.num_tensors; ++t) {
            long long tensor_addr = expert_base_addr + static_cast<long long>(t) * stream.tensor_size;
            std::string tname = "T_" + exp.experiment_id + "_S" + std::to_string(s) + "_W" + std::to_string(t);
            Tensor::Ptr weight = Tensor::Create(tname, {1, static_cast<int>(stream.tensor_size)}, "weight", device);
            weight->weight_target = MemoryTarget::OFFCHIP_DRAM;
            weight->allocateMemoryObject(MemoryObject::Create(
                MMap::ALL_CHANNEL, tensor_addr, stream.tensor_size, device->offchip_mmap_controller));
            batch_tensors.push_back(weight);
            this_stream_bytes += stream.tensor_size;
        }

        stream_bytes.push_back(this_stream_bytes);
        total_experiment_bytes += this_stream_bytes;
    }

    res.total_bytes = total_experiment_bytes;
    if (exp.concurrency_streams > 0) {
        res.bytes_per_stream = total_experiment_bytes / static_cast<uint64_t>(exp.concurrency_streams);
    } else {
        res.bytes_per_stream = total_experiment_bytes;
    }

    auto host_start = std::chrono::high_resolution_clock::now();
    double sim_time_before = 0.0;
    if (device->offchip_dram_interface != nullptr) {
        sim_time_before = device->offchip_dram_interface->time;
    }

    // Unconditionally execute Ramulator DDR5 simulation
    device->config.bypass_offchip_execution_cache = true;
    ExecStatus status = issueRamulatorBatch(
        device,
        LayerType::LINEAR,
        ProcessorType::GPU,
        DRAMRequestType::kRead,
        PIMOperandType::kDRAM,
        batch_tensors,
        MemoryTarget::OFFCHIP_DRAM);

    auto host_end = std::chrono::high_resolution_clock::now();
    double sim_time_after = 0.0;
    if (device->offchip_dram_interface != nullptr) {
        sim_time_after = device->offchip_dram_interface->time;
    }

    std::chrono::duration<double> host_dur = host_end - host_start;
    res.host_runtime_seconds = host_dur.count();
    res.simulated_start_time_ns = sim_time_before;
    res.simulated_completion_time_ns = sim_time_after;

    double duration_ns = sim_time_after - sim_time_before;
    if (duration_ns <= 0.0) {
        duration_ns = status.memory_duration;
    }
    res.simulated_duration_ns = duration_ns;

    if (duration_ns > 0.0) {
        // Aggregate effective bandwidth in GB/s (10^9 B/s)
        double duration_sec = duration_ns / 1e9;
        res.aggregate_effective_bandwidth_gbps = (static_cast<double>(total_experiment_bytes) / duration_sec) / 1e9;
    } else {
        res.aggregate_effective_bandwidth_gbps = 0.0;
    }

    // Extract per-stream completion latencies from status.per_request_duration
    std::vector<double> stream_latencies;
    int tensor_idx = 0;
    for (size_t s = 0; s < exp.streams.size(); ++s) {
        double max_stream_req_dur = 0.0;
        for (int t = 0; t < exp.streams[s].num_tensors; ++t) {
            if (tensor_idx < static_cast<int>(status.per_request_duration.size())) {
                double req_dur = status.per_request_duration[tensor_idx];
                if (req_dur > max_stream_req_dur) {
                    max_stream_req_dur = req_dur;
                }
            }
            tensor_idx++;
        }
        if (max_stream_req_dur > 0.0) {
            stream_latencies.push_back(max_stream_req_dur);
        } else {
            stream_latencies.push_back(duration_ns);
        }
    }

    if (!stream_latencies.empty()) {
        double min_lat = stream_latencies[0];
        double max_lat = stream_latencies[0];
        double sum_lat = 0.0;
        for (double lat : stream_latencies) {
            if (lat < min_lat) {
                min_lat = lat;
            }
            if (lat > max_lat) {
                max_lat = lat;
            }
            sum_lat += lat;
        }
        res.per_stream_min_latency_ns = min_lat;
        res.per_stream_max_latency_ns = max_lat;
        res.per_stream_mean_latency_ns = sum_lat / static_cast<double>(stream_latencies.size());

        if (res.per_stream_mean_latency_ns > 0.0) {
            double stream_sec = res.per_stream_mean_latency_ns / 1e9;
            res.per_stream_effective_bandwidth_gbps = (static_cast<double>(res.bytes_per_stream) / stream_sec) / 1e9;
        } else {
            res.per_stream_effective_bandwidth_gbps = 0.0;
        }
    } else {
        res.per_stream_min_latency_ns = duration_ns;
        res.per_stream_max_latency_ns = duration_ns;
        res.per_stream_mean_latency_ns = duration_ns;
        res.per_stream_effective_bandwidth_gbps = res.aggregate_effective_bandwidth_gbps;
    }

    return res;
}

} // namespace

int main() {
    std::cout << "======================================================================\n";
    std::cout << "PHASE 4B: Ramulator DDR5 Transfer Calibration Harness\n";
    std::cout << "Executable: calibrate_ddr5_transfers\n";
    std::cout << "======================================================================\n" << std::flush;

    std::string ddr5_config_path = "reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/config_snapshot/dram_config_DDR5_retry.yaml";
    if (!std::filesystem::exists(ddr5_config_path)) {
        std::cerr << "[ERROR] DDR5 config not found: " << ddr5_config_path << "\n";
        return 1;
    }

    SystemConfig system_config;
    system_config.gpu_gen = "A100";
    system_config.use_ramulator = true;
    system_config.bypass_offchip_execution_cache = true;
    system_config.offchip_dram_cfg_path = ddr5_config_path;

    Cluster::Ptr cluster = Cluster::Create(system_config, nullptr);
    Device::Ptr device = Device::Create(system_config, 0, cluster);
    device->perform_execution = true;
    device->reset_expert_cache();

    std::cout << "[INIT] Cluster and Device created successfully.\n";
    std::cout << "[INIT] DDR5 Config: " << ddr5_config_path << "\n";
    std::cout << "[INIT] Cache bypass active: Ramulator will execute cycle-by-cycle for calibration.\n\n";

    const long long expert_size = 88080384LL;       // 88.08 MB
    const long long size_per_tensor = 29360128LL;   // 29.36 MB

    std::vector<ExperimentConfig> matrix;

    // C1: 1 Demand Transfer
    {
        ExperimentConfig c1;
        c1.experiment_id = "C1";
        c1.workload_label = "DEMAND_1";
        c1.transfer_type = "DEMAND";
        c1.concurrency_streams = 1;
        c1.demand_streams = 1;
        c1.prefetch_streams = 0;
        c1.streams.push_back({10, "DEMAND", 3, size_per_tensor});
        matrix.push_back(c1);
    }

    // C2: 2 Concurrent Demand Transfers
    {
        ExperimentConfig c2;
        c2.experiment_id = "C2";
        c2.workload_label = "CONCURRENT_DEMAND_2";
        c2.transfer_type = "DEMAND";
        c2.concurrency_streams = 2;
        c2.demand_streams = 2;
        c2.prefetch_streams = 0;
        c2.streams.push_back({11, "DEMAND", 3, size_per_tensor});
        c2.streams.push_back({12, "DEMAND", 3, size_per_tensor});
        matrix.push_back(c2);
    }

    // C3: 4 Concurrent Demand Transfers
    {
        ExperimentConfig c3;
        c3.experiment_id = "C3";
        c3.workload_label = "CONCURRENT_DEMAND_4";
        c3.transfer_type = "DEMAND";
        c3.concurrency_streams = 4;
        c3.demand_streams = 4;
        c3.prefetch_streams = 0;
        c3.streams.push_back({13, "DEMAND", 3, size_per_tensor});
        c3.streams.push_back({14, "DEMAND", 3, size_per_tensor});
        c3.streams.push_back({15, "DEMAND", 3, size_per_tensor});
        c3.streams.push_back({16, "DEMAND", 3, size_per_tensor});
        matrix.push_back(c3);
    }

    // C4: 1 Prefetch Transfer
    {
        ExperimentConfig c4;
        c4.experiment_id = "C4";
        c4.workload_label = "PREFETCH_1";
        c4.transfer_type = "PREFETCH";
        c4.concurrency_streams = 1;
        c4.demand_streams = 0;
        c4.prefetch_streams = 1;
        c4.streams.push_back({20, "PREFETCH", 3, size_per_tensor});
        matrix.push_back(c4);
    }

    // C5: 2 Concurrent Prefetch Transfers
    {
        ExperimentConfig c5;
        c5.experiment_id = "C5";
        c5.workload_label = "CONCURRENT_PREFETCH_2";
        c5.transfer_type = "PREFETCH";
        c5.concurrency_streams = 2;
        c5.demand_streams = 0;
        c5.prefetch_streams = 2;
        c5.streams.push_back({21, "PREFETCH", 3, size_per_tensor});
        c5.streams.push_back({22, "PREFETCH", 3, size_per_tensor});
        matrix.push_back(c5);
    }

    // C6: 4 Concurrent Prefetch Transfers
    {
        ExperimentConfig c6;
        c6.experiment_id = "C6";
        c6.workload_label = "CONCURRENT_PREFETCH_4";
        c6.transfer_type = "PREFETCH";
        c6.concurrency_streams = 4;
        c6.demand_streams = 0;
        c6.prefetch_streams = 4;
        c6.streams.push_back({23, "PREFETCH", 3, size_per_tensor});
        c6.streams.push_back({24, "PREFETCH", 3, size_per_tensor});
        c6.streams.push_back({25, "PREFETCH", 3, size_per_tensor});
        c6.streams.push_back({26, "PREFETCH", 3, size_per_tensor});
        matrix.push_back(c6);
    }

    // C7: 1 Demand + 1 Prefetch (Mixed 2 streams)
    {
        ExperimentConfig c7;
        c7.experiment_id = "C7";
        c7.workload_label = "MIXED_1D_1P";
        c7.transfer_type = "MIXED";
        c7.concurrency_streams = 2;
        c7.demand_streams = 1;
        c7.prefetch_streams = 1;
        c7.streams.push_back({30, "DEMAND", 3, size_per_tensor});
        c7.streams.push_back({31, "PREFETCH", 3, size_per_tensor});
        matrix.push_back(c7);
    }

    // C8: 1 Demand + 3 Prefetches (Mixed 4 streams)
    {
        ExperimentConfig c8;
        c8.experiment_id = "C8";
        c8.workload_label = "MIXED_1D_3P";
        c8.transfer_type = "MIXED";
        c8.concurrency_streams = 4;
        c8.demand_streams = 1;
        c8.prefetch_streams = 3;
        c8.streams.push_back({40, "DEMAND", 3, size_per_tensor});
        c8.streams.push_back({41, "PREFETCH", 3, size_per_tensor});
        c8.streams.push_back({42, "PREFETCH", 3, size_per_tensor});
        c8.streams.push_back({43, "PREFETCH", 3, size_per_tensor});
        matrix.push_back(c8);
    }

    // C9: Size scaling - 1 tensor (~29.36 MB)
    {
        ExperimentConfig c9;
        c9.experiment_id = "C9";
        c9.workload_label = "SIZE_1_TENSOR";
        c9.transfer_type = "DEMAND";
        c9.concurrency_streams = 1;
        c9.demand_streams = 1;
        c9.prefetch_streams = 0;
        c9.streams.push_back({50, "DEMAND", 1, size_per_tensor});
        matrix.push_back(c9);
    }

    // C10: Size scaling - 2 tensors (~58.72 MB)
    {
        ExperimentConfig c10;
        c10.experiment_id = "C10";
        c10.workload_label = "SIZE_2_TENSORS";
        c10.transfer_type = "DEMAND";
        c10.concurrency_streams = 1;
        c10.demand_streams = 1;
        c10.prefetch_streams = 0;
        c10.streams.push_back({51, "DEMAND", 2, size_per_tensor});
        matrix.push_back(c10);
    }

    // C11: Size scaling - 3 tensors (Full expert 88.08 MB)
    {
        ExperimentConfig c11;
        c11.experiment_id = "C11";
        c11.workload_label = "SIZE_3_TENSORS";
        c11.transfer_type = "DEMAND";
        c11.concurrency_streams = 1;
        c11.demand_streams = 1;
        c11.prefetch_streams = 0;
        c11.streams.push_back({52, "DEMAND", 3, size_per_tensor});
        matrix.push_back(c11);
    }

    // Repeat C1 (Run 2) for stability verification
    {
        ExperimentConfig c1_rep;
        c1_rep.experiment_id = "C1_REP2";
        c1_rep.workload_label = "DEMAND_1_REPEAT";
        c1_rep.transfer_type = "DEMAND";
        c1_rep.concurrency_streams = 1;
        c1_rep.demand_streams = 1;
        c1_rep.prefetch_streams = 0;
        c1_rep.streams.push_back({10, "DEMAND", 3, size_per_tensor});
        matrix.push_back(c1_rep);
    }

    std::vector<CalibrationResult> results;
    auto total_campaign_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < matrix.size(); ++i) {
        const auto& exp = matrix[i];
        std::cout << "[RUNNING " << (i + 1) << "/" << matrix.size() << "] Experiment "
                  << exp.experiment_id << " (" << exp.workload_label 
                  << "): Streams=" << exp.concurrency_streams 
                  << ", Type=" << exp.transfer_type << "...\n" << std::flush;

        CalibrationResult res = run_calibration_experiment(exp, ddr5_config_path, device, cluster);
        results.push_back(res);

        std::cout << "  -> Finished in " << std::fixed << std::setprecision(2) << res.host_runtime_seconds << "s | "
                  << "Simulated Duration: " << (res.simulated_duration_ns / 1e6) << " ms | "
                  << "Aggregate BW: " << std::setprecision(2) << res.aggregate_effective_bandwidth_gbps << " GB/s | "
                  << "Per-Stream BW: " << res.per_stream_effective_bandwidth_gbps << " GB/s\n\n" << std::flush;
    }

    auto total_campaign_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_campaign_dur = total_campaign_end - total_campaign_start;

    std::cout << "======================================================================\n";
    std::cout << "CALIBRATION CAMPAIGN COMPLETE\n";
    std::cout << "Total Wall-Clock Time: " << std::fixed << std::setprecision(2) 
              << total_campaign_dur.count() << " seconds\n";
    std::cout << "======================================================================\n\n";

    // Write results to CSV and JSON
    std::string out_dir = "results/phase_4b_ddr5_calibration";
    std::filesystem::create_directories(out_dir);
    std::filesystem::create_directories("configs/prefetch");

    std::string csv_raw_path = out_dir + "/ddr5_transfer_calibration_raw.csv";
    std::string csv_prod_path = "configs/prefetch/ddr5_transfer_calibration.csv";
    std::string json_summary_path = out_dir + "/ddr5_transfer_calibration_summary.json";

    std::ofstream csv_raw(csv_raw_path);
    std::ofstream csv_prod(csv_prod_path);

    std::string header = "experiment_id,workload_label,concurrency_streams,transfer_type,demand_streams,prefetch_streams,"
                         "total_bytes,bytes_per_stream,simulated_start_time_ns,simulated_completion_time_ns,simulated_duration_ns,"
                         "aggregate_effective_bandwidth_gbps,per_stream_effective_bandwidth_gbps,"
                         "per_stream_min_latency_ns,per_stream_max_latency_ns,per_stream_mean_latency_ns,"
                         "ddr5_config,repetitions,host_runtime_seconds\n";

    csv_raw << header;
    csv_prod << header;

    for (const auto& r : results) {
        std::ostringstream row;
        row << r.experiment_id << ","
            << r.workload_label << ","
            << r.concurrency_streams << ","
            << r.transfer_type << ","
            << r.demand_streams << ","
            << r.prefetch_streams << ","
            << r.total_bytes << ","
            << r.bytes_per_stream << ","
            << std::fixed << std::setprecision(2) << r.simulated_start_time_ns << ","
            << r.simulated_completion_time_ns << ","
            << r.simulated_duration_ns << ","
            << std::setprecision(4) << r.aggregate_effective_bandwidth_gbps << ","
            << r.per_stream_effective_bandwidth_gbps << ","
            << std::setprecision(2) << r.per_stream_min_latency_ns << ","
            << r.per_stream_max_latency_ns << ","
            << r.per_stream_mean_latency_ns << ","
            << "\"" << r.ddr5_config << "\","
            << r.repetitions << ","
            << std::setprecision(4) << r.host_runtime_seconds << "\n";

        csv_raw << row.str();
        csv_prod << row.str();
    }

    csv_raw.close();
    csv_prod.close();

    // Write JSON summary
    std::ofstream json_file(json_summary_path);
    json_file << "{\n"
              << "  \"total_campaign_wall_clock_seconds\": " << total_campaign_dur.count() << ",\n"
              << "  \"ddr5_config\": \"" << ddr5_config_path << "\",\n"
              << "  \"experiments\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        json_file << "    {\n"
                  << "      \"experiment_id\": \"" << r.experiment_id << "\",\n"
                  << "      \"workload_label\": \"" << r.workload_label << "\",\n"
                  << "      \"concurrency_streams\": " << r.concurrency_streams << ",\n"
                  << "      \"transfer_type\": \"" << r.transfer_type << "\",\n"
                  << "      \"total_bytes\": " << r.total_bytes << ",\n"
                  << "      \"simulated_duration_ms\": " << (r.simulated_duration_ns / 1e6) << ",\n"
                  << "      \"aggregate_effective_bandwidth_gbps\": " << r.aggregate_effective_bandwidth_gbps << ",\n"
                  << "      \"per_stream_effective_bandwidth_gbps\": " << r.per_stream_effective_bandwidth_gbps << ",\n"
                  << "      \"per_stream_mean_latency_ms\": " << (r.per_stream_mean_latency_ns / 1e6) << ",\n"
                  << "      \"host_runtime_seconds\": " << r.host_runtime_seconds << "\n"
                  << "    }";
        if (i + 1 < results.size()) {
            json_file << ",";
        }
        json_file << "\n";
    }
    json_file << "  ]\n}\n";
    json_file.close();

    std::cout << "[OUTPUT] Raw CSV written to: " << csv_raw_path << "\n";
    std::cout << "[OUTPUT] Production CSV written to: " << csv_prod_path << "\n";
    std::cout << "[OUTPUT] Summary JSON written to: " << json_summary_path << "\n";

    return 0;
}

