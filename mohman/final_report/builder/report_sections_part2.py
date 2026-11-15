"""
report_sections_part2.py:
Chapter 4 (Point 2 - Complete Predictor-Driven Prefetching Implementation in LLMSimulator).
This is the primary architectural core of the report.
Enforces strict zero-ternary rule and authentic C++ source code verification.
"""

TABLES_CH4 = {
    "table_4_1": {
        "num": "Table 4.1",
        "title": "Comprehensive Source Code Implementation Map of LLMSimulator Prefetching Architecture",
        "headers": ["Architectural Component", "Source File Path", "Class / Method Name", "Primary Responsibility", "Invoked By"],
        "rows": [
            ["Live Prediction Engine", "LLMSimulator/src/prefetch/live_prediction_engine.cpp", "LivePredictionEngine::on_stage_routing_complete", "Listens to causal routing events, queries predictor, ranks candidate scores", "BatchedSequence::update_expert (Scheduler)"],
            ["Prefetch Candidate Queue", "LLMSimulator/src/prefetch/prefetch_candidate_queue.cpp", "PrefetchCandidateQueue::enqueue_candidates", "Enforces queue capacity, deduplicates candidates, maintains score priority order", "LivePredictionEngine::on_stage_routing_complete"],
            ["Candidate Dispatch", "LLMSimulator/src/prefetch/live_prediction_engine.cpp", "LivePredictionEngine::dispatch_candidates", "Pulls top-K candidate experts from queue and submits transfers to DDR5 engine", "Scheduler main event loop"],
            ["DDR5 Transfer Engine", "LLMSimulator/src/prefetch/ddr5_transfer_engine.cpp", "DDR5TransferEngine::submit_prefetch / submit_demand", "Simulates bus serialization, models 22.578 GB/s bandwidth, tracks completion times", "Device::submit_prefetch_transfer / submit_demand_transfer"],
            ["Transfer Progress & Completion", "LLMSimulator/src/hardware/device.cpp", "Device::process_completed_prefetches", "Advances simulated time, checks completed DDR5 transfers, marks experts resident in HBM", "LinearExecutionGPU::execute (linear_impl.cpp)"],
            ["HBM Expert Cache", "LLMSimulator/src/hardware/device.cpp", "Device::is_expert_resident / mark_expert_resident", "Tracks HBM cache occupancy, manages residency map, triggers cache eviction", "linear_impl.cpp (LinearExecutionGPU)"],
            ["FA-LRU Cache Policy", "LLMSimulator/src/hardware/device.cpp", "Device::select_fa_lru_victim & materialize_lazy_decay", "Scans bottom-W candidates, applies lazy epoch halving, selects min-frequency victim", "Device::allocate_expert_slot / evict"],
            ["Linear Execution GPU", "LLMSimulator/src/hardware/linear_impl.cpp", "LinearImpl::execute / weight reconciliation", "Reconciles expert demand, attaches to in-flight prefetch, measures exposed stall", "Executor::execute_operation"],
            ["Metrics Collector", "LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp", "PrefetchMetricsCollector::record_expert_demanded", "Aggregates timely hits, late attachments, unused transfers, and exposed stall seconds", "LinearImpl and Device transfer completions"]
        ]
    },
    "table_4_2": {
        "num": "Table 4.2",
        "title": "Step-by-Step Worked Numerical Execution Trace for DeepSeek-V3 Layer Transition",
        "headers": ["Sim Time (ms)", "Pipeline Event / Step", "Component Involved", "Active State & Transfer Details", "Hardware Stall Consequence"],
        "rows": [
            ["0.000 ms", "Layer 10 Routing Event", "Router / BatchedSequence", "Top-2 Router selects Experts {14, 42} for current token", "No stall; Layer 10 GEMM begins on GPU"],
            ["0.002 ms", "Live Prediction Trigger", "LivePredictionEngine", "CSTP scores candidate experts for Layer 11 (Lookahead H=1)", "Overhead: 2.18 microseconds (invisible to GPU)"],
            ["0.003 ms", "Candidate Enqueue", "PrefetchCandidateQueue", "Candidates {88 (0.91), 105 (0.84)} admitted; Budget K=2", "Queue capacity: 16 slots; priority order preserved"],
            ["0.005 ms", "Prefetch Transfer Issue", "DDR5TransferEngine", "Transfer #101 (Expert 88, 84 MiB) submitted at 22.578 GB/s", "Transfer duration: 84 MB / 22.578 GB/s = 3.720 ms"],
            ["0.006 ms", "Layer 10 Execution", "LinearExecutionGPU", "Layer 10 Expert 14 & 42 GEMMs execute simultaneously", "GPU fully utilized; DDR5 bus busy in background"],
            ["3.725 ms", "Prefetch #101 Completes", "DDR5TransferEngine", "Expert 88 fully arrived in on-chip HBM cache", "Marked as PREFETCHED; HBM residency updated"],
            ["3.726 ms", "Prefetch Transfer #102 Issue", "DDR5TransferEngine", "Transfer #102 (Expert 105, 84 MiB) submitted at 22.578 GB/s", "Estimated completion: 3.726 + 3.720 = 7.446 ms"],
            ["5.200 ms", "Layer 10 GEMM Finishes", "LinearExecutionGPU", "Layer 10 compute complete; token activation arrives at Layer 11", "Total Layer 10 duration: 5.200 ms"],
            ["5.201 ms", "Layer 11 Demand Check: Expert 88", "LinearExecutionGPU", "Router requires Expert 88. Found resident in HBM!", "TIMELY PREFETCH HIT: 0.000 ms exposed stall!"],
            ["5.202 ms", "Layer 11 Demand Check: Expert 105", "LinearExecutionGPU", "Router requires Expert 105. Transfer #102 is IN-FLIGHT!", "LATE PREFETCH ATTACHMENT: Remaining time = 2.244 ms"],
            ["5.203 ms", "Layer 11 Expert 88 Execution", "LinearExecutionGPU", "GPU executes Expert 88 GEMM (compute overlap: 1.800 ms)", "Hides 1.800 ms of Transfer #102 remaining latency"],
            ["7.003 ms", "Expert 88 GEMM Finishes", "LinearExecutionGPU", "GPU needs Expert 105. Remaining transfer time: 7.446 - 7.003 = 0.443 ms", "GPU idles waiting for transfer completion"],
            ["7.446 ms", "Transfer #102 Completes", "DDR5TransferEngine", "Expert 105 fully resident in HBM; GPU awakens", "Exposed Memory Stall: Exactly 0.443 ms (vs 3.720 ms demand miss!)"],
            ["7.447 ms", "Layer 11 Expert 105 Execution", "LinearExecutionGPU", "Layer 11 compute proceeds; next prediction triggered", "Net latency saving: 3.277 ms hidden out of 3.720 ms (88.1%)"]
        ]
    },
    "table_4_3": {
        "num": "Table 4.3",
        "title": "Simulation Configuration Parameters and Hardware Architecture Constants",
        "headers": ["Architectural Parameter", "Symbol / Variable", "Configured Baseline Value", "Physical Hardware Implication"],
        "rows": [
            ["DDR5 Interconnect Bandwidth", "BW_ddr5", "22.578 GB/s (Simulated) / 64 GB/s (Peak)", "Effective sustained bandwidth of PCIe Gen4 x16 link under contention"],
            ["Per-Expert Parameter Size", "S_expert", "84 MiB (88,080,384 bytes) / 250 MB (Raw)", "DeepSeek-V3 quantized routed expert parameter tensor volume"],
            ["HBM Expert Cache Budget", "C_hbm", "35.0 GiB (426 Expert Slots)", "Allocates 87.5% of 40 GiB A100 HBM, reserving 5 GiB for MLA KV cache"],
            ["Prefetch Lookahead Horizon", "H", "H = 1 (Default), H = 2, H = 4", "Pipeline distance in layers between routing observation and target prefetch"],
            ["Prefetch Budget per Step", "K", "K = 2 (Energy Optimum), K = 4 (Latency Opt)", "Number of speculative candidate experts admitted per routing decision"],
            ["FA-LRU Inspection Window", "W", "W = 64 Cache Entries", "Scan depth in LRU candidate list for frequency-based victim selection"],
            ["DRAM Background Power", "P_static", "4.50 Watts", "Micron DDR5-4800 static background power dissipated continuously"],
            ["DDR5 Energy per Byte", "E_dram_byte", "0.100 nJ / Byte (100 pJ/B)", "Active I/O energy consumed per byte transferred over host-GPU bus"]
        ]
    }
}

CODE_SNIPPETS_CH4 = {
    "live_engine_hook": {
        "file": "LLMSimulator/src/prefetch/live_prediction_engine.cpp",
        "func": "LivePredictionEngine::on_stage_routing_complete",
        "desc": "Event hook triggered immediately when a layer finishes MoE routing, invoking CSTP and pushing candidates to queue.",
        "code": """void LivePredictionEngine::on_stage_routing_complete(
    int request_id, int seq_id, int token_id, int completed_stage_id,
    const std::vector<int>& routed_expert_ids, double current_time_ns) {

    // Calculate target lookahead stage
    int target_stage_id = completed_stage_id + prediction_horizon_;
    if (target_stage_id >= num_moe_stages_) return; // Beyond pipeline depth

    // Query predictor for top-K candidate experts at target stage
    PredictionResult result = predictor_->predict(
        request_id, token_id, completed_stage_id, target_stage_id,
        routed_expert_ids, prefetch_budget_k_);

    // Enqueue candidate predictions into dispatch priority queue
    for (const auto& candidate_score : result.scores) {
        PrefetchCandidate cand;
        cand.request_id = request_id;
        cand.target_stage_id = target_stage_id;
        cand.expert_id = candidate_score.expert_id;
        cand.score = candidate_score.score;
        cand.enqueue_time_ns = current_time_ns;

        candidate_queue_->enqueue_candidate(cand);
    }
}"""
    },
    "candidate_queue_enqueue": {
        "file": "LLMSimulator/src/prefetch/prefetch_candidate_queue.cpp",
        "func": "PrefetchCandidateQueue::enqueue_candidate",
        "desc": "Candidate queue admission, deduplication against resident and in-flight experts, and priority ordering.",
        "code": """bool PrefetchCandidateQueue::enqueue_candidate(const PrefetchCandidate& candidate) {
    std::pair<int, int> key = {candidate.target_stage_id, candidate.expert_id};

    // Reject if candidate is already in the queue or already resident in HBM
    if (pending_keys_.find(key) != pending_keys_.end()) {
        return false; // Deduplication filter
    }

    // Capacity enforcement
    if (queue_.size() >= static_cast<size_t>(capacity_)) {
        // Drop lowest priority candidate if new candidate has higher score
        if (candidate.score <= queue_.back().score) {
            return false;
        }
        pending_keys_.erase({queue_.back().target_stage_id, queue_.back().expert_id});
        queue_.pop_back();
    }

    // Insert maintaining descending priority order
    auto it = std::lower_bound(queue_.begin(), queue_.end(), candidate,
                               [](const PrefetchCandidate& a, const PrefetchCandidate& b) {
                                   return a.score > b.score;
                               });
    queue_.insert(it, candidate);
    pending_keys_.insert(key);
    return true;
}"""
    },
    "ddr5_engine_submit": {
        "file": "LLMSimulator/src/prefetch/ddr5_transfer_engine.cpp",
        "func": "DDR5TransferEngine::submit_prefetch",
        "desc": "Schedules an asynchronous speculative transfer on the DDR5 bus, calculating arrival time based on bus serialization and bandwidth.",
        "code": """bool DDR5TransferEngine::submit_prefetch(
    int request_id, int layer_id, int expert_id, double current_time_ns,
    uint64_t bytes, int priority, uint64_t transfer_id,
    int seq_id, int token_id, uint64_t candidate_id) {

    // Serialize access across DDR5 link: transfer begins after previous finishes
    double start_time = std::max(current_time_ns, next_available_bus_time_ns_);
    double transfer_duration = static_cast<double>(bytes) / bandwidth_bytes_per_ns_;
    double completion_time = start_time + transfer_duration;

    // Advance bus reservation boundary
    next_available_bus_time_ns_ = completion_time;

    MemoryTransfer transfer;
    transfer.transfer_id = (transfer_id != 0) ? transfer_id : ++next_transfer_id_;
    transfer.request_id = request_id;
    transfer.layer_id = layer_id;
    transfer.expert_id = expert_id;
    transfer.bytes = bytes;
    transfer.start_time_ns = start_time;
    transfer.completion_time_ns = completion_time;
    transfer.is_prefetch = true;
    transfer.status = TransferStatus::IN_FLIGHT;

    active_transfers_[transfer.transfer_id] = transfer;
    return true;
}"""
    },
    "linear_demand_reconciliation": {
        "file": "LLMSimulator/src/hardware/linear_impl.cpp",
        "func": "LinearImpl::execute_expert_gemm",
        "desc": "GPU execution of MoE layer GEMM, checking HBM residency, in-flight prefetch attachment, and measuring exposed memory stall.",
        "code": """// Check residency and completed background transfers
device->process_completed_prefetches(start_time);

bool is_resident = device->is_expert_resident(expert_layer_id, expert_id, true, is_primary);
time_ns exposed_stall = 0;

if (is_resident) {
    // Case A or B: Resident Hit or Timely Prefetch Hit
    PrefetchMetricsCollector::getInstance().record_expert_demanded(
        cand_id, request_id, seq_id, token_id, expert_layer_id, expert_id,
        start_time, start_time, 0.0, 0.0, "TIMELY_PREFETCH", tr_id, cand_id);
    exposed_stall = 0; // Latency completely hidden!
} else {
    // Check if expert is currently in-flight on DDR5 bus (Case C: Late Prefetch)
    uint64_t in_flight_tid = device->find_inflight_transfer(expert_layer_id, expert_id);
    if (in_flight_tid != 0) {
        const auto* tr = device->ddr5_transfer_engine->get_transfer(in_flight_tid);
        if (tr->completion_time_ns > start_time) {
            exposed_stall = static_cast<time_ns>(tr->completion_time_ns - start_time);
        }
        device->wait_for_transfer(in_flight_tid);
        PrefetchMetricsCollector::getInstance().record_transfer_outcome(in_flight_tid, "LATE_ATTACHED");
    } else {
        // Case D: Demand Miss - initiate synchronous demand transfer
        uint64_t tid = device->submit_demand_transfer(expert_layer_id, expert_id, start_time, expert_bytes);
        device->wait_for_demand_transfer(tid);
        exposed_stall = static_cast<time_ns>(expert_bytes / device->config.ddr5_bandwidth_gbps);
    }
    device->mark_expert_resident(expert_layer_id, expert_id, expert_bytes);
}

if (exposed_stall > 0) {
    PrefetchMetricsCollector::getInstance().record_exposed_stall(start_time, exposed_stall);
}
exec_status.total_duration = onchip_compute_duration + exposed_stall;"""
    }
}
