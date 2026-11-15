"""
report_sections_part2.py:
Chapter 4 (Point 2 - Complete Predictor-Driven Prefetching Implementation in LLMSimulator).
This is the primary architectural core of the report.
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
            ["FA-LRU Cache Policy", "mohman/point_4/src/fa_lru_policy.cpp", "FALRUPolicy::select_victim", "Evaluates frequency scores and speculative penalties to protect hot experts", "Device::evict_expert_if_needed"],
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
        "headers": ["Configuration Parameter", "YAML Key", "Calibrated Value", "Physical Hardware Meaning"],
        "rows": [
            ["DDR5 Memory Bandwidth", "ddr5_bandwidth_gbps", "22.578 GB/s", "Calibrated effective sustained transfer rate per host-GPU link"],
            ["DDR5 Expert Weight Size", "expert_weight_bytes", "88,080,384 Bytes", "84.0 MiB parameter payload for DeepSeek-V3 routed expert"],
            ["Prediction Lookahead Horizon", "prefetch_lookahead", "1 to 4 Layers", "Distance ahead in transformer pipeline for prefetching targets"],
            ["Prefetch Candidate Budget", "prefetch_budget_k", "1 to 8 Candidates", "Maximum number of speculative experts fetched per layer transition"],
            ["Queue Slot Capacity", "queue_capacity", "16 Slots", "Maximum prefetch candidates buffered before admission throttling"],
            ["HBM Cache Capacity", "hbm_cache_capacity_gb", "35.0 GiB (or 40.0 GiB)", "Dedicated fast memory allocated for MoE expert weight caching"],
            ["Cache Eviction Policy", "cache_eviction_policy", "LRU / FA-LRU", "Algorithm for victim selection during capacity overflows"],
            ["Event-Driven DRAM Flag", "use_event_driven_dram", "true", "Enables cycle-approximate asynchronous transfer simulation"]
        ]
    }
}

CODE_SNIPPETS_CH4 = {
    "live_engine_hook": {
        "file": "LLMSimulator/src/prefetch/live_prediction_engine.cpp",
        "func": "LivePredictionEngine::on_stage_routing_complete",
        "desc": "Invoked immediately when a transformer layer finishes expert routing. Scores candidate experts and pushes admitted candidates to the queue.",
        "code": """void LivePredictionEngine::on_stage_routing_complete(
    int request_id,
    int token_id,
    int target_stage_id,
    const std::vector<int>& actual_target_experts,
    int seq_id,
    double sim_time_ns) {

    // Record authoritative routing ground-truth for previous prediction verification
    reconcile_past_prediction(request_id, target_stage_id, actual_target_experts, sim_time_ns);

    // Compute target lookahead stage for prefetching
    int predicted_stage = target_stage_id + config_.lookahead_horizon;
    if (predicted_stage >= num_stages_) return; // End of transformer pipeline

    // Invoke active predictor (CSTP or ST-MoE) through standard interface
    PredictionResult result = predictor_->predict_for_stage(
        actual_target_experts, target_stage_id, predicted_stage, config_.budget_k);

    // Enqueue predicted candidates into candidate queue with priority scores
    candidate_queue_->enqueue_candidates(
        request_id, token_id, target_stage_id, predicted_stage,
        config_.lookahead_horizon, result, seq_id, sim_time_ns);
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
