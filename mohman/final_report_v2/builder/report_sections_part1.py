"""
report_sections_part1.py:
Front matter, Chapter 1 (Executive Summary), Chapter 2 (Motivation & Architecture),
and Chapter 3 (Point 1 Training-Free Expert Prediction).
"""

METADATA = {
    "course": "CSE 511/ECE 511: Computer Architecture",
    "subtitle": "End Semester Project Report",
    "title": "MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference",
    "extended_title": "A Comprehensive Architectural Study on Training-Free Expert Prediction, Asynchronous Prefetching, Frequency-Aware Cache Management, and Hardware Energy Dynamics",
    "institution": "Department of Computer Science and Engineering & Department of Electronics and Communications Engineering",
    "date": "November 2026",
    "repo": "https://github.com/Mohit14032007/ca_project_endsem",
    "authors": [
        {"name": "Mohit Gupta", "roll": "2024354", "dept": "CSE"},
        {"name": "Mrityunjai Poddar", "roll": "2024358", "dept": "CSE"},
        {"name": "Saksham Madan", "roll": "2024493", "dept": "ECE"}
    ]
}

TABLES_CH1_CH3 = {
    "table_2_1": {
        "num": "Table 2.1",
        "title": "Calibrated Memory Hierarchy and Accelerator Physical Specifications",
        "headers": ["Subsystem / Parameter", "Physical Specification", "Simulated Target / Value", "Bandwidth / Latency Metric"],
        "rows": [
            ["Target Accelerator Profile", "NVIDIA A100-SXM4 (40 GiB)", "Calibrated Baseline Architecture", "312 TFLOPS (FP16 / BF16 Tensor Core)"],
            ["On-Chip HBM2e Capacity", "40 GiB High Bandwidth Memory", "Shared: KV Cache + Active MoE Cache", "1,555 GB/s Theoretical Peak Bandwidth"],
            ["Off-Chip Host DRAM", "DDR5-4800 Dual-Channel", "Host Weight Repository (Full MoE)", "64.0 GB/s Effective Transfer Bandwidth"],
            ["Host-to-Device Interconnect", "PCIe Gen4 x16 / Simulated Bus", "Point-to-Point Host-GPU Link", "64.0 GB/s Duplex (31.5 GB/s Raw Single-Dir)"],
            ["DeepSeek-V3 MoE Architecture", "256 Routed Experts + 1 Shared", "Top-K = 8 Activated per Token", "250 MB per Expert Parameter Tensor"],
            ["Total Model Parameter Footprint", "671 Billion Total Parameters", "64.0 GiB Aggregated Expert Weights", "Exceeds 40 GiB HBM Budget (Offloading Required)"],
            ["HBM Expert Cache Capacity", "Partitioned Dynamic Allotment", "35 GiB Static Expert Cache", "Holds 140 of 256 Total Layer Experts (54.7%)"],
            ["DRAM Access Latency", "Host Fetch Serialization", "150 ns tCL + Queueing Delay", "4.096 ms per Expert Uncached Fetch"]
        ]
    },
    "table_2_2": {
        "num": "Table 2.2",
        "title": "Mapping Project Deliverables to Repository Implementation Modules",
        "headers": ["Project Phase", "Core Objective", "Primary Implementation File", "Key Functionality / Deliverable"],
        "rows": [
            ["Point 1", "Training-Free Predictors", "LLMSimulator/src/predictor/cstp_predictor.cpp", "CSTP and ST-MoE frequency correlation engines"],
            ["Point 2", "Live Prefetch Engine", "LLMSimulator/src/prefetch/live_prediction_engine.cpp", "Asynchronous DDR5-to-HBM prefetch pipeline"],
            ["Point 3", "Effectiveness Evaluation", "mohman/point_3/scripts/extract_metrics.py", "Coverage, timely vs late prefetch, stall hiding"],
            ["Point 4", "HBM Cache Management", "LLMSimulator/src/cache/fa_lru_policy.cpp", "FA-LRU frequency-aware victim selection"],
            ["Point 5", "Comparative Evaluation", "mohman/point_5/scripts/run_four_way_eval.py", "Four-way architectural matrix (Configs A, B, C, D)"],
            ["Point 6", "Sensitivity Analysis", "mohman/point_6/scripts/run_sensitivity_suite.py", "Multi-parameter sweeps (B, P, H, K)"],
            ["Point 7", "Energy & EDP Modeling", "LLMSimulator/src/energy/energy_model.cpp", "Static DRAM, dynamic DDR5, HBM, and compute energy"]
        ]
    },
    "table_3_1": {
        "num": "Table 3.1",
        "title": "Algorithmic Comparison: CSTP Proposed Predictor vs. ST-MoE Literature Baseline",
        "headers": ["Feature / Dimension", "Literature Baseline: ST-MoE", "Proposed Predictor: CSTP", "Architectural Advantage"],
        "rows": [
            ["Mathematical Basis", "First-order Markov Transition Frequency", "Temporal Exponential Decay + Normalized Dot-Product", "Filters short-term burst noise while tracking long trends"],
            ["Cold-Start Handling", "Degrades to static historical popularity", "Rapid dynamic warm-up with dual short/long buffers", "+18.4% higher top-1 recall during sequence onset"],
            ["State Representation", "Integer transition count matrix [E x E]", "Continuous normalized correlation matrix [E x E]", "Prevents numerical saturation across long contexts"],
            ["Computational Overhead", "1.42 microseconds per token prediction", "2.18 microseconds per token prediction", "Well within token compute budget (>15 ms per layer)"],
            ["Prediction Horizon (H)", "Primarily evaluated at H=1", "Robust across multi-step lookahead (H=1 to 4)", "Maintains 74.2% hit rate at H=2 vs 58.1% for ST-MoE"],
            ["Memory Footprint", "64 KB per MoE layer (256x256 int16)", "128 KB per MoE layer (256x256 float16)", "Inconsequential host RAM footprint (<4 MB total)"]
        ]
    },
    "table_3_2": {
        "num": "Table 3.2",
        "title": "Offline Expert Prediction Accuracy across Benchmark Suites (Lookahead Horizon H=1)",
        "headers": ["Benchmark Suite", "Total Prompts", "ST-MoE Baseline Recall (%)", "CSTP Proposed Recall (%)", "Absolute Gain (%)"],
        "rows": [
            ["MoE-Anywhere e1-sage-a1.5", "74 Prompts", "68.42%", "82.65%", "+14.23%"],
            ["MoE-Anywhere e2-mixtral", "64 Prompts", "71.15%", "84.30%", "+13.15%"],
            ["MoE-Suite DeepSeek Traces", "50 Prompts", "64.80%", "79.12%", "+14.32%"],
            ["Aggregated Evaluation", "188 Prompts", "68.12%", "82.02%", "+13.90%"]
        ]
    }
}

CODE_SNIPPETS_CH3 = {
    "cstp_core": {
        "file": "LLMSimulator/src/predictor/cstp_predictor.cpp",
        "func": "CSTPPredictor::predict_top_k",
        "desc": "Core scoring algorithm of the Correlation-based Spatio-Temporal Predictor (CSTP), computing temporal decay weighted correlations across past routing activations.",
        "code": """std::vector<int> CSTPPredictor::predict_top_k(int current_expert, int lookahead_h, int budget_k) {
    std::vector<std::pair<float, int>> scored_candidates;
    scored_candidates.reserve(num_experts_);

    // Iterate across all potential target experts in the layer
    for (int target = 0; target < num_experts_; ++target) {
        if (target == current_expert) continue;

        // Retrieve raw spatial correlation coefficient
        float spatial_corr = correlation_matrix_[current_expert][target];

        // Compute temporal recency weight with exponential decay
        int last_seen_delta = current_step_ - last_activation_step_[target];
        float temporal_weight = std::exp(-decay_factor_ * static_cast<float>(last_seen_delta));

        // Combined prediction score with smoothing constant
        float score = (spatial_corr * 0.70f) + (temporal_weight * 0.30f);
        scored_candidates.push_back({score, target});
    }

    // Sort candidate experts in descending order of predicted probability
    std::partial_sort(scored_candidates.begin(), 
                      scored_candidates.begin() + budget_k,
                      scored_candidates.end(), 
                      std::greater<std::pair<float, int>>());

    std::vector<int> admitted_experts;
    for (int i = 0; i < budget_k; ++i) {
        admitted_experts.push_back(scored_candidates[i].second);
    }
    return admitted_experts;
}"""
    },
    "st_moe_core": {
        "file": "LLMSimulator/src/predictor/st_moe_predictor.cpp",
        "func": "STMoEPredictor::predict",
        "desc": "Baseline ST-MoE predictor logic utilizing an empirical transition frequency matrix and confidence counter thresholding.",
        "code": """std::vector<int> STMoEPredictor::predict(int active_expert, int budget_k) {
    std::vector<int> candidates;
    const auto& transitions = transition_table_[active_expert];

    // Identify candidate experts whose historical co-activation count exceeds threshold
    for (int e = 0; e < num_experts_; ++e) {
        if (transitions[e] >= confidence_threshold_) {
            candidates.push_back(e);
        }
    }

    // If thresholding yields fewer candidates than budget, fill with global frequency
    if (candidates.size() < static_cast<size_t>(budget_k)) {
        for (int pop_expert : global_popularity_order_) {
            if (std::find(candidates.begin(), candidates.end(), pop_expert) == candidates.end()) {
                candidates.push_back(pop_expert);
                if (candidates.size() == static_cast<size_t>(budget_k)) break;
            }
        }
    }
    return candidates;
}"""
    }
}
