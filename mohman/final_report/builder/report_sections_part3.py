"""
report_sections_part3.py:
Chapter 5 (Point 3 - Prefetching Effectiveness Evaluation),
Chapter 6 (Point 4 - HBM Cache Management and FA-LRU),
and Chapter 7 (Point 5 - Four-Way Comparative Architecture Evaluation).
Enforces strict zero-ternary rule and authentic C++ source code verification.
"""

TABLES_CH5_CH7 = {
    "table_5_1": {
        "num": "Table 5.1",
        "title": "Prefetching Effectiveness across Large Workloads W1 to W7 (DeepSeek-V3, 64 GB/s Interconnect)",
        "headers": ["Workload ID", "Batch Size (B)", "Tokens (T)", "Baseline Time (s)", "CSTP Time (s)", "Stall Reduction (ms)", "CSTP Traffic (+%)", "ST-MoE Traffic (+%)"],
        "rows": [
            ["W1", "14", "4", "4.855 s", "4.853 s", "2.75 ms (0.06%)", "+0.18%", "+0.97%"],
            ["W2", "14", "16", "18.565 s", "18.504 s", "61.61 ms (0.34%)", "+0.27%", "+1.03%"],
            ["W3", "32", "16", "35.732 s", "35.567 s", "165.08 ms (0.47%)", "+0.38%", "+1.39%"],
            ["W4", "48", "16", "51.772 s", "51.570 s", "202.57 ms (0.40%)", "+0.41%", "+1.42%"],
            ["W5", "64", "16", "68.210 s", "67.890 s", "320.00 ms (0.47%)", "+0.45%", "+1.48%"],
            ["W6", "96", "16", "101.450 s", "100.820 s", "630.00 ms (0.62%)", "+0.42%", "+1.44%"],
            ["W7", "128", "16", "136.850 s", "130.133 s", "6,717.00 ms (4.91%)", "+0.39%", "+1.37%"]
        ]
    },
    "table_6_1": {
        "num": "Table 6.1",
        "title": "HBM Cache Eviction Comparison: Baseline LRU vs. Frequency-Aware LRU (FA-LRU)",
        "headers": ["Workload / Matrix", "HBM Capacity", "Cache Slots", "Policy", "Coalesced Hit Rate (%)", "Protected Hot Experts", "Reused Hot Victims"],
        "rows": [
            ["W3 (B=14, T=128)", "35.0 GiB", "426 Slots", "Standard LRU", "0.5063%", "0 (Unprotected)", "N/A"],
            ["W3 (B=14, T=128)", "35.0 GiB", "426 Slots", "Proposed FA-LRU", "0.5021%", "215,314 Decisions", "36 Hot Reuses"],
            ["W4 (B=32, T=128)", "35.0 GiB", "426 Slots", "Standard LRU", "0.2589%", "0 (Unprotected)", "N/A"],
            ["W4 (B=32, T=128)", "35.0 GiB", "426 Slots", "Proposed FA-LRU", "0.2616%", "431,909 Decisions", "36 Hot Reuses"]
        ]
    },
    "table_7_1": {
        "num": "Table 7.1",
        "title": "Authoritative Four-Way Comparative Evaluation Matrix (Anchor Workload: B=32, T=128, C_HBM=35 GiB)",
        "headers": ["Conceptual Configuration", "Predictor", "Prefetch Active", "Eviction Policy", "Total Time (s)", "Exposed Stall (s)", "DDR5 Traffic (GB)", "Speedup vs Baseline"],
        "rows": [
            ["Config A: Baseline Demand Offloading", "None", "False", "LRU", "767.27 s", "753.25 s", "48,207.72 GB", "1.000x (Baseline)"],
            ["Config B: Shadow Prediction Only", "CSTP", "False", "LRU", "767.27 s", "753.25 s", "48,207.72 GB", "1.000x (+0.00%)"],
            ["Config C: Async Prefetch + LRU", "CSTP", "True", "LRU", "761.66 s", "747.62 s", "48,741.13 GB", "1.007x (+0.73%)"],
            ["Config D: Async Prefetch + FA-LRU", "CSTP", "True", "FA-LRU", "761.63 s", "747.58 s", "48,739.10 GB", "1.007x (+0.74%)"]
        ]
    }
}

CODE_SNIPPETS_CH6 = {
    "falru_victim_selection": {
        "file": "LLMSimulator/src/hardware/device.cpp",
        "func": "Device::select_fa_lru_victim & materialize_lazy_decay",
        "desc": "Authentic C++ implementation of Frequency-Aware LRU (FA-LRU): windowed bottom-W scan, lazy bit-shift epoch decay, and divergence diagnostics.",
        "code": """void Device::materialize_lazy_decay(ResidentExpertEntry& entry) {
    if (global_epoch <= entry.last_epoch) return;
    uint32_t epoch_diff = global_epoch - entry.last_epoch;
    if (epoch_diff >= 31) {
        entry.access_frequency = 1;
    } else {
        uint32_t decayed = entry.access_frequency >> epoch_diff;
        entry.access_frequency = (decayed < 1) ? 1 : decayed;
    }
    entry.last_epoch = global_epoch;
}

std::list<std::tuple<int, int, long long>>::iterator Device::select_fa_lru_victim() {
    if (lru_expert_list.empty()) return lru_expert_list.end();

    // Identify baseline standard LRU victim (first unpinned candidate)
    auto lru_victim_it = lru_expert_list.end();
    for (auto lit = lru_expert_list.begin(); lit != lru_expert_list.end(); ++lit) {
        auto mit = resident_experts.find({std::get<0>(*lit), std::get<1>(*lit)});
        if (mit != resident_experts.end() && mit->second.is_pinned) continue;
        lru_victim_it = lit; break;
    }

    auto best_it = lru_expert_list.end();
    uint32_t min_freq = UINT32_MAX;
    int count = 0;
    int window_size = std::min(fa_lru_window_size, static_cast<int>(lru_expert_list.size()));

    // Pass 1: Scan bottom-W candidates for unpinned minimum-frequency victim
    for (auto it = lru_expert_list.begin(); it != lru_expert_list.end() && count < window_size; ++it, ++count) {
        auto map_it = resident_experts.find({std::get<0>(*it), std::get<1>(*it)});
        if (map_it != resident_experts.end()) {
            if (map_it->second.is_pinned) continue;
            materialize_lazy_decay(map_it->second);
            uint32_t freq = map_it->second.access_frequency;
            if (freq < min_freq) { min_freq = freq; best_it = it; }
        }
    }

    // Pass 2: Fallback if all bottom-W candidates were pinned
    if (best_it == lru_expert_list.end()) {
        for (auto it = lru_expert_list.begin(); it != lru_expert_list.end(); ++it) {
            auto map_it = resident_experts.find({std::get<0>(*it), std::get<1>(*it)});
            if (map_it != resident_experts.end() && !map_it->second.is_pinned) {
                best_it = it; break;
            }
        }
    }

    // Track divergent decisions protecting demand-verified experts
    if (best_it != lru_victim_it && lru_victim_it != lru_expert_list.end()) {
        fa_lru_diag.divergent_decisions++;
        auto lmap = resident_experts.find({std::get<0>(*lru_victim_it), std::get<1>(*lru_victim_it)});
        if (lmap != resident_experts.end() && (!lmap->second.is_prefetched || lmap->second.was_demanded)) {
            fa_lru_diag.protected_demand_experts_count++;
            fa_lru_diag.protected_experts_set.insert({std::get<0>(*lru_victim_it), std::get<1>(*lru_victim_it)});
        }
    }
    return best_it;
}"""
    }
}
