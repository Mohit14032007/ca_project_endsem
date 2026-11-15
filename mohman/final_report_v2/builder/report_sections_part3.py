"""
report_sections_part3.py:
Chapter 5 (Point 3 - Prefetching Effectiveness Evaluation),
Chapter 6 (Point 4 - HBM Cache Management and FA-LRU),
and Chapter 7 (Point 5 - Four-Way Comparative Architecture Evaluation).
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
            ["W3 (B=14, T=128)", "35.0 GiB", "426 Slots", "Proposed FA-LRU", "0.5021%", "215,314 Experts", "36 Hot Reuses"],
            ["W4 (B=32, T=128)", "35.0 GiB", "426 Slots", "Standard LRU", "0.2589%", "0 (Unprotected)", "N/A"],
            ["W4 (B=32, T=128)", "35.0 GiB", "426 Slots", "Proposed FA-LRU", "0.2616%", "431,909 Experts", "36 Hot Reuses"]
        ]
    },
    "table_7_1": {
        "num": "Table 7.1",
        "title": "Authoritative Four-Way Comparative Evaluation Matrix (Workload W4: B=32, T=128, C_HBM=35 GiB)",
        "headers": ["Conceptual Configuration", "Predictor", "Prefetch Active", "Eviction Policy", "Total Time (s)", "Exposed Stall (s)", "DDR5 Traffic (GB)", "Speedup vs Baseline"],
        "rows": [
            ["Config A: Baseline Demand Offloading", "None", "False", "LRU", "766.97 s", "753.25 s", "48,207.72 GB", "1.000x (Baseline)"],
            ["Config B: Shadow Prediction Only", "CSTP", "False", "LRU", "766.97 s", "753.25 s", "48,207.72 GB", "1.000x (+0.00%)"],
            ["Config C: Async Prefetch + LRU", "CSTP", "True", "LRU", "761.61 s", "747.86 s", "48,738.84 GB", "1.007x (+0.70%)"],
            ["Config D: Async Prefetch + FA-LRU", "CSTP", "True", "FA-LRU", "761.58 s", "747.83 s", "48,736.81 GB", "1.007x (+0.70%)"]
        ]
    }
}

CODE_SNIPPETS_CH6 = {
    "falru_victim_selection": {
        "file": "mohman/point_4/src/fa_lru_policy.cpp",
        "func": "FALRUPolicy::select_victim",
        "desc": "Victim selection logic of FA-LRU, balancing frequency protection counters against speculative pollution penalties.",
        "code": """int FALRUPolicy::select_victim(const std::vector<CacheEntry>& entries, double current_time) {
    int best_victim = -1;
    double lowest_score = std::numeric_limits<double>::max();

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        if (entry.is_pinned) continue; // Pinned entries cannot be evicted

        // Compute normalized recency (age in cache)
        double recency_age = current_time - entry.last_access_time;
        
        // Frequency protection factor
        double frequency_weight = static_cast<double>(entry.demand_hits);

        // Speculative pollution penalty: prefetched but never demanded entries get penalized
        double speculative_penalty = 1.0;
        if (entry.is_speculative && !entry.was_demanded) {
            speculative_penalty = 0.10; // Highly vulnerable to rapid eviction
        }

        // Composite victim score: low score indicates prime eviction candidate
        double score = (frequency_weight / (recency_age + 1e-5)) * speculative_penalty;

        if (score < lowest_score) {
            lowest_score = score;
            best_victim = static_cast<int>(i);
        }
    }
    return best_victim;
}"""
    }
}
