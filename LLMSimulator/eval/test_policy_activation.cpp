#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "hardware/device.h"
#include "prefetch/prefetch_config.h"
#include "prefetch/prefetch_metrics_collector.h"

using namespace llm_system;

struct EvictionRecord {
    int step = 0;
    std::string policy;
    std::pair<int, int> accessed_key;
    std::string outcome; // HIT or MISS
    std::vector<std::pair<int, int>> candidates_in_window;
    std::vector<uint32_t> candidate_frequencies;
    std::pair<int, int> selected_victim;
    bool victim_diverged = false;
};

void run_policy_trace(
    EvictionPolicy policy,
    int window_size,
    const std::vector<std::pair<int, int>>& access_sequence,
    const std::vector<int>& epoch_sequence,
    std::vector<EvictionRecord>& records,
    int& total_hits,
    int& total_misses,
    int& total_evictions,
    int& total_reloads,
    std::vector<std::pair<int, int>>& final_residents) {

    Device dev;
    dev.cache_capacity_bytes = 160; // 4 experts of 40 bytes
    dev.eviction_policy = policy;
    dev.fa_lru_window_size = window_size;
    dev.reset_expert_cache();

    total_hits = 0;
    total_misses = 0;
    total_evictions = 0;
    total_reloads = 0;

    std::string policy_str = "LRU";
    if (policy == EvictionPolicy::FA_LRU) {
        policy_str = "FA-LRU";
    }

    for (size_t i = 0; i < access_sequence.size(); ++i) {
        auto key = access_sequence[i];
        int layer = key.first;
        int expert = key.second;
        dev.global_epoch = epoch_sequence[i];

        EvictionRecord rec;
        rec.step = static_cast<int>(i);
        rec.policy = policy_str;
        rec.accessed_key = key;

        bool is_hit = dev.is_expert_resident(layer, expert, true, true);
        if (is_hit) {
            total_hits++;
            rec.outcome = "HIT";
            records.push_back(rec);
            continue;
        }

        total_misses++;
        rec.outcome = "MISS";

        // Check if this is a reload
        if (dev.previously_evicted_experts.count(key) > 0) {
            total_reloads++;
            dev.previously_evicted_experts.erase(key);
            dev.reloaded_experts_count++;
        }

        // Before admission, if cache is full, record candidates and victim decision
        if (dev.resident_bytes + 40 > dev.cache_capacity_bytes && !dev.lru_expert_list.empty()) {
            total_evictions++;

            // Record bottom-W candidates and their decayed frequencies
            int w_count = 0;
            int effective_w = window_size;
            int list_sz = static_cast<int>(dev.lru_expert_list.size());
            if (effective_w > list_sz) {
                effective_w = list_sz;
            }

            for (auto it = dev.lru_expert_list.begin(); it != dev.lru_expert_list.end() && w_count < effective_w; ++it, ++w_count) {
                int cl = std::get<0>(*it);
                int ce = std::get<1>(*it);
                rec.candidates_in_window.push_back({cl, ce});
                auto mit = dev.resident_experts.find({cl, ce});
                if (mit != dev.resident_experts.end()) {
                    dev.materialize_lazy_decay(mit->second);
                    rec.candidate_frequencies.push_back(mit->second.access_frequency);
                } else {
                    rec.candidate_frequencies.push_back(0);
                }
            }

            std::list<std::tuple<int, int, long long>>::iterator vit;
            if (policy == EvictionPolicy::FA_LRU) {
                vit = dev.select_fa_lru_victim();
            } else {
                vit = dev.lru_expert_list.begin();
            }

            if (vit != dev.lru_expert_list.end()) {
                rec.selected_victim = {std::get<0>(*vit), std::get<1>(*vit)};
            }
        }

        // Admit expert
        dev.mark_expert_resident(layer, expert, 40, ExpertClassification::DEMAND_VERIFIED, false);
        records.push_back(rec);
    }

    for (const auto& item : dev.lru_expert_list) {
        final_residents.push_back({std::get<0>(item), std::get<1>(item)});
    }
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "POINT 4 PHASE 2.5: POLICY ACTIVATION EXPERIMENT (LRU vs FA-LRU)\n";
    std::cout << "======================================================================\n";

    // Controlled workload sequence:
    // Capacity = 4 experts. Size = 40 bytes each. Max capacity = 160 bytes.
    // Heavy hitters: (0, 1) and (0, 2).
    // Ephemeral experts: (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8).

    std::vector<std::pair<int, int>> access_seq = {
        // Step 0-3 (Epoch 0): Warm up cache with 4 initial experts
        {0, 1}, {0, 2}, {0, 3}, {0, 4},
        // Step 4-5 (Epoch 0): Re-access heavy-hitters (0, 1) and (0, 2) -> F becomes 2
        {0, 1}, {0, 2},
        // Step 6-7 (Epoch 1): Epoch advances to 1. Re-access (0, 1) twice -> F becomes 3
        {0, 1}, {0, 1},
        // Step 8-10 (Epoch 1): Ephemeral burst (0, 5), (0, 6), (0, 7) pushes (0, 1) to oldest position!
        {0, 5}, {0, 6}, {0, 7},
        // Step 11 (Epoch 1): Critical Eviction Step! Access (0, 8).
        // Candidate window holds: (0, 1) [F=3, oldest], (0, 5) [F=1], (0, 6) [F=1], (0, 7) [F=1].
        // LRU evicts (0, 1) [oldest].
        // FA-LRU evicts (0, 5) [min freq].
        {0, 8},
        // Step 12 (Epoch 1): Re-access heavy-hitter (0, 1).
        // LRU experiences DEMAND MISS + RELOAD!
        // FA-LRU experiences RESIDENT HIT!
        {0, 1},
        // Step 13 (Epoch 1): Access (0, 5).
        // LRU hits (0, 5); FA-LRU misses (0, 5).
        {0, 5}
    };

    std::vector<int> epoch_seq = {
        0, 0, 0, 0, // 0-3
        0, 0,       // 4-5
        1, 1,       // 6-7
        1, 1, 1,    // 8-10
        1,          // 11 (critical eviction)
        1,          // 12 (heavy-hitter test)
        1           // 13 (ephemeral test)
    };

    std::vector<EvictionRecord> lru_records;
    std::vector<EvictionRecord> falru_records;
    int lru_hits = 0, lru_misses = 0, lru_evicts = 0, lru_reloads = 0;
    int falru_hits = 0, falru_misses = 0, falru_evicts = 0, falru_reloads = 0;
    std::vector<std::pair<int, int>> lru_final;
    std::vector<std::pair<int, int>> falru_final;

    run_policy_trace(EvictionPolicy::LRU, 4, access_seq, epoch_seq,
                     lru_records, lru_hits, lru_misses, lru_evicts, lru_reloads, lru_final);

    run_policy_trace(EvictionPolicy::FA_LRU, 4, access_seq, epoch_seq,
                     falru_records, falru_hits, falru_misses, falru_evicts, falru_reloads, falru_final);

    std::cout << "\n[STEP-BY-STEP COMPARISON]\n";
    std::cout << "| Step | Access | Epoch | LRU Outcome | LRU Victim | FA-LRU Outcome | FA-LRU Victim | Divergence? |\n";
    std::cout << "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n";

    bool found_divergence = false;
    for (size_t i = 0; i < access_seq.size(); ++i) {
        const auto& r1 = lru_records[i];
        const auto& r2 = falru_records[i];
        std::string v1_str = "-";
        std::string v2_str = "-";
        if (r1.selected_victim.first != 0 || r1.selected_victim.second != 0) {
            v1_str = "(" + std::to_string(r1.selected_victim.first) + "," + std::to_string(r1.selected_victim.second) + ")";
        }
        if (r2.selected_victim.first != 0 || r2.selected_victim.second != 0) {
            v2_str = "(" + std::to_string(r2.selected_victim.first) + "," + std::to_string(r2.selected_victim.second) + ")";
        }

        std::string div_str = "No";
        if (v1_str != v2_str && v1_str != "-") {
            div_str = "**YES** (DIVERGED)";
            found_divergence = true;
        } else if (r1.outcome != r2.outcome) {
            div_str = "**OUTCOME DIVERGED**";
        }

        std::cout << "| " << i << " | (" << access_seq[i].first << "," << access_seq[i].second << ") | "
                  << epoch_seq[i] << " | "
                  << r1.outcome << " | " << v1_str << " | "
                  << r2.outcome << " | " << v2_str << " | "
                  << div_str << " |\n";
    }

    std::cout << "\n======================================================================\n";
    std::cout << "SUMMARY METRICS:\n";
    std::cout << "  LRU:    Hits=" << lru_hits << ", Misses=" << lru_misses << ", Evictions=" << lru_evicts << ", Reloads=" << lru_reloads << "\n";
    std::cout << "  FA-LRU: Hits=" << falru_hits << ", Misses=" << falru_misses << ", Evictions=" << falru_evicts << ", Reloads=" << falru_reloads << "\n";
    std::cout << "======================================================================\n";

    // Dump records to CSV
    std::ofstream csv("/home/mohman/ca_project/mohman/point_4/phase_2_5/policy_activation_decisions.csv");
    if (csv.is_open()) {
        csv << "step,layer,expert,epoch,lru_outcome,lru_victim_layer,lru_victim_expert,falru_outcome,falru_victim_layer,falru_victim_expert,diverged\n";
        for (size_t i = 0; i < access_seq.size(); ++i) {
            const auto& r1 = lru_records[i];
            const auto& r2 = falru_records[i];
            bool div = (r1.selected_victim != r2.selected_victim && (r1.selected_victim.first != 0 || r1.selected_victim.second != 0));
            int div_int = 0;
            if (div) {
                div_int = 1;
            }
            csv << i << "," << access_seq[i].first << "," << access_seq[i].second << "," << epoch_seq[i] << ","
                << r1.outcome << "," << r1.selected_victim.first << "," << r1.selected_victim.second << ","
                << r2.outcome << "," << r2.selected_victim.first << "," << r2.selected_victim.second << ","
                << div_int << "\n";
        }
        csv.close();
    }

    if (found_divergence) {
        std::cout << "[SUCCESS] Genuine frequency-driven victim divergence verified!\n";
        return 0;
    } else {
        std::cerr << "[FAILURE] No victim divergence observed!\n";
        return 1;
    }
}
