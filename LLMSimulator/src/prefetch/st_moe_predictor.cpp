#include "prefetch/st_moe_predictor.h"

#include <algorithm>

namespace llm_system {

STMoEPredictor::STMoEPredictor(
    int num_moe_stages,
    int num_routed_experts,
    int top_k,
    int cct_capacity,
    int confidence_threshold,
    bool online_adaptation,
    int prediction_horizon)
    : num_moe_stages_(num_moe_stages),
      num_routed_experts_(num_routed_experts),
      top_k_(top_k),
      cct_capacity_(cct_capacity),
      confidence_threshold_(confidence_threshold),
      online_adaptation_(online_adaptation),
      prediction_horizon_(prediction_horizon) {
    if (cct_capacity_ <= 0) {
        cct_capacity_ = top_k_;
    }

    cct_.resize(num_moe_stages_);
    for (int s = 0; s < num_moe_stages_; ++s) {
        cct_[s].resize(num_routed_experts_);
    }
}

void STMoEPredictor::reset(int request_id) {
    auto it = history_table_.find(request_id);
    if (it != history_table_.end()) {
        history_table_.erase(it);
    }
}

void STMoEPredictor::set_cct_entry(
    int source_stage_id,
    int source_expert,
    int dest_expert,
    int confidence) {
    if (source_stage_id >= 0 && source_stage_id < num_moe_stages_) {
        if (source_expert >= 0 && source_expert < num_routed_experts_) {
            if (dest_expert >= 0 && dest_expert < num_routed_experts_) {
                int clamped = confidence;
                if (clamped < 0) {
                    clamped = 0;
                }
                if (clamped > 3) {
                    clamped = 3;
                }
                cct_[source_stage_id][source_expert][dest_expert] = clamped;
            }
        }
    }
}

int STMoEPredictor::get_cct_entry(
    int source_stage_id,
    int source_expert,
    int dest_expert) const {
    if (source_stage_id >= 0 && source_stage_id < num_moe_stages_) {
        if (source_expert >= 0 && source_expert < num_routed_experts_) {
            const auto& dest_map = cct_[source_stage_id][source_expert];
            auto it = dest_map.find(dest_expert);
            if (it != dest_map.end()) {
                return it->second;
            }
        }
    }
    return 0;
}

void STMoEPredictor::set_history_entry(
    int request_id,
    int stage_id,
    const std::vector<int>& experts) {
    history_table_[request_id][stage_id] = experts;
}

std::vector<int> STMoEPredictor::get_history_entry(
    int request_id,
    int stage_id) const {
    auto req_it = history_table_.find(request_id);
    if (req_it != history_table_.end()) {
        auto stage_it = req_it->second.find(stage_id);
        if (stage_it != req_it->second.end()) {
            return stage_it->second;
        }
    }
    return {};
}

std::vector<double> STMoEPredictor::compute_confidence_scores(
    int request_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts) const {
    std::vector<double> scores(num_routed_experts_, 0.0);

    // 1. Spatial accumulation across all observed source experts (Algorithm 2, Lines 3-5)
    if (source_stage_id >= 0 && source_stage_id < num_moe_stages_) {
        if (!source_experts.empty()) {
            for (int u : source_experts) {
                if (u >= 0 && u < num_routed_experts_) {
                    const auto& dest_map = cct_[source_stage_id][u];
                    for (const auto& pair : dest_map) {
                        int v = pair.first;
                        int cnt = pair.second;
                        if (v >= 0 && v < num_routed_experts_) {
                            scores[v] = scores[v] + static_cast<double>(cnt);
                        }
                    }
                }
            }
        }
    }

    // 2. Temporal contribution from History Table (HT) (Equation 1: Score(e) = CCT(e) + HT(e), HT(e) = 2)
    auto req_it = history_table_.find(request_id);
    if (req_it != history_table_.end()) {
        auto stage_it = req_it->second.find(target_stage_id);
        if (stage_it != req_it->second.end()) {
            for (int h : stage_it->second) {
                if (h >= 0 && h < num_routed_experts_) {
                    scores[h] = scores[h] + 2.0;
                }
            }
        }
    }

    return scores;
}

std::vector<int> STMoEPredictor::predict_threshold(
    int request_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    int threshold) const {
    std::vector<double> scores = compute_confidence_scores(
        request_id, source_stage_id, target_stage_id, source_experts);

    std::vector<CandidateScore> candidates;
    for (int exp_id = 0; exp_id < num_routed_experts_; ++exp_id) {
        if (scores[exp_id] >= static_cast<double>(threshold)) {
            CandidateScore cs;
            cs.expert_id = exp_id;
            cs.score = scores[exp_id];
            candidates.push_back(cs);
        }
    }

    // Deterministic sort: descending score, then ascending expert ID
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const CandidateScore& a, const CandidateScore& b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            return a.expert_id < b.expert_id;
        });

    std::vector<int> result;
    result.reserve(candidates.size());
    for (const auto& c : candidates) {
        result.push_back(c.expert_id);
    }
    return result;
}

PredictionResult STMoEPredictor::predict(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    int budget) {
    PredictionResult result;
    result.request_id = request_id;
    result.token_id = token_id;
    result.source_stage = source_stage_id;
    result.current_stage = source_stage_id;
    result.target_stage = target_stage_id;
    result.horizon = target_stage_id - source_stage_id;

    if (budget <= 0) {
        result.finalize_and_sort(0);
        return result;
    }

    std::vector<double> scores = compute_confidence_scores(
        request_id, source_stage_id, target_stage_id, source_experts);

    result.scores.reserve(num_routed_experts_);
    for (int exp_id = 0; exp_id < num_routed_experts_; ++exp_id) {
        CandidateScore cs;
        cs.expert_id = exp_id;
        cs.score = scores[exp_id];
        result.scores.push_back(cs);
    }

    result.finalize_and_sort(budget);
    return result;
}

void STMoEPredictor::update(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    const std::vector<int>& actual_target_experts) {
    (void)token_id;

    std::unordered_set<int> actual_set(
        actual_target_experts.begin(), actual_target_experts.end());

    // 1. Update Cross-Layer Correlation Table (Algorithm 3)
    if (online_adaptation_) {
        if (source_stage_id >= 0 && source_stage_id < num_moe_stages_) {
            if (!source_experts.empty()) {
                for (int u : source_experts) {
                    if (u >= 0 && u < num_routed_experts_) {
                        auto& dest_map = cct_[source_stage_id][u];

                        std::vector<int> stored_keys;
                        stored_keys.reserve(dest_map.size());
                        for (const auto& pair : dest_map) {
                            stored_keys.push_back(pair.first);
                        }

                        for (int k : stored_keys) {
                            int conf = dest_map[k];
                            if (actual_set.find(k) != actual_set.end()) {
                                // Match: increment counter up to 3 (Algorithm 3, Lines 6-7)
                                if (conf < 3) {
                                    dest_map[k] = conf + 1;
                                } else {
                                    dest_map[k] = 3;
                                }
                            } else {
                                // Mismatch: decrement counter (Algorithm 3, Lines 8-10)
                                if (conf > 0) {
                                    dest_map[k] = conf - 1;
                                } else {
                                    // ConfScore == 0: eligible for replacement (Algorithm 3, Lines 11-14)
                                    std::vector<int> eligible;
                                    for (int cand : actual_target_experts) {
                                        if (dest_map.find(cand) == dest_map.end()) {
                                            eligible.push_back(cand);
                                        }
                                    }
                                    if (!eligible.empty()) {
                                        std::sort(eligible.begin(), eligible.end());
                                        int g = eligible[0];
                                        dest_map.erase(k);
                                        dest_map[g] = 2; // Initialized to decimal 2 (binary 10)
                                    }
                                }
                            }
                        }

                        // If CCT entry has capacity remaining, insert un-stored actual target experts
                        if (static_cast<int>(dest_map.size()) < cct_capacity_) {
                            for (int cand : actual_target_experts) {
                                if (static_cast<int>(dest_map.size()) >= cct_capacity_) {
                                    break;
                                }
                                if (dest_map.find(cand) == dest_map.end()) {
                                    dest_map[cand] = 2; // Initialized to decimal 2 (binary 10)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Update History Table (HT) for temporal tracking
    history_table_[request_id][target_stage_id] = actual_target_experts;
}

} // namespace llm_system
