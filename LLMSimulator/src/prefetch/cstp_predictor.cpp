#include "prefetch/cstp_predictor.h"

#include <numeric>

namespace llm_system {

CSTPPredictor::CSTPPredictor(
    int num_moe_stages,
    int num_routed_experts,
    int top_k,
    double alpha,
    double smoothing_lambda,
    double prior_epsilon,
    bool online_adaptation,
    int prediction_horizon)
    : num_moe_stages_(num_moe_stages),
      num_routed_experts_(num_routed_experts),
      top_k_(top_k),
      alpha_(alpha),
      smoothing_lambda_(smoothing_lambda),
      prior_epsilon_(prior_epsilon),
      online_adaptation_(online_adaptation),
      prediction_horizon_(prediction_horizon) {
    stage_marginal_counts_.resize(
        num_moe_stages_, std::vector<double>(num_routed_experts_, 0.0));

    spatial_counts_.resize(
        num_moe_stages_,
        std::vector<std::vector<double>>(
            num_routed_experts_, std::vector<double>(num_routed_experts_, 0.0)));

    temporal_counts_.resize(
        num_moe_stages_,
        std::vector<std::vector<double>>(
            num_routed_experts_, std::vector<double>(num_routed_experts_, 0.0)));
}

void CSTPPredictor::reset(int request_id) {
    auto it = history_table_.find(request_id);
    if (it != history_table_.end()) {
        history_table_.erase(it);
    }
}

void CSTPPredictor::set_marginal_count(int stage_id, int expert_id, double count) {
    if (stage_id >= 0 && stage_id < num_moe_stages_) {
        if (expert_id >= 0 && expert_id < num_routed_experts_) {
            stage_marginal_counts_[stage_id][expert_id] = count;
        }
    }
}

void CSTPPredictor::set_spatial_count(
    int source_stage_id, int source_expert, int target_expert, double count) {
    if (source_stage_id >= 0 && source_stage_id < num_moe_stages_) {
        if (source_expert >= 0 && source_expert < num_routed_experts_) {
            if (target_expert >= 0 && target_expert < num_routed_experts_) {
                spatial_counts_[source_stage_id][source_expert][target_expert] = count;
            }
        }
    }
}

void CSTPPredictor::set_temporal_count(
    int stage_id, int prev_expert, int curr_expert, double count) {
    if (stage_id >= 0 && stage_id < num_moe_stages_) {
        if (prev_expert >= 0 && prev_expert < num_routed_experts_) {
            if (curr_expert >= 0 && curr_expert < num_routed_experts_) {
                temporal_counts_[stage_id][prev_expert][curr_expert] = count;
            }
        }
    }
}

void CSTPPredictor::set_history_entry(
    int request_id, int stage_id, const std::vector<int>& experts) {
    history_table_[request_id][stage_id] = experts;
}

std::vector<double> CSTPPredictor::get_marginal_prior(int stage_id) const {
    std::vector<double> prior(num_routed_experts_, 0.0);

    if (stage_id < 0 || stage_id >= num_moe_stages_) {
        double uniform = 1.0 / static_cast<double>(num_routed_experts_);
        for (int e = 0; e < num_routed_experts_; ++e) {
            prior[e] = uniform;
        }
        return prior;
    }

    const auto& counts = stage_marginal_counts_[stage_id];
    double total_counts = 0.0;
    for (int e = 0; e < num_routed_experts_; ++e) {
        total_counts = total_counts + counts[e];
    }

    double denom = total_counts + (static_cast<double>(num_routed_experts_) * prior_epsilon_);
    if (denom > 0.0) {
        for (int e = 0; e < num_routed_experts_; ++e) {
            prior[e] = (counts[e] + prior_epsilon_) / denom;
        }
    } else {
        double uniform = 1.0 / static_cast<double>(num_routed_experts_);
        for (int e = 0; e < num_routed_experts_; ++e) {
            prior[e] = uniform;
        }
    }

    return prior;
}

std::vector<double> CSTPPredictor::compute_spatial_score(
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts) const {
    std::vector<double> prior = get_marginal_prior(target_stage_id);

    if (source_stage_id < 0 || source_stage_id >= num_moe_stages_) {
        return prior;
    }

    std::vector<int> valid_sources;
    for (int u : source_experts) {
        if (u >= 0 && u < num_routed_experts_) {
            valid_sources.push_back(u);
        }
    }

    if (valid_sources.empty()) {
        return prior;
    }

    std::vector<double> spatial_accum(num_routed_experts_, 0.0);

    for (int u : valid_sources) {
        const auto& co_counts = spatial_counts_[source_stage_id][u];
        double total_co = 0.0;
        for (int v = 0; v < num_routed_experts_; ++v) {
            total_co = total_co + co_counts[v];
        }

        double denom = total_co + smoothing_lambda_;
        if (denom > 0.0) {
            for (int v = 0; v < num_routed_experts_; ++v) {
                double p_cond = (co_counts[v] + (smoothing_lambda_ * prior[v])) / denom;
                spatial_accum[v] = spatial_accum[v] + p_cond;
            }
        } else {
            for (int v = 0; v < num_routed_experts_; ++v) {
                spatial_accum[v] = spatial_accum[v] + prior[v];
            }
        }
    }

    double num_sources = static_cast<double>(valid_sources.size());
    for (int v = 0; v < num_routed_experts_; ++v) {
        spatial_accum[v] = spatial_accum[v] / num_sources;
    }

    return spatial_accum;
}

std::vector<double> CSTPPredictor::compute_temporal_score(
    int request_id,
    int target_stage_id) const {
    std::vector<double> prior = get_marginal_prior(target_stage_id);

    if (target_stage_id < 0 || target_stage_id >= num_moe_stages_) {
        return prior;
    }

    auto req_it = history_table_.find(request_id);
    if (req_it == history_table_.end()) {
        return prior;
    }

    auto stage_it = req_it->second.find(target_stage_id);
    if (stage_it == req_it->second.end()) {
        return prior;
    }

    std::vector<int> valid_prev;
    for (int w : stage_it->second) {
        if (w >= 0 && w < num_routed_experts_) {
            valid_prev.push_back(w);
        }
    }

    if (valid_prev.empty()) {
        return prior;
    }

    std::vector<double> temporal_accum(num_routed_experts_, 0.0);

    for (int w : valid_prev) {
        const auto& trans_counts = temporal_counts_[target_stage_id][w];
        double total_trans = 0.0;
        for (int v = 0; v < num_routed_experts_; ++v) {
            total_trans = total_trans + trans_counts[v];
        }

        double denom = total_trans + smoothing_lambda_;
        if (denom > 0.0) {
            for (int v = 0; v < num_routed_experts_; ++v) {
                double p_trans = (trans_counts[v] + (smoothing_lambda_ * prior[v])) / denom;
                temporal_accum[v] = temporal_accum[v] + p_trans;
            }
        } else {
            for (int v = 0; v < num_routed_experts_; ++v) {
                temporal_accum[v] = temporal_accum[v] + prior[v];
            }
        }
    }

    double num_prev = static_cast<double>(valid_prev.size());
    for (int v = 0; v < num_routed_experts_; ++v) {
        temporal_accum[v] = temporal_accum[v] / num_prev;
    }

    return temporal_accum;
}

PredictionResult CSTPPredictor::predict(
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

    std::vector<double> s_spatial = compute_spatial_score(
        source_stage_id, target_stage_id, source_experts);
    std::vector<double> s_temporal = compute_temporal_score(
        request_id, target_stage_id);

    result.scores.reserve(num_routed_experts_);
    for (int v = 0; v < num_routed_experts_; ++v) {
        double composite = (alpha_ * s_spatial[v]) + ((1.0 - alpha_) * s_temporal[v]);
        CandidateScore cs;
        cs.expert_id = v;
        cs.score = composite;
        result.scores.push_back(cs);
    }

    result.finalize_and_sort(budget);
    return result;
}

void CSTPPredictor::update(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    const std::vector<int>& actual_target_experts) {
    (void)token_id;

    if (online_adaptation_) {
        // 1. Update target stage marginal frequency counts
        if (target_stage_id >= 0 && target_stage_id < num_moe_stages_) {
            for (int v : actual_target_experts) {
                if (v >= 0 && v < num_routed_experts_) {
                    stage_marginal_counts_[target_stage_id][v] =
                        stage_marginal_counts_[target_stage_id][v] + 1.0;
                }
            }
        }

        // 2. Update spatial co-activation transitions
        if (source_stage_id >= 0 && source_stage_id < num_moe_stages_) {
            for (int u : source_experts) {
                if (u >= 0 && u < num_routed_experts_) {
                    for (int v : actual_target_experts) {
                        if (v >= 0 && v < num_routed_experts_) {
                            spatial_counts_[source_stage_id][u][v] =
                                spatial_counts_[source_stage_id][u][v] + 1.0;
                        }
                    }
                }
            }
        }

        // 3. Update temporal transitions from previous token at target_stage_id
        if (target_stage_id >= 0 && target_stage_id < num_moe_stages_) {
            auto req_it = history_table_.find(request_id);
            if (req_it != history_table_.end()) {
                auto stage_it = req_it->second.find(target_stage_id);
                if (stage_it != req_it->second.end()) {
                    for (int w : stage_it->second) {
                        if (w >= 0 && w < num_routed_experts_) {
                            for (int v : actual_target_experts) {
                                if (v >= 0 && v < num_routed_experts_) {
                                    temporal_counts_[target_stage_id][w][v] =
                                        temporal_counts_[target_stage_id][w][v] + 1.0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Always update History Table for target_stage_id
    history_table_[request_id][target_stage_id] = actual_target_experts;
}

} // namespace llm_system
