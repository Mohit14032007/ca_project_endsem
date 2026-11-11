#include "prefetch/acsh_predictor.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace llm_system {

ACSHPredictor::ACSHPredictor(
    PredictorType variant,
    std::shared_ptr<CSTPPredictor> cstp,
    std::shared_ptr<STMoEPredictor> st_moe,
    double beta,
    double tau,
    double temp_c,
    double temp_m,
    double initial_reliability,
    int shift_window,
    double shift_threshold,
    double fast_beta,
    int fast_cooldown_steps)
    : variant_(variant),
      cstp_(cstp),
      st_moe_(st_moe),
      beta_(beta),
      tau_(tau),
      temp_c_(temp_c),
      temp_m_(temp_m),
      initial_reliability_(initial_reliability),
      shift_window_(shift_window),
      shift_threshold_(shift_threshold),
      fast_beta_(fast_beta),
      fast_cooldown_steps_(fast_cooldown_steps) {
    if (!cstp_ || !st_moe_) {
        throw std::invalid_argument(
            "[ACSH_ERROR] Both CSTP and ST-MoE component predictors must be provided.");
    }

    num_routed_experts_ = cstp_->get_num_routed_experts();
    num_moe_stages_ = cstp_->get_num_moe_stages();

    if (tau_ <= 0.0) {
        tau_ = 1e-4;
    }
    if (temp_c_ <= 0.0) {
        temp_c_ = 1e-4;
    }
    if (temp_m_ <= 0.0) {
        temp_m_ = 1e-4;
    }
    if (shift_window_ <= 0) {
        shift_window_ = 1;
    }
}

std::string ACSHPredictor::get_name() const {
    if (variant_ == PredictorType::ACSH_GLOBAL) {
        return "acsh_global";
    }
    if (variant_ == PredictorType::ACSH_STAGE_AWARE) {
        return "acsh_stage_aware";
    }
    if (variant_ == PredictorType::ACSH_FAST_RECOVERY) {
        return "acsh_fast_recovery";
    }
    return "acsh";
}

void ACSHPredictor::reset(int request_id) {
    // 1. Reset component predictors
    if (cstp_) {
        cstp_->reset(request_id);
    }
    if (st_moe_) {
        st_moe_->reset(request_id);
    }

    // 2. Clear reliability state for this request
    rel_c_global_.erase(request_id);
    rel_m_global_.erase(request_id);

    rel_c_stage_.erase(request_id);
    rel_m_stage_.erase(request_id);

    recent_errors_.erase(request_id);
    fast_cooldown_.erase(request_id);

    // 3. Clear pending predictions for this request
    // We clear any record whose upper 28 bits match request_id
    std::vector<long long> keys_to_remove;
    for (const auto& pair : pending_predictions_) {
        int req = static_cast<int>(pair.first >> 36);
        if (req == request_id) {
            keys_to_remove.push_back(pair.first);
        }
    }
    for (long long k : keys_to_remove) {
        pending_predictions_.erase(k);
    }
}

std::pair<double, double> ACSHPredictor::get_current_reliability(
    int request_id, int target_stage_id, int horizon) const {
    if (variant_ == PredictorType::ACSH_GLOBAL) {
        auto it_c = rel_c_global_.find(request_id);
        if (it_c == rel_c_global_.end()) {
            rel_c_global_[request_id] = initial_reliability_;
            rel_m_global_[request_id] = initial_reliability_;
            return {initial_reliability_, initial_reliability_};
        }
        return {it_c->second, rel_m_global_[request_id]};
    }

    // Stage-aware or Fast-recovery
    long long key = make_stage_key(target_stage_id, horizon);
    auto& stage_map_c = rel_c_stage_[request_id];
    auto it_c = stage_map_c.find(key);
    if (it_c == stage_map_c.end()) {
        stage_map_c[key] = initial_reliability_;
        rel_m_stage_[request_id][key] = initial_reliability_;
        return {initial_reliability_, initial_reliability_};
    }
    return {it_c->second, rel_m_stage_[request_id][key]};
}

std::pair<double, double> ACSHPredictor::get_fusion_weights(
    int request_id, int target_stage_id, int horizon) const {
    std::pair<double, double> rels = get_current_reliability(
        request_id, target_stage_id, horizon);
    double r_c = rels.first;
    double r_m = rels.second;

    double max_r = r_c;
    if (r_m > max_r) {
        max_r = r_m;
    }

    double e_c = std::exp((r_c - max_r) / tau_);
    double e_m = std::exp((r_m - max_r) / tau_);
    double denom = e_c + e_m;

    double w_c = 0.5;
    double w_m = 0.5;
    if (denom > 0.0) {
        w_c = e_c / denom;
        w_m = e_m / denom;
    }

    return {w_c, w_m};
}

double ACSHPredictor::calculate_recall_utility(
    const std::vector<int>& predicted,
    const std::vector<int>& actual) const {
    if (actual.empty()) {
        return 0.0;
    }

    std::unordered_set<int> act_set(actual.begin(), actual.end());
    int hits = 0;
    for (int p : predicted) {
        if (act_set.find(p) != act_set.end()) {
            hits = hits + 1;
        }
    }

    return static_cast<double>(hits) / static_cast<double>(actual.size());
}

PredictionResult ACSHPredictor::predict(
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
    int horizon = target_stage_id - source_stage_id;
    result.horizon = horizon;

    if (budget <= 0) {
        result.finalize_and_sort(0);
        return result;
    }

    // 1. Obtain CSTP component scores
    std::vector<double> s_spatial = cstp_->compute_spatial_score(
        source_stage_id, target_stage_id, source_experts);
    std::vector<double> s_temporal = cstp_->compute_temporal_score(
        request_id, target_stage_id);

    double cstp_alpha = cstp_->get_alpha();
    std::vector<double> raw_cstp(num_routed_experts_, 0.0);
    double max_cstp = -1e9;
    for (int v = 0; v < num_routed_experts_; ++v) {
        raw_cstp[v] = (cstp_alpha * s_spatial[v]) + ((1.0 - cstp_alpha) * s_temporal[v]);
        if (raw_cstp[v] > max_cstp) {
            max_cstp = raw_cstp[v];
        }
    }

    // 2. Obtain ST-MoE component scores
    std::vector<double> raw_st = st_moe_->compute_confidence_scores(
        request_id, source_stage_id, target_stage_id, source_experts);
    double max_st = -1e9;
    for (int v = 0; v < num_routed_experts_; ++v) {
        if (raw_st[v] > max_st) {
            max_st = raw_st[v];
        }
    }

    // 3. Compute component Top-B candidate sets for causal update evaluation
    std::vector<CandidateScore> cstp_candidates(num_routed_experts_);
    std::vector<CandidateScore> st_candidates(num_routed_experts_);
    for (int v = 0; v < num_routed_experts_; ++v) {
        cstp_candidates[v] = CandidateScore{v, raw_cstp[v]};
        st_candidates[v] = CandidateScore{v, raw_st[v]};
    }

    PredictionResult cstp_temp;
    cstp_temp.scores = cstp_candidates;
    cstp_temp.finalize_and_sort(budget);

    PredictionResult st_temp;
    st_temp.scores = st_candidates;
    st_temp.finalize_and_sort(budget);

    // 4. Softmax score normalization
    std::vector<double> p_c(num_routed_experts_, 0.0);
    double sum_exp_c = 0.0;
    for (int v = 0; v < num_routed_experts_; ++v) {
        p_c[v] = std::exp((raw_cstp[v] - max_cstp) / temp_c_);
        sum_exp_c = sum_exp_c + p_c[v];
    }
    if (sum_exp_c > 0.0) {
        for (int v = 0; v < num_routed_experts_; ++v) {
            p_c[v] = p_c[v] / sum_exp_c;
        }
    } else {
        double uniform = 1.0 / static_cast<double>(num_routed_experts_);
        for (int v = 0; v < num_routed_experts_; ++v) {
            p_c[v] = uniform;
        }
    }

    std::vector<double> p_m(num_routed_experts_, 0.0);
    double sum_exp_m = 0.0;
    for (int v = 0; v < num_routed_experts_; ++v) {
        p_m[v] = std::exp((raw_st[v] - max_st) / temp_m_);
        sum_exp_m = sum_exp_m + p_m[v];
    }
    if (sum_exp_m > 0.0) {
        for (int v = 0; v < num_routed_experts_; ++v) {
            p_m[v] = p_m[v] / sum_exp_m;
        }
    } else {
        double uniform = 1.0 / static_cast<double>(num_routed_experts_);
        for (int v = 0; v < num_routed_experts_; ++v) {
            p_m[v] = uniform;
        }
    }

    // 5. Softmax fusion weights
    std::pair<double, double> weights = get_fusion_weights(
        request_id, target_stage_id, horizon);
    double w_c = weights.first;
    double w_m = weights.second;

    // 6. Blended ACSH score
    result.scores.reserve(num_routed_experts_);
    for (int v = 0; v < num_routed_experts_; ++v) {
        double p_acsh = (w_c * p_c[v]) + (w_m * p_m[v]);
        CandidateScore cs;
        cs.expert_id = v;
        cs.score = p_acsh;
        result.scores.push_back(cs);
    }

    result.finalize_and_sort(budget);

    // 7. Record pending prediction outcome for update()
    long long pred_key = make_pred_key(request_id, target_stage_id, token_id);
    PendingRecord rec;
    rec.cstp_top_b = cstp_temp.candidate_experts;
    rec.st_top_b = st_temp.candidate_experts;
    rec.acsh_top_b = result.candidate_experts;
    rec.w_c = w_c;
    rec.w_m = w_m;
    rec.horizon = horizon;
    rec.budget = budget;
    pending_predictions_[pred_key] = rec;

    return result;
}

void ACSHPredictor::update(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    const std::vector<int>& actual_target_experts) {
    // 1. Update component predictors
    cstp_->update(
        request_id, token_id, source_stage_id, target_stage_id,
        source_experts, actual_target_experts);
    st_moe_->update(
        request_id, token_id, source_stage_id, target_stage_id,
        source_experts, actual_target_experts);

    // 2. Retrieve pending prediction record
    long long pred_key = make_pred_key(request_id, target_stage_id, token_id);
    auto it = pending_predictions_.find(pred_key);
    if (it == pending_predictions_.end()) {
        return;
    }

    PendingRecord rec = it->second;
    pending_predictions_.erase(it);

    int horizon = rec.horizon;

    // 3. Calculate instantaneous event utilities (Recall)
    double r_c_event = calculate_recall_utility(rec.cstp_top_b, actual_target_experts);
    double r_m_event = calculate_recall_utility(rec.st_top_b, actual_target_experts);
    double r_acsh_event = calculate_recall_utility(rec.acsh_top_b, actual_target_experts);

    // 4. Determine adaptive momentum beta
    double active_beta = beta_;
    long long stage_key = make_stage_key(target_stage_id, horizon);

    if (variant_ == PredictorType::ACSH_FAST_RECOVERY) {
        auto& err_deque = recent_errors_[request_id][stage_key];
        double err = 1.0 - r_acsh_event;
        err_deque.push_back(err);
        if (static_cast<int>(err_deque.size()) > shift_window_) {
            err_deque.pop_front();
        }

        int& cooldown = fast_cooldown_[request_id][stage_key];
        if (cooldown > 0) {
            active_beta = fast_beta_;
            cooldown = cooldown - 1;
        } else {
            if (static_cast<int>(err_deque.size()) >= shift_window_) {
                double mean_err = 0.0;
                for (double e : err_deque) {
                    mean_err = mean_err + e;
                }
                mean_err = mean_err / static_cast<double>(err_deque.size());

                if (mean_err >= shift_threshold_) {
                    active_beta = fast_beta_;
                    cooldown = fast_cooldown_steps_;
                    shift_detected_count_ = shift_detected_count_ + 1;
                }
            }
        }
    }

    // 5. Causal EMA update of reliability estimates
    if (variant_ == PredictorType::ACSH_GLOBAL) {
        double old_rc = initial_reliability_;
        double old_rm = initial_reliability_;
        auto it_rc = rel_c_global_.find(request_id);
        if (it_rc != rel_c_global_.end()) {
            old_rc = it_rc->second;
            old_rm = rel_m_global_[request_id];
        }

        double new_rc = (active_beta * old_rc) + ((1.0 - active_beta) * r_c_event);
        double new_rm = (active_beta * old_rm) + ((1.0 - active_beta) * r_m_event);

        rel_c_global_[request_id] = new_rc;
        rel_m_global_[request_id] = new_rm;
    } else {
        // Stage-aware or Fast-recovery
        double old_rc = initial_reliability_;
        double old_rm = initial_reliability_;
        auto& stage_map_c = rel_c_stage_[request_id];
        auto it_rc = stage_map_c.find(stage_key);
        if (it_rc != stage_map_c.end()) {
            old_rc = it_rc->second;
            old_rm = rel_m_stage_[request_id][stage_key];
        }

        double new_rc = (active_beta * old_rc) + ((1.0 - active_beta) * r_c_event);
        double new_rm = (active_beta * old_rm) + ((1.0 - active_beta) * r_m_event);

        stage_map_c[stage_key] = new_rc;
        rel_m_stage_[request_id][stage_key] = new_rm;
    }
}

} // namespace llm_system
