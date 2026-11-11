#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "prefetch/acsh_predictor.h"
#include "prefetch/control_predictors.h"
#include "prefetch/cstp_predictor.h"
#include "prefetch/none_predictor.h"
#include "prefetch/prefetch_config.h"
#include "prefetch/predictor_interface.h"
#include "prefetch/predictor_registry.h"
#include "prefetch/st_moe_predictor.h"

namespace llm_system {

/**
 * Event-level metrics structure.
 */
struct EventMetrics {
    int hits = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double hit_rate = 0.0;
    double mrr = 0.0;
    double ndcg = 0.0;
    double latency_ns = 0.0;
};

/**
 * Summary metrics aggregator for stratified analysis.
 */
struct MetricAccumulator {
    int count = 0;
    int total_hits = 0;
    double sum_precision = 0.0;
    double sum_recall = 0.0;
    double sum_f1 = 0.0;
    double sum_hit_rate = 0.0;
    double sum_mrr = 0.0;
    double sum_ndcg = 0.0;
    double sum_latency_ns = 0.0;

    void add(const EventMetrics& m) {
        count = count + 1;
        total_hits = total_hits + m.hits;
        sum_precision = sum_precision + m.precision;
        sum_recall = sum_recall + m.recall;
        sum_f1 = sum_f1 + m.f1;
        sum_hit_rate = sum_hit_rate + m.hit_rate;
        sum_mrr = sum_mrr + m.mrr;
        sum_ndcg = sum_ndcg + m.ndcg;
        sum_latency_ns = sum_latency_ns + m.latency_ns;
    }

    double mean_precision() const {
        if (count > 0) {
            return sum_precision / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_recall() const {
        if (count > 0) {
            return sum_recall / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_f1() const {
        if (count > 0) {
            return sum_f1 / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_hit_rate() const {
        if (count > 0) {
            return sum_hit_rate / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_mrr() const {
        if (count > 0) {
            return sum_mrr / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_ndcg() const {
        if (count > 0) {
            return sum_ndcg / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_latency_ns() const {
        if (count > 0) {
            return sum_latency_ns / static_cast<double>(count);
        }
        return 0.0;
    }
};

/**
 * Raw record parsed from trace CSV.
 */
struct RawTraceRecord {
    int request_id = 0;
    int token_id = 0;
    int layer_id = 0;
    int rank = 0;
    int expert_id = 0;
};

/**
 * Computes Precision, Recall, F1, Hit Rate, MRR, nDCG@B.
 * Enforces zero ternary operators.
 */
static EventMetrics compute_metrics(
    const std::vector<int>& predicted,
    const std::vector<int>& actual,
    double latency_ns) {
    EventMetrics m;
    m.latency_ns = latency_ns;

    std::unordered_set<int> actual_set(actual.begin(), actual.end());
    m.hits = 0;
    int first_hit_rank = -1;

    for (size_t i = 0; i < predicted.size(); ++i) {
        int candidate = predicted[i];
        if (actual_set.find(candidate) != actual_set.end()) {
            m.hits = m.hits + 1;
            if (first_hit_rank < 0) {
                first_hit_rank = static_cast<int>(i + 1);
            }
        }
    }

    if (!predicted.empty()) {
        m.precision = static_cast<double>(m.hits) / static_cast<double>(predicted.size());
    } else {
        m.precision = 0.0;
    }

    if (!actual.empty()) {
        m.recall = static_cast<double>(m.hits) / static_cast<double>(actual.size());
    } else {
        m.recall = 0.0;
    }

    if ((m.precision + m.recall) > 0.0) {
        m.f1 = (2.0 * m.precision * m.recall) / (m.precision + m.recall);
    } else {
        m.f1 = 0.0;
    }

    if (m.hits > 0) {
        m.hit_rate = 1.0;
    } else {
        m.hit_rate = 0.0;
    }

    if (first_hit_rank > 0) {
        m.mrr = 1.0 / static_cast<double>(first_hit_rank);
    } else {
        m.mrr = 0.0;
    }

    // nDCG calculation
    double dcg = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        int candidate = predicted[i];
        if (actual_set.find(candidate) != actual_set.end()) {
            double rank_discount = std::log2(static_cast<double>(i + 2));
            dcg = dcg + (1.0 / rank_discount);
        }
    }

    size_t ideal_hits = predicted.size();
    if (actual.size() < ideal_hits) {
        ideal_hits = actual.size();
    }

    double idcg = 0.0;
    for (size_t i = 0; i < ideal_hits; ++i) {
        double rank_discount = std::log2(static_cast<double>(i + 2));
        idcg = idcg + (1.0 / rank_discount);
    }

    if (idcg > 0.0) {
        m.ndcg = dcg / idcg;
    } else {
        m.ndcg = 0.0;
    }

    return m;
}

/**
 * Extracts predictor-specific internal state as JSON string.
 */
static std::string get_internal_state_json(
    const IPredictor::Ptr& predictor,
    int request_id,
    int target_stage,
    int horizon) {
    std::ostringstream ss;
    ss << "{";
    PredictorType type = predictor->get_type();

    if (type == PredictorType::ST_MOE) {
        auto st = std::dynamic_pointer_cast<STMoEPredictor>(predictor);
        if (st) {
            std::vector<int> hist = st->get_history_entry(request_id, target_stage);
            ss << "\"history_count\":" << hist.size();
        }
    } else {
        if (type == PredictorType::CSTP) {
            auto cstp = std::dynamic_pointer_cast<CSTPPredictor>(predictor);
            if (cstp) {
                std::vector<double> prior = cstp->get_marginal_prior(target_stage);
                double max_prior = 0.0;
                for (double p : prior) {
                    if (p > max_prior) {
                        max_prior = p;
                    }
                }
                ss << "\"max_marginal_prior\":" << max_prior;
            }
        } else {
            if (type == PredictorType::ACSH_GLOBAL ||
                type == PredictorType::ACSH_STAGE_AWARE ||
                type == PredictorType::ACSH_FAST_RECOVERY) {
                auto acsh = std::dynamic_pointer_cast<ACSHPredictor>(predictor);
                if (acsh) {
                    auto rels = acsh->get_current_reliability(request_id, target_stage, horizon);
                    auto weights = acsh->get_fusion_weights(request_id, target_stage, horizon);
                    ss << "\"rel_c\":" << rels.first << ",\"rel_m\":" << rels.second;
                    ss << ",\"w_c\":" << weights.first << ",\"w_m\":" << weights.second;
                    ss << ",\"shift_count\":" << acsh->get_shift_detected_count();
                }
            } else {
                ss << "\"type\":\"control\"";
            }
        }
    }

    ss << "}";
    return ss.str();
}

/**
 * Strictly verifies causal separation contract.
 */
static bool run_causal_separation_test() {
    std::cout << "[CAUSAL_TEST] Verifying strict separation between prediction and target labels...\n";

    PrefetchConfig config;
    config.predictor_type = PredictorType::ST_MOE;
    config.online_adaptation = true;
    config.confidence_threshold = 2;
    config.prediction_horizon = 1;

    auto pred1 = PredictorRegistry::create_predictor(
        PredictorType::ST_MOE, config, 4, 16, 2);
    auto pred2 = PredictorRegistry::create_predictor(
        PredictorType::ST_MOE, config, 4, 16, 2);

    int req_id = 1;
    pred1->reset(req_id);
    pred2->reset(req_id);

    // Call predict on identical state for both
    PredictionResult r1 = pred1->predict(req_id, 0, -1, 1, {}, 2);
    PredictionResult r2 = pred2->predict(req_id, 0, -1, 1, {}, 2);

    if (r1.candidate_experts != r2.candidate_experts) {
        std::cerr << "  [FAIL] Initial predictions diverged before any target was revealed!\n";
        return false;
    }

    // Now reveal two DIFFERENT ground truth target sets to pred1 and pred2
    std::vector<int> actual_A = {3, 7};
    std::vector<int> actual_B = {8, 12};

    pred1->update(req_id, 0, -1, 1, {}, actual_A);
    pred2->update(req_id, 0, -1, 1, {}, actual_B);

    // Now predict for next token (token 1, stage 1)
    PredictionResult next_r1 = pred1->predict(req_id, 1, -1, 1, {}, 2);
    PredictionResult next_r2 = pred2->predict(req_id, 1, -1, 1, {}, 2);

    if (next_r1.candidate_experts == next_r2.candidate_experts) {
        std::cerr << "  [FAIL] State update failed to record revealed targets in History Table!\n";
        return false;
    }

    std::cout << "  [PASS] Causal separation strictly verified: predict() is independent of future targets,\n"
              << "         and update() causally propagates revealed targets to subsequent predictions.\n";
    return true;
}

} // namespace llm_system

int main(int argc, char* argv[]) {
    using namespace llm_system;

    std::string trace_path = "";
    std::string output_path = "";
    std::string predictor_name = "cstp";
    int budget = 4;
    int num_stages = -1;
    int num_experts = 256;
    int top_k = 8;
    int horizon = 1;
    int max_tokens = -1;
    double alpha = 0.5;
    double smoothing_lambda = 1.0;
    unsigned int random_seed = 42;
    bool run_causal = false;
    bool summary_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--trace" && i + 1 < argc) {
            trace_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--predictor" && i + 1 < argc) {
            predictor_name = argv[++i];
        } else if (arg == "--budget" && i + 1 < argc) {
            budget = std::stoi(argv[++i]);
        } else if (arg == "--num_stages" && i + 1 < argc) {
            num_stages = std::stoi(argv[++i]);
        } else if (arg == "--num_experts" && i + 1 < argc) {
            num_experts = std::stoi(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--horizon" && i + 1 < argc) {
            horizon = std::stoi(argv[++i]);
        } else if (arg == "--max_tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            alpha = std::stod(argv[++i]);
        } else if (arg == "--lambda" && i + 1 < argc) {
            smoothing_lambda = std::stod(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            random_seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        } else if (arg == "--causal_test") {
            run_causal = true;
        } else if (arg == "--summary_only") {
            summary_only = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --trace <path>         Input trace CSV\n"
                      << "  --output <path>        Output JSONL path\n"
                      << "  --predictor <name>     Predictor identifier (cstp, st_moe, acsh_global, etc.)\n"
                      << "  --budget <int>         Prefetch candidate budget B (default 4)\n"
                      << "  --num_stages <int>     Number of MoE stages (-1 to auto-detect)\n"
                      << "  --num_experts <int>    Total pool of routed experts (default 256)\n"
                      << "  --top_k <int>          Number of selected experts per token (default 8)\n"
                      << "  --horizon <int>        Prediction horizon (default 1)\n"
                      << "  --max_tokens <int>     Maximum tokens to process (-1 for all)\n"
                      << "  --alpha <float>        CSTP alpha parameter (default 0.5)\n"
                      << "  --lambda <float>       CSTP Dirichlet smoothing lambda (default 1.0)\n"
                      << "  --seed <int>           Random seed (default 42)\n"
                      << "  --causal_test          Run causal separation contract self-test\n"
                      << "  --summary_only         Suppress event JSONL output\n";
            return 0;
        }
    }

    if (run_causal) {
        bool ok = run_causal_separation_test();
        if (ok) {
            return 0;
        }
        return 1;
    }

    if (trace_path.empty()) {
        std::cerr << "[ERROR] Missing required --trace argument.\n";
        return 1;
    }

    std::ifstream infile(trace_path);
    if (!infile.is_open()) {
        std::cerr << "[ERROR] Could not open trace file: " << trace_path << "\n";
        return 1;
    }

    std::string header_line;
    if (!std::getline(infile, header_line)) {
        std::cerr << "[ERROR] Trace file is empty.\n";
        return 1;
    }

    // Parse header schema (support seq_id or request_id)
    std::vector<std::string> col_names;
    {
        std::stringstream ss(header_line);
        std::string col;
        while (std::getline(ss, col, ',')) {
            // Trim whitespace and carriage returns
            col.erase(col.find_last_not_of(" \r\n\t") + 1);
            col.erase(0, col.find_first_not_of(" \r\n\t"));
            col_names.push_back(col);
        }
    }

    int idx_req = -1;
    int idx_tok = -1;
    int idx_layer = -1;
    int idx_rank = -1;
    int idx_exp = -1;

    for (size_t i = 0; i < col_names.size(); ++i) {
        std::string c = col_names[i];
        if (c == "seq_id" || c == "request_id") {
            idx_req = static_cast<int>(i);
        } else if (c == "token_id") {
            idx_tok = static_cast<int>(i);
        } else if (c == "layer_id" || c == "stage_id" || c == "moe_stage_id") {
            idx_layer = static_cast<int>(i);
        } else if (c == "topk_rank" || c == "rank") {
            idx_rank = static_cast<int>(i);
        } else if (c == "expert_id") {
            idx_exp = static_cast<int>(i);
        }
    }

    if (idx_req < 0 || idx_tok < 0 || idx_layer < 0 || idx_exp < 0) {
        std::cerr << "[ERROR] Unrecognized CSV schema in header: " << header_line << "\n";
        return 1;
    }

    // Read and organize trace records
    std::vector<RawTraceRecord> records;
    std::string line;
    int detected_max_layer = -1;
    int detected_max_expert = -1;

    while (std::getline(infile, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string token_str;
        std::vector<std::string> fields;
        while (std::getline(ss, token_str, ',')) {
            fields.push_back(token_str);
        }

        if (fields.size() <= static_cast<size_t>(std::max({idx_req, idx_tok, idx_layer, idx_exp}))) {
            continue;
        }

        RawTraceRecord rec;
        rec.request_id = std::stoi(fields[idx_req]);
        rec.token_id = std::stoi(fields[idx_tok]);
        rec.layer_id = std::stoi(fields[idx_layer]);
        if (idx_rank >= 0 && static_cast<size_t>(idx_rank) < fields.size()) {
            rec.rank = std::stoi(fields[idx_rank]);
        } else {
            rec.rank = 0;
        }
        rec.expert_id = std::stoi(fields[idx_exp]);

        if (rec.layer_id > detected_max_layer) {
            detected_max_layer = rec.layer_id;
        }
        if (rec.expert_id > detected_max_expert) {
            detected_max_expert = rec.expert_id;
        }

        records.push_back(rec);
    }
    infile.close();

    if (records.empty()) {
        std::cerr << "[ERROR] No valid routing records found in trace.\n";
        return 1;
    }

    if (num_stages <= 0) {
        num_stages = detected_max_layer + 1;
    }
    if (detected_max_expert >= num_experts) {
        num_experts = detected_max_expert + 1;
    }

    std::cout << "[REPLAY_PREDICTORS] Loaded " << records.size() << " records from " << trace_path << "\n";
    std::cout << "[REPLAY_PREDICTORS] Stages: " << num_stages << ", Experts: " << num_experts 
              << ", Budget: " << budget << ", Predictor: " << predictor_name << "\n";

    // Group records by (request_id, token_id, layer_id)
    // Map key: tuple of (request_id, token_id, layer_id) -> list of expert_id sorted by rank
    struct GroupKey {
        int req;
        int tok;
        int layer;
        bool operator<(const GroupKey& o) const {
            if (req != o.req) return req < o.req;
            if (tok != o.tok) return tok < o.tok;
            return layer < o.layer;
        }
    };

    struct RankedExpert {
        int rank;
        int expert_id;
    };

    std::map<GroupKey, std::vector<RankedExpert>> grouped;
    for (const auto& r : records) {
        GroupKey k{r.request_id, r.token_id, r.layer_id};
        grouped[k].push_back(RankedExpert{r.rank, r.expert_id});
    }

    // Configure and instantiate predictor
    PrefetchConfig config;
    try {
        config.predictor_type = parse_predictor_type(predictor_name);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Unknown predictor identifier: " << predictor_name << "\n";
        return 1;
    }

    config.budget = budget;
    config.alpha = alpha;
    config.smoothing_lambda = smoothing_lambda;
    config.random_seed = random_seed;
    config.confidence_threshold = 2;
    config.online_adaptation = true;
    config.prediction_horizon = horizon;

    IPredictor::Ptr predictor = PredictorRegistry::create_predictor(
        config.predictor_type, config, num_stages, num_experts, top_k);

    std::ofstream outfile;
    if (!output_path.empty() && !summary_only) {
        outfile.open(output_path);
        if (!outfile.is_open()) {
            std::cerr << "[ERROR] Could not open output file: " << output_path << "\n";
            return 1;
        }
    }

    // Metric accumulators
    MetricAccumulator acc_overall;
    MetricAccumulator acc_cold;
    MetricAccumulator acc_warm;
    MetricAccumulator acc_steady;

    // Collect unique requests
    std::vector<int> unique_requests;
    for (const auto& pair : grouped) {
        int r = pair.first.req;
        if (unique_requests.empty() || unique_requests.back() != r) {
            unique_requests.push_back(r);
        }
    }

    // Chronological replay loop
    for (int req_id : unique_requests) {
        predictor->reset(req_id);

        // Collect tokens for this request
        std::vector<int> req_tokens;
        for (const auto& pair : grouped) {
            if (pair.first.req == req_id) {
                int tok = pair.first.tok;
                if (req_tokens.empty() || req_tokens.back() != tok) {
                    req_tokens.push_back(tok);
                }
            }
        }

        if (max_tokens > 0 && static_cast<int>(req_tokens.size()) > max_tokens) {
            req_tokens.resize(max_tokens);
        }

        for (int tok_id : req_tokens) {
            // Buffer to hold actual experts of each stage for this token
            std::map<int, std::vector<int>> token_stage_actuals;

            for (int stage_id = 0; stage_id < num_stages; ++stage_id) {
                GroupKey k{req_id, tok_id, stage_id};
                auto it = grouped.find(k);
                if (it == grouped.end()) {
                    continue;
                }

                auto ranked_list = it->second;
                std::sort(ranked_list.begin(), ranked_list.end(), [](const RankedExpert& a, const RankedExpert& b) {
                    return a.rank < b.rank;
                });

                std::vector<int> actual_experts;
                actual_experts.reserve(ranked_list.size());
                for (const auto& item : ranked_list) {
                    actual_experts.push_back(item.expert_id);
                }

                token_stage_actuals[stage_id] = actual_experts;

                int source_stage = -1;
                std::vector<int> source_experts;
                if (stage_id >= horizon) {
                    source_stage = stage_id - horizon;
                    auto src_it = token_stage_actuals.find(source_stage);
                    if (src_it != token_stage_actuals.end()) {
                        source_experts = src_it->second;
                    }
                }

                // 1. Causal Prediction: call predict BEFORE revealing actual target experts
                auto t_start = std::chrono::high_resolution_clock::now();
                PredictionResult pred_res = predictor->predict(
                    req_id, tok_id, source_stage, stage_id, source_experts, budget);
                auto t_end = std::chrono::high_resolution_clock::now();

                double latency_ns = std::chrono::duration<double, std::nano>(t_end - t_start).count();

                // 2. Compute evaluation metrics
                EventMetrics metrics = compute_metrics(
                    pred_res.candidate_experts, actual_experts, latency_ns);

                // 3. Accumulate into stratification buckets
                acc_overall.add(metrics);
                if (tok_id == 0) {
                    acc_cold.add(metrics);
                } else {
                    if (tok_id < 4) {
                        acc_warm.add(metrics);
                    } else {
                        acc_steady.add(metrics);
                    }
                }

                // 4. Capture internal state snapshot
                std::string state_json = get_internal_state_json(predictor, req_id, stage_id, horizon);

                // 5. Output event record to JSONL
                if (outfile.is_open()) {
                    std::string strat = "steady";
                    if (tok_id == 0) {
                        strat = "cold";
                    } else {
                        if (tok_id < 4) {
                            strat = "warm";
                        }
                    }

                    outfile << "{"
                            << "\"request_id\":" << req_id << ","
                            << "\"token_id\":" << tok_id << ","
                            << "\"source_stage\":" << source_stage << ","
                            << "\"target_stage\":" << stage_id << ","
                            << "\"horizon\":" << horizon << ","
                            << "\"budget\":" << budget << ","
                            << "\"stratification\":\"" << strat << "\",";

                    outfile << "\"source_experts\":[";
                    for (size_t e_idx = 0; e_idx < source_experts.size(); ++e_idx) {
                        outfile << source_experts[e_idx];
                        if (e_idx + 1 < source_experts.size()) outfile << ",";
                    }
                    outfile << "],";

                    outfile << "\"actual_target_experts\":[";
                    for (size_t e_idx = 0; e_idx < actual_experts.size(); ++e_idx) {
                        outfile << actual_experts[e_idx];
                        if (e_idx + 1 < actual_experts.size()) outfile << ",";
                    }
                    outfile << "],";

                    outfile << "\"predicted_candidates\":[";
                    for (size_t e_idx = 0; e_idx < pred_res.candidate_experts.size(); ++e_idx) {
                        outfile << pred_res.candidate_experts[e_idx];
                        if (e_idx + 1 < pred_res.candidate_experts.size()) outfile << ",";
                    }
                    outfile << "],";

                    outfile << "\"candidate_scores\":[";
                    for (size_t e_idx = 0; e_idx < pred_res.scores.size(); ++e_idx) {
                        outfile << std::setprecision(10) << pred_res.scores[e_idx].score;
                        if (e_idx + 1 < pred_res.scores.size()) outfile << ",";
                    }
                    outfile << "],";

                    outfile << "\"hits\":" << metrics.hits << ","
                            << "\"precision\":" << std::setprecision(8) << metrics.precision << ","
                            << "\"recall\":" << std::setprecision(8) << metrics.recall << ","
                            << "\"f1\":" << std::setprecision(8) << metrics.f1 << ","
                            << "\"hit_rate\":" << std::setprecision(8) << metrics.hit_rate << ","
                            << "\"mrr\":" << std::setprecision(8) << metrics.mrr << ","
                            << "\"ndcg\":" << std::setprecision(8) << metrics.ndcg << ","
                            << "\"latency_ns\":" << std::setprecision(4) << latency_ns << ","
                            << "\"state\":" << state_json
                            << "}\n";
                }

                // 6. Reveal ground truth target labels and trigger causal state update
                predictor->update(
                    req_id, tok_id, source_stage, stage_id, source_experts, actual_experts);
            }
        }
    }

    if (outfile.is_open()) {
        outfile.close();
        std::cout << "[REPLAY_PREDICTORS] Wrote event logs to " << output_path << "\n";
    }

    // Print summary report
    std::cout << "\n======================================================================\n";
    std::cout << "REPLAY SUMMARY: " << predictor_name << " (Budget = " << budget << ")\n";
    std::cout << "======================================================================\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Metric         | Overall   | Cold (T=0)| Warm (1-3)| Steady (>=4)\n";
    std::cout << "---------------+-----------+-----------+-----------+-------------\n";
    std::cout << "Events         | " << std::setw(9) << acc_overall.count << " | "
              << std::setw(9) << acc_cold.count << " | "
              << std::setw(9) << acc_warm.count << " | "
              << std::setw(11) << acc_steady.count << "\n";
    std::cout << "Precision@B    | " << std::setw(9) << acc_overall.mean_precision() << " | "
              << std::setw(9) << acc_cold.mean_precision() << " | "
              << std::setw(9) << acc_warm.mean_precision() << " | "
              << std::setw(11) << acc_steady.mean_precision() << "\n";
    std::cout << "Recall@B       | " << std::setw(9) << acc_overall.mean_recall() << " | "
              << std::setw(9) << acc_cold.mean_recall() << " | "
              << std::setw(9) << acc_warm.mean_recall() << " | "
              << std::setw(11) << acc_steady.mean_recall() << "\n";
    std::cout << "F1@B           | " << std::setw(9) << acc_overall.mean_f1() << " | "
              << std::setw(9) << acc_cold.mean_f1() << " | "
              << std::setw(9) << acc_warm.mean_f1() << " | "
              << std::setw(11) << acc_steady.mean_f1() << "\n";
    std::cout << "Hit Rate       | " << std::setw(9) << acc_overall.mean_hit_rate() << " | "
              << std::setw(9) << acc_cold.mean_hit_rate() << " | "
              << std::setw(9) << acc_warm.mean_hit_rate() << " | "
              << std::setw(11) << acc_steady.mean_hit_rate() << "\n";
    std::cout << "MRR            | " << std::setw(9) << acc_overall.mean_mrr() << " | "
              << std::setw(9) << acc_cold.mean_mrr() << " | "
              << std::setw(9) << acc_warm.mean_mrr() << " | "
              << std::setw(11) << acc_steady.mean_mrr() << "\n";
    std::cout << "nDCG@B         | " << std::setw(9) << acc_overall.mean_ndcg() << " | "
              << std::setw(9) << acc_cold.mean_ndcg() << " | "
              << std::setw(9) << acc_warm.mean_ndcg() << " | "
              << std::setw(11) << acc_steady.mean_ndcg() << "\n";
    std::cout << "Mean Lat (ns)  | " << std::setw(9) << acc_overall.mean_latency_ns() << " | "
              << std::setw(9) << acc_cold.mean_latency_ns() << " | "
              << std::setw(9) << acc_warm.mean_latency_ns() << " | "
              << std::setw(11) << acc_steady.mean_latency_ns() << "\n";
    std::cout << "======================================================================\n";

    return 0;
}

