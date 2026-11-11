#include "scheduler/sequence.h"

#include "scheduler/scheduler.h"
#include "hardware/device.h"
#include "prefetch/live_prediction_engine.h"

namespace llm_system {
Sequence::Sequence(int expert_seq_id, int input_len, int output_len)
    : expert_seq_id(expert_seq_id),
      input_len(input_len),
      output_len(output_len) {
  id = count_id;
  count_id++;
  num_process_token = 0;
  current_len = 0;
  total_len = input_len + output_len;
  get_expert_from_list = true;

  gen_start_time = 0;

  assertTrue(input_len > 0, "input len is not valid, input len is " +
                                std::to_string(input_len));
  assertTrue(output_len > 0, "output len is not valid, output len is " +
                                 std::to_string(output_len));

  arrival_time = 0;
  first_token_time = 0;
  end_token_time = 0;
  queueing_delay = 0;
  num_sum_iter = 0;
  sum_stage = true;
  record = false;
}

void Sequence::setGenStartTime(time_ns time) { gen_start_time = time; }

void Sequence::update(time_ns time) {
  // update end_token_time
  end_token_time += time;

  if (sum_stage) {
    first_token_time += time;
    num_sum_iter++;
  }

  // update status
  current_len += num_process_token;
  num_process_token = 0;
  if (sum_stage && current_len == input_len) {
    sum_stage = false;
  }
}

void BatchedSequence::add(Sequence::Ptr seq) {
  sequence.push_back(seq);
  return;
}

std::vector<Sequence::Ptr> BatchedSequence::get_seq() { return sequence; }

std::vector<Sequence::Ptr> BatchedSequence::get_sum() {
  std::vector<Sequence::Ptr> sum;
  sum.resize(0);
  for (auto& seq : sequence) {
    if (seq->current_len < seq->input_len) {
      sum.push_back(seq);
    }
  }
  return sum;
}

std::vector<Sequence::Ptr> BatchedSequence::get_gen() {
  std::vector<Sequence::Ptr> gen;
  for (auto& seq : sequence) {
    if (seq->current_len >= seq->input_len) {
      gen.push_back(seq);
    }
  }
  return gen;
}

void BatchedSequence::update(time_ns time) {
  for (auto& seq : sequence) {
    seq->update(time);
  }
  if (scheduler != nullptr && scheduler->prediction_engine != nullptr) {
    if (scheduler->system_config.prefetch_config.enabled &&
        scheduler->system_config.prefetch_config.mode == PrefetchMode::HARDWARE_ASYNC &&
        scheduler->system_config.prefetch_config.pabi_pinning_enabled) {
      std::shared_ptr<Device> dev = scheduler->get_device();
      if (dev != nullptr && last_completed_stage >= 0) {
        scheduler->prediction_engine->expire_pins_for_stage(dev, last_completed_stage);
        last_completed_stage = -1;
      }
    }
  }
  // cur += 32; // cur += num_layer임
}

void BatchedSequence::pop(Sequence::Ptr seq) {
  for (auto target = sequence.begin(); target != sequence.end();) {
    if ((*target)->id == seq->id) {
      target = sequence.erase(target);
    } else {
      target++;
    }
  }
}

int BatchedSequence::get_num_seq() { return sequence.size(); }

int BatchedSequence::get_process_token() {
  int token = 0;

  for (auto& seq : sequence) {
    token += seq->num_process_token;
  }
  return token;
}

int BatchedSequence::get_gen_process_token() {
  int token = 0;
  std::vector<Sequence::Ptr> gen_seq = get_gen();

  for (auto& seq : gen_seq) {
    token += seq->num_process_token;
  }
  return token;
}

int BatchedSequence::get_sum_process_token() {
  int token = 0;
  std::vector<Sequence::Ptr> sum_seq = get_sum();

  for (auto& seq : sum_seq) {
    token += seq->num_process_token;
  }
  return token;
}

int BatchedSequence::get_average_sequence_length() {
  std::vector<Sequence::Ptr> gen_seq = get_gen();
  int total_length = 0;
  for (auto seq : gen_seq) {
    total_length += seq->current_len;
  }
  if (gen_seq.size()) {
    total_length /= gen_seq.size();
  } else {
    total_length = 0;
  }
  return total_length;
}

int BatchedSequence::get_total_sequence_length() {
  std::vector<Sequence::Ptr> gen_seq = get_gen();
  int total_length = 0;
  for (auto seq : gen_seq) {
    total_length += seq->current_len;
  }

  return total_length;
}

void BatchedSequence::update_expert(int num_expert, int top_k,
                                    bool need_new_expert) {
  num_token_in_expert.clear();
  int size = num_token_in_expert.size();
  int num_layer = scheduler->model_config.num_layers;
  if (scheduler->model_config.model_name == "deepseekV3" || scheduler->model_config.model_name == "debugV3") {
    num_layer = scheduler->model_config.num_layers - scheduler->model_config.first_k_dense;
  } else if (scheduler->model_config.expert_freq != 0) {
    num_layer = scheduler->model_config.num_layers / scheduler->model_config.expert_freq;
  }

  static int expert_id = 0;

  // init
  if (size == 0) { 
    for (int i = 0; i < num_expert; i++) {
      num_token_in_expert.push_back(0);
    }
  }
  clear_expert();
  if (scheduler != nullptr && scheduler->prediction_engine != nullptr) {
    if (scheduler->system_config.prefetch_config.enabled &&
        scheduler->system_config.prefetch_config.mode == PrefetchMode::HARDWARE_ASYNC &&
        scheduler->system_config.prefetch_config.pabi_pinning_enabled) {
      std::shared_ptr<Device> dev = scheduler->get_device();
      if (dev != nullptr && last_completed_stage >= 0) {
        scheduler->prediction_engine->expire_pins_for_stage(dev, last_completed_stage);
        last_completed_stage = -1;
      }
    }
  }
  std::vector<double> skewness_weight(num_expert); // w ~ 1/k^s
  for (int i = 0 ; i < num_expert; i++){
    skewness_weight[i] = 1.0 / pow((i + 1), scheduler->model_config.skewness);
  }

  static bool init_routing = false;
  static std::ofstream routing_file;
  if (!init_routing) {
    const char* trace_dir_env = std::getenv("TRACE_DIR");
    std::string path = "expert_routing.csv";
    if (trace_dir_env) {
      path = std::string(trace_dir_env) + "/expert_routing.csv";
    }
    routing_file.open(path, std::ios::app);
    routing_file.seekp(0, std::ios::end);
    if (routing_file.tellp() == 0) {
      routing_file << "request_id,token_id,layer_id,topk_rank,expert_id\n";
    }
    init_routing = true;
  }

  current_moe_stage_id = cur_layer;

  for (auto seq : sequence) {
    if (seq->num_process_token == 0) {
      continue;
    }
    int seq_id = seq->expert_seq_id;
    int request_id = seq->id;
    int current_len = seq->current_len;

    for (int token_id = current_len;
         token_id < current_len + seq->num_process_token; token_id++) {
      if (scheduler != nullptr && scheduler->prediction_engine != nullptr) {
        scheduler->prediction_engine->on_stage_routing_start(
            request_id, token_id, cur_layer, seq_id);
        if (scheduler->system_config.prefetch_config.enabled &&
            scheduler->system_config.prefetch_config.mode == PrefetchMode::HARDWARE_ASYNC) {
          std::shared_ptr<Device> dev = scheduler->get_device();
          if (dev != nullptr) {
            double cur_time = static_cast<double>(dev->get_time());
            scheduler->prediction_engine->dispatch_candidates(dev, cur_time);
          }
        }
      }

      std::vector<int> actual_target_experts;

      if (seq->get_expert_from_list) {
        if (seq_id < 0 || seq_id >= (int)scheduler->sequences_info.size()) {
          throw std::runtime_error("[ERROR] Invalid seq_id " + std::to_string(seq_id) + " for sequences_info of size " + std::to_string(scheduler->sequences_info.size()));
        }
        int rel_token_id = token_id - seq->input_len;
        if (rel_token_id < 0) rel_token_id = 0;
        size_t table_idx = (size_t)rel_token_id * num_layer + cur_layer;
        if (table_idx >= scheduler->sequences_info[seq_id]->expert_list.size()) {
          throw std::runtime_error("[ERROR] Trace out of bounds: table_idx " + std::to_string(table_idx) + " >= expert_list.size " + std::to_string(scheduler->sequences_info[seq_id]->expert_list.size()) + " (token_id=" + std::to_string(token_id) + ", rel_token_id=" + std::to_string(rel_token_id) + ", cur_layer=" + std::to_string(cur_layer) + ")");
        }
        const auto& routed = scheduler->sequences_info[seq_id]->expert_list[table_idx];
        if ((int)routed.size() < top_k) {
          throw std::runtime_error("[ERROR] Incomplete trace record: expected " + std::to_string(top_k) + " experts, got " + std::to_string(routed.size()) + " for token " + std::to_string(token_id) + " layer " + std::to_string(cur_layer));
        }
        actual_target_experts.reserve(top_k);
        for (int k = 0; k < top_k; k++) {
          int e_id = routed[k];
          if (e_id < 0 || e_id >= num_expert) {
            throw std::runtime_error("[ERROR] Out-of-range expert_id " + std::to_string(e_id) + " (num_expert=" + std::to_string(num_expert) + ")");
          }
          actual_target_experts.push_back(e_id);
          num_token_in_expert[e_id] += 1;
          routing_file << request_id << "," << token_id << "," << cur_layer << "," << (k + 1) << "," << e_id << "\n";
        }
      } else {
        std::vector<int> expert_vec;
        auto route_key = std::make_tuple(request_id, seq_id, token_id, cur_layer);
        auto it = cached_routing.find(route_key);
        if (it != cached_routing.end()) {
          expert_vec = it->second;
        } else {
          std::set<int> expert_list;
          if (scheduler->model_config.skewness > 0) {
            expert_list = scheduler->getZipfianRandomExpert(skewness_weight, top_k); // for skewed expert test
          } else {
            expert_list = scheduler->getRandomExpert(top_k);
          }
          for (auto idx : expert_list) {
            expert_vec.push_back(idx);
          }
          cached_routing[route_key] = expert_vec;
        }
        actual_target_experts = expert_vec;
        int k_rank = 1;
        for (auto idx : expert_vec) {
          num_token_in_expert[idx] += 1;
          routing_file << request_id << "," << token_id << "," << cur_layer << "," << k_rank << "," << idx << "\n";
          k_rank++;
        }
      }

      double cur_dev_time = 0.0;
      if (scheduler != nullptr && scheduler->get_device() != nullptr) {
        cur_dev_time = static_cast<double>(scheduler->get_device()->get_time());
      }
      if (scheduler != nullptr && scheduler->prediction_engine != nullptr) {
        scheduler->prediction_engine->on_stage_routing_complete(
            request_id, token_id, cur_layer, actual_target_experts, seq_id, cur_dev_time);
      }

      if (scheduler != nullptr && scheduler->prediction_engine != nullptr) {
        if (scheduler->system_config.prefetch_config.enabled &&
            scheduler->system_config.prefetch_config.mode == PrefetchMode::HARDWARE_ASYNC) {
          std::shared_ptr<Device> dev = scheduler->get_device();
          if (dev != nullptr) {
            double cur_time = static_cast<double>(dev->get_time());
            scheduler->prediction_engine->dispatch_candidates(dev, cur_time);
          }
        }
      }
    }
  }
  if (routing_file.is_open()) {
    routing_file.flush();
  }

  last_completed_stage = cur_layer;
  cur_layer++;
  cur_layer %= num_layer;
}

void BatchedSequence::clear_expert() {
  int size = num_token_in_expert.size();
  for (int i = 0; i < size; i++) {
    num_token_in_expert.at(i) = 0;
  }
}

void BatchedSequence::add_sequence(std::vector<int> expert_seq_ids) {
  for (int seq_id : expert_seq_ids) {
    int Lin = scheduler->sequences_info.at(seq_id)->Lin;
    int Lout = scheduler->sequences_info.at(seq_id)->Lout;
    Sequence::Ptr seq = Sequence::Create(seq_id, Lin, Lout);
    add(seq);
  }
}

void BatchedSequence::add_dummy_sequence(int num_seq, int input_len,
                                         int output_len) {
  for (int i = 0; i < num_seq; i++) {
    Sequence::Ptr seq = Sequence::Create(0, input_len, output_len);
    seq->get_expert_from_list = false;
    add(seq);
  }
}

}  // namespace llm_system