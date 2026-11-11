
#include "linear.h"

#include "common/assert.h"
#include "hardware/hardware_config.h"
#include "scheduler/scheduler.h"
// Linear //

namespace llm_system {

Linear::Linear(std::string& prefix, std::string& name, int input_size,
               int output_size, std::vector<int> device_list,
               Device::Ptr device)
    : Module(prefix, name, device, device_list, true) {
  std::vector<int> shape = {input_size, output_size};

  Tensor::Ptr A = Tensor::Create("A", shape, "weight", device, device->model_config.precision_byte);
  Tensor::Ptr Y = Tensor::Create("Y", shape, "act", device, device->model_config.precision_byte);
  add_tensor(A);
  add_tensor(Y);
}

Tensor::Ptr Linear::forward(const Tensor::Ptr input,
                            BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr A = tensor_list.at("A");

  int m = input->shape[0];
  int k = input->shape[1];

  int n = A->shape[1];

  assertTrue(k == A->shape[0],
             module_map_name +
                 ": Dimension of input tensor for Linear is not "
                 "matched, expected " +
                 std::to_string(A->shape[0]) + " but " + std::to_string(k));

  std::vector<int> shape = {m, n};
  Tensor::Ptr output = get_activation("Y", shape);
  output->perform_with_optimal = input->perform_with_optimal;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.push_back(input);
  tensor_list.push_back(A);
  tensor_list.push_back(output);

  A->weight_target = MemoryTarget::HBM;
  A->load_sisters = false;
  if (A->is_expert_weight && input->getSize() > 0) {
    int layer_id = sequences_metadata->current_moe_stage_id;
    A->expert_layer_id = layer_id;
    auto seq_list = sequences_metadata->get_seq();
    if (!seq_list.empty()) {
      auto first_seq = seq_list[0];
      A->request_id = first_seq->id;
      A->seq_id = first_seq->expert_seq_id;
      A->token_id = first_seq->current_len;
      for (auto s : seq_list) {
        int s_seq_id = s->expert_seq_id;
        int s_tok_id = s->current_len;
        if (s->get_expert_from_list) {
          int rel_tok = s_tok_id - s->input_len;
          if (rel_tok < 0) {
            rel_tok = 0;
          }
          if (sequences_metadata->scheduler != nullptr && s_seq_id >= 0 && s_seq_id < (int)sequences_metadata->scheduler->sequences_info.size()) {
            int num_layer = sequences_metadata->scheduler->model_config.num_layers;
            if (sequences_metadata->scheduler->model_config.model_name == "deepseekV3" || sequences_metadata->scheduler->model_config.model_name == "debugV3") {
              num_layer = sequences_metadata->scheduler->model_config.num_layers - sequences_metadata->scheduler->model_config.first_k_dense;
            } else if (sequences_metadata->scheduler->model_config.expert_freq != 0) {
              num_layer = sequences_metadata->scheduler->model_config.num_layers / sequences_metadata->scheduler->model_config.expert_freq;
            }
            size_t t_idx = (size_t)rel_tok * num_layer + layer_id;
            if (t_idx < sequences_metadata->scheduler->sequences_info[s_seq_id]->expert_list.size()) {
              for (int eid : sequences_metadata->scheduler->sequences_info[s_seq_id]->expert_list[t_idx]) {
                if (eid == A->expert_id) {
                  A->request_id = s->id;
                  A->seq_id = s_seq_id;
                  A->token_id = s_tok_id;
                  break;
                }
              }
            }
          }
        } else {
          auto route_key = std::make_tuple(s->id, s_seq_id, s_tok_id, layer_id);
          auto it = sequences_metadata->cached_routing.find(route_key);
          if (it != sequences_metadata->cached_routing.end()) {
            for (int eid : it->second) {
              if (eid == A->expert_id) {
                A->request_id = s->id;
                A->seq_id = s_seq_id;
                A->token_id = s_tok_id;
                break;
              }
            }
          }
        }
      }
    }
    if (!device->config.offchip_dram_cfg_path.empty() || device->config.use_event_driven_dram) {
      if (device->config.use_event_driven_dram) {
        device->set_current_stage(layer_id);
        device->process_completed_prefetches(device->get_time());
      }
      bool is_primary = A->is_primary_expert_weight;
      if (!device->is_expert_resident(layer_id, A->expert_id, true, is_primary)) {
        A->weight_target = MemoryTarget::OFFCHIP_DRAM;
        A->load_sisters = true;
        for (auto sister : A->sister_weights) {
          sister->request_id = A->request_id;
          sister->seq_id = A->seq_id;
          sister->token_id = A->token_id;
        }
      }
    }
  }

  // device->execution(total_flops);

  LayerInfo info;
  info.processor_type = device->config.processor_type;
  if (input->parallel_execution) {
    info.parallel_execution = true;
    if (input->isPerformHigh()) {
      info.processor_type = {device->config.high_processor_type};
      output->setPerformHigh();
    } else {
      info.processor_type = {device->config.low_processor_type};
      output->setPerformLow();
    }
  } else if (device->config.hetero_subbatch) {
    info.processor_type = {device->config.high_processor_type};
  } else if (device->config.processor_type.size() != 1) {
    if (!input->perform_with_optimal){
      info.processor_type = {device->config.high_processor_type};
    }
  }

  device->execution(LayerType::LINEAR, tensor_list, sequences_metadata, info);

  return output;
}

BatchedLinear::BatchedLinear(std::string& prefix, std::string& name, int num_batched_gemm, int input_size,
               int output_size, bool duplicated_input, bool use_plain_linear, std::vector<int> device_list,
               Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
    duplicated_input(duplicated_input),
    use_plain_linear(use_plain_linear) {
  if(!use_plain_linear){
    std::vector<int> shape = {num_batched_gemm, input_size, output_size};
    Tensor::Ptr batched_linear_input = Tensor::Create("batched_linear_input", shape, "act", device, device->model_config.precision_byte);
    Tensor::Ptr batched_linear_output = Tensor::Create("batched_linear_output", shape, "act", device, device->model_config.precision_byte);
    Tensor::Ptr weight = Tensor::Create("weight", shape, "weight", device, device->model_config.precision_byte);

    shape = {input_size, output_size};
    Tensor::Ptr output = Tensor::Create("output", shape, "act", device, device->model_config.precision_byte);

    add_tensor(batched_linear_input);
    add_tensor(batched_linear_output);
    add_tensor(weight);
    add_tensor(output);
  }
  else{
    std::vector<int> shape = {input_size, output_size};
    Tensor::Ptr plain_linear_input = Tensor::Create("plain_linear_input", shape, "act", device, device->model_config.precision_byte);
    Tensor::Ptr plain_linear_output = Tensor::Create("plain_linear_output", shape, "act", device, device->model_config.precision_byte);
    Tensor::Ptr weight = Tensor::Create("weight", shape, "weight", device, device->model_config.precision_byte);

    add_tensor(plain_linear_input);
    add_tensor(plain_linear_output);
    add_tensor(weight);
  }
}

TensorVec BatchedLinear::forward(const TensorVec input,
                            BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr weight = tensor_list.at("weight");
  TensorVec output_vec;

  if(!use_plain_linear){
    int num_batched_gemm = input.size();
    int m = input[0]->shape[0];
    int k = input[0]->shape[1];
    int n = weight->shape[2];

    Tensor::Ptr batched_linear_input = get_activation("batched_linear_input", {num_batched_gemm, m, k});
    Tensor::Ptr batched_linear_output = get_activation("batched_linear_output", {num_batched_gemm, m, n});

    assertTrue(k == weight->shape[1],
              module_map_name +
                  ": Dimension of input tensor for Linear is not "
                  "matched, expected " +
                  std::to_string(weight->shape[1]) + " but " + std::to_string(k));

    std::vector<int> shape = {m, n};
    Tensor::Ptr output = get_activation("output", shape);

    std::vector<Tensor::Ptr> tensor_list;
    tensor_list.push_back(batched_linear_input);
    tensor_list.push_back(weight);
    tensor_list.push_back(batched_linear_output);

    LayerInfo info;
    info.processor_type = device->config.processor_type;
    if (input[0]->parallel_execution) {
      info.parallel_execution = true;
      if (input[0]->isPerformHigh()) {
        info.processor_type = {device->config.high_processor_type};
        output->setPerformHigh();
      } else {
        info.processor_type = {device->config.low_processor_type};
        output->setPerformLow();
      }
    } else if (device->config.processor_type.size() != 1) {
      if (!input[0]->perform_with_optimal){
        info.processor_type = {device->config.high_processor_type};
      }
    }
    

    info.duplicated_input = duplicated_input;
    
    for(int bg = 0; bg < num_batched_gemm; bg++){
      output_vec.push_back(output);
    }
    
    device->execution(LayerType::BATCHED_LINEAR, tensor_list, sequences_metadata, info);
  }
  else{ // plain_linear
    int num_heads = input.size();
    int batch_size = input[0]->shape[0];
    int col = input[0]->shape[1] * num_heads;
    int weight_row = weight->shape[0];
    int weight_col = weight->shape[1];

    Tensor::Ptr plain_linear_input = get_activation("plain_linear_input", {batch_size, col});
    Tensor::Ptr plain_linear_output = get_activation("plain_linear_output", {batch_size, weight_col});

    assertTrue(col == weight_row,
              module_map_name +
                  ": Dimension of input tensor for Linear is not "
                  "matched, expected " +
                  std::to_string(weight_row) + " but " + std::to_string(col));

    std::vector<Tensor::Ptr> tensor_list;
    tensor_list.push_back(plain_linear_input);
    tensor_list.push_back(weight);
    tensor_list.push_back(plain_linear_output);

    LayerInfo info;
    info.processor_type = device->config.processor_type;
    if (input[0]->parallel_execution) {
      info.parallel_execution = true;
      if (input[0]->isPerformHigh()) {
        info.processor_type = {device->config.high_processor_type};
        plain_linear_output->setPerformHigh();
      } else {
        info.processor_type = {device->config.low_processor_type};
        plain_linear_output->setPerformLow();
      }
    } else if (device->config.processor_type.size() != 1) {
      if (!input[0]->perform_with_optimal){
        info.processor_type = {device->config.high_processor_type};
      }
    }
    

    info.duplicated_input = duplicated_input;
    
    output_vec.push_back(plain_linear_output);
    
    device->execution(LayerType::LINEAR, tensor_list, sequences_metadata, info);
  }

  return output_vec;
}

}  // namespace llm_system