#include <memory>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"
#include "prefetch/prefetch_metrics_collector.h"

namespace llm_system {
class DRAMRequest;
class Tensor;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;

ExecStatus LinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  exec_status.compute_duration = compute_duration;

  time_ns weight_memory_cost = 0;
  bool is_demand_miss = false;
  
  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input, MemoryTarget::HBM);

    time_ns start_time = device->get_time();
    std::vector<Tensor_Ptr> batch_tensors;
    batch_tensors.push_back(weight);
    if (weight->load_sisters) {
      for (auto sister : weight->sister_weights) {
        sister->expert_layer_id = weight->expert_layer_id;
        batch_tensors.push_back(sister);
      }
      weight->load_sisters = false;
    }

    ExecStatus weight_status;
    if (weight->is_expert_weight) {
      if (weight->weight_target == MemoryTarget::OFFCHIP_DRAM) {
        if (device->config.use_event_driven_dram) {
          device->set_current_stage(weight->expert_layer_id);
          device->process_completed_prefetches(start_time);
          bool is_primary = weight->is_primary_expert_weight;
          if (!device->is_expert_resident(weight->expert_layer_id, weight->expert_id, true, is_primary)) {
            is_demand_miss = true;
          }
        } else {
          is_demand_miss = true;
        }
      }
    }

    if (weight->is_expert_weight && !is_demand_miss && device->config.use_event_driven_dram) {
      std::pair<int, int> exp_key = {weight->expert_layer_id, weight->expert_id};
      std::string hit_outcome = "RESIDENT_HIT";
      uint64_t cand_id = 0;
      uint64_t tr_id = 0;
      auto res_it = device->resident_experts.find(exp_key);
      if (res_it != device->resident_experts.end()) {
        cand_id = res_it->second.candidate_id;
        tr_id = res_it->second.transfer_id;
        if (res_it->second.is_prefetched && !res_it->second.was_demanded) {
          hit_outcome = "TIMELY_PREFETCH";
          res_it->second.was_demanded = true;
          if (tr_id != 0) {
            PrefetchMetricsCollector::getInstance().record_transfer_outcome(tr_id, "DEMANDED");
          }
        }
      }
      PrefetchMetricsCollector::getInstance().record_expert_demanded(
          cand_id, weight->request_id, weight->seq_id, weight->token_id,
          weight->expert_layer_id, weight->expert_id, start_time,
          start_time, 0.0, 0.0, hit_outcome, tr_id, cand_id);
    }

    uint64_t demand_transfer_id = 0;
    if (is_demand_miss && device->config.use_event_driven_dram) {
      long long total_expert_size = 0;
      for (auto t : batch_tensors) {
        total_expert_size += t->getSize();
      }

      uint64_t tid = device->submit_demand_transfer(
          weight->expert_layer_id, weight->expert_id, start_time, static_cast<uint64_t>(total_expert_size),
          weight->request_id, weight->seq_id, weight->token_id);
      demand_transfer_id = tid;

      bool transfer_success = false;
      if (tid != 0) {
        transfer_success = device->wait_for_demand_transfer(tid);
      }

      if (transfer_success) {
        if (device->ddr5_transfer_engine != nullptr) {
          const auto* tr = device->ddr5_transfer_engine->get_transfer(tid);
          if (tr != nullptr) {
            if (tr->completion_time_ns > start_time) {
              weight_memory_cost = static_cast<time_ns>(tr->completion_time_ns - start_time);
            } else {
              weight_memory_cost = 0;
            }
          }
        }
        device->mark_expert_resident(weight->expert_layer_id, weight->expert_id, total_expert_size);
      }
      device->clear_inflight_expert(weight->expert_layer_id, weight->expert_id);
      if (weight_memory_cost > 0) {
        PrefetchMetricsCollector::getInstance().record_exposed_stall(
            start_time, static_cast<double>(weight_memory_cost));
      }

      weight_status.memory_duration = weight_memory_cost;
      for (auto t : batch_tensors) {
        time_ns t_cost = 0;
        double bw = device->config.ddr5_bandwidth_gbps;
        if (bw > 0.0) {
          t_cost = static_cast<time_ns>(t->getSize() / bw);
        }
        weight_status.per_request_duration.push_back(t_cost);
      }
      exec_status += weight_status;
    } else {
      weight_status =
          issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU,
                              DRAMRequestType::kRead, PIMOperandType::kDRAM, batch_tensors, weight->weight_target);
      exec_status += weight_status;
      weight_memory_cost = weight_status.memory_duration;

      if (weight->is_expert_weight) {
        if (weight->weight_target == MemoryTarget::OFFCHIP_DRAM) {
          long long total_expert_size = 0;
          for (auto t : batch_tensors) {
            total_expert_size += t->getSize();
          }
          device->mark_expert_resident(weight->expert_layer_id, weight->expert_id, total_expert_size);
        }
      }
    }

    if (weight->is_expert_weight) {
      time_ns memory_cost = 0;
      if (weight->weight_target == MemoryTarget::OFFCHIP_DRAM) {
        memory_cost = weight_memory_cost;
      }
      time_ns end_time = start_time + memory_cost;

      static bool init_expert_load = false;
      static std::ofstream csv;
      if (!init_expert_load) {
        const char* trace_dir_env = std::getenv("TRACE_DIR");
        std::string path = "expert_load.csv";
        if (trace_dir_env != nullptr) {
          path = std::string(trace_dir_env) + "/expert_load.csv";
        }
        csv.open(path, std::ios::app);
        csv.seekp(0, std::ios::end);
        if (csv.tellp() == 0) {
          csv << "request_id,seq_id,token_id,layer_id,expert_id,weight_tensor,size_bytes,start,end,duration,memory_source,memory_destination,cache_hit_or_miss,transfer_id\n";
        }
        init_expert_load = true;
      }
      std::string memory_source = "HBM";
      if (weight->weight_target == MemoryTarget::OFFCHIP_DRAM) {
        memory_source = "OFFCHIP_DRAM";
      }
      std::string hit_miss = "hit";
      if (weight->weight_target == MemoryTarget::OFFCHIP_DRAM) {
        hit_miss = "miss";
      }

      long total_size = 0;
      for (size_t i = 0; i < batch_tensors.size(); i++) {
        auto t = batch_tensors[i];
        total_size += t->getSize();
        
        time_ns t_cost = 0;
        if (hit_miss == "miss" && i < weight_status.per_request_duration.size()) {
          t_cost = weight_status.per_request_duration[i];
        }
        
        csv << t->request_id << ","
            << t->seq_id << ","
            << t->token_id << ","
            << t->expert_layer_id << ","
            << t->expert_id << ","
            << t->name << ","
            << t->getSize() << ","
            << start_time << ","
            << start_time + t_cost << ","
            << t_cost << ","
            << memory_source << ","
            << "HBM" << ","
            << hit_miss << ","
            << demand_transfer_id << "\n";
      }

      csv << weight->request_id << ","
          << weight->seq_id << ","
          << weight->token_id << ","
          << weight->expert_layer_id << ","
          << weight->expert_id << ","
          << "all_weights" << ","
          << total_size << ","
          << start_time << ","
          << end_time << ","
          << memory_cost << ","
          << memory_source << ","
          << "HBM" << ","
          << hit_miss << ","
          << demand_transfer_id << "\n";
      csv.flush();
    }

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output, MemoryTarget::HBM);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, weight);
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  if (device->config.compute_time_multiplier > 0.0 && device->config.compute_time_multiplier != 1.0) {
    if (use_ramulator) {
      time_ns offchip_stall = 0;
      if (is_demand_miss) {
        offchip_stall = weight_memory_cost;
      }
      time_ns onchip_mem = 0;
      if (exec_status.memory_duration > offchip_stall) {
        onchip_mem = exec_status.memory_duration - offchip_stall;
      }
      time_ns onchip_duration = std::max(exec_status.compute_duration, onchip_mem);
      onchip_duration = static_cast<time_ns>(onchip_duration * device->config.compute_time_multiplier);
      exec_status.total_duration = onchip_duration + offchip_stall;
    } else {
      time_ns onchip_duration = std::max(exec_status.compute_duration, exec_status.memory_duration);
      onchip_duration = static_cast<time_ns>(onchip_duration * device->config.compute_time_multiplier);
      exec_status.total_duration = onchip_duration;
    }
  } else {
    exec_status.total_duration =
        std::max(exec_status.compute_duration, exec_status.memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  exec_status.opb = total_flops / total_memory_size;

  return exec_status;
};

ExecStatus LinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr output,
                                bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;

  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus LinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;

  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus BatchedLinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator, bool duplicated_input) {
  assertTrue(input->shape.size() == 3, "input tensor is not 3D tensor");
  assertTrue(weight->shape.size() == 3, "weight tensor is not 3D tensor");
  assertTrue(output->shape.size() == 3, "output tensor is not 3D tensor");

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int num_heads = input->shape[0];
  std::vector<int> input_orig_shape = input->shape;
  std::vector<int> weight_orig_shape = weight->shape;
  std::vector<int> output_orig_shape = output->shape;

  int m = input->shape[1];
  int k = input->shape[2];
  int n = weight->shape[2];

  hw_metric total_flops = 2.0 * m * k * n * 1.0 * num_heads;
  
  hw_metric total_memory_size;
  if(duplicated_input){
    total_memory_size = 1.0 * (m * k + k * n * 1.0  * num_heads + m * n * 1.0  * num_heads) * weight->precision_byte;
  }
  else{
    total_memory_size = 1.0 * (m * k + k * n + m * n) * 1.0  * num_heads * weight->precision_byte;
  }

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    if(duplicated_input){
      input->setShape({m, k});
    }
    else{
      input->setShape({m, k * num_heads});
    }
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    
    weight->setShape({k * num_heads, n});
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, weight);

    output->setShape({m, n * num_heads});
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  exec_status.opb = total_flops / total_memory_size;

  input->setShape(input_orig_shape);
  weight->setShape(weight_orig_shape);
  output->setShape(output_orig_shape);

  return exec_status;
};

ExecStatus BatchedLinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr output,
                                bool use_ramulator, bool duplicated_input) {
  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus BatchedLinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator, bool duplicated_input) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;

  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

}  // namespace llm_system