#include "hardware/layer_impl.h"

#include "dram/dram_interface.h"
#include "hardware/device.h"
#include "module/tensor.h"

namespace llm_system {

ExecStatus issueRamulator(Device_Ptr device, LayerType layer_type,
                          ProcessorType processor_type,
                          DRAMRequestType dram_request_type,
                          PIMOperandType pim_operand_type, Tensor_Ptr tensor, MemoryTarget target) {
  CacheKey key = std::make_tuple(layer_type, processor_type, dram_request_type,
                                 tensor->getSize(), target);
  ExecStatus exec_status;
  // NOTE on CacheKey limitation:
  // CacheKey consists of (LayerType, ProcessorType, DRAMRequestType, size, MemoryTarget).
  // It does NOT contain physical DDR5 addresses, expert IDs, layer IDs, or request IDs.
  // When bypass_offchip_execution_cache is false (default), identically sized off-chip
  // tensor transfers reuse the memoized ExecStatus from the initial Ramulator simulation.
  // This memoized shortcut is an analytical acceleration for sequential decoding latency,
  // NOT a cycle-accurate DRAM simulation of row-buffer conflicts or bus contention.
  // For cycle-accurate contention modeling (e.g., asynchronous prefetching), set
  // bypass_offchip_execution_cache = true.
  bool bypass_cache = false;
  if (target == MemoryTarget::OFFCHIP_DRAM) {
    if (device->config.bypass_offchip_execution_cache) {
      bypass_cache = true;
    }
  }

  if (bypass_cache || !device->checkExecutionCache(exec_status, key)) {
    DRAMRequest::Ptr dram_request = DRAMRequest::Create(dram_request_type);
    if (target == MemoryTarget::OFFCHIP_DRAM && device->offchip_mmap_controller) {
      auto offchip_mem_obj = MemoryObject::Create(
          tensor->getMMap(), tensor->getMemoryObject()->getLogicAddr(),
          tensor->getSize(), device->offchip_mmap_controller);
      dram_request->AddOperand(offchip_mem_obj, pim_operand_type);
    } else {
      dram_request->AddOperand(tensor->getMemoryObject(), pim_operand_type);
    }
    device->run_ramulator(dram_request, target);
    if (target == MemoryTarget::OFFCHIP_DRAM && device->offchip_dram_interface) {
      exec_status = device->offchip_dram_interface->getExecStatus(); 
    } else {
      exec_status = device->dram_interface->getExecStatus(); 
    }
    
    if (!bypass_cache) {
        device->addExecutionCache(exec_status, key);
    }
  }

  return exec_status;
};

ExecStatus issueRamulatorBatch(Device_Ptr device, LayerType layer_type,
                          ProcessorType processor_type,
                          DRAMRequestType dram_request_type,
                          PIMOperandType pim_operand_type, const std::vector<Tensor_Ptr>& tensors, MemoryTarget target) {
  long total_size = 0;
  for (auto tensor : tensors) {
      total_size += tensor->getSize();
  }
  
  CacheKey key = std::make_tuple(layer_type, processor_type, dram_request_type,
                                 total_size, target);
  ExecStatus exec_status;
  // NOTE on CacheKey limitation:
  // CacheKey consists of (LayerType, ProcessorType, DRAMRequestType, size, MemoryTarget).
  // It does NOT contain physical DDR5 addresses, expert IDs, layer IDs, or request IDs.
  // When bypass_offchip_execution_cache is false (default), identically sized off-chip
  // tensor transfers reuse the memoized ExecStatus from the initial Ramulator simulation.
  // This memoized shortcut is an analytical acceleration for sequential decoding latency,
  // NOT a cycle-accurate DRAM simulation of row-buffer conflicts or bus contention.
  // For cycle-accurate contention modeling (e.g., asynchronous prefetching), set
  // bypass_offchip_execution_cache = true.
  bool bypass_cache = false;
  if (target == MemoryTarget::OFFCHIP_DRAM) {
    if (device->config.bypass_offchip_execution_cache) {
      bypass_cache = true;
    }
  }

  if (bypass_cache || !device->checkExecutionCache(exec_status, key)) {
    std::list<DRAMRequest::Ptr> requests;
    for (auto tensor : tensors) {
      DRAMRequest::Ptr dram_request = DRAMRequest::Create(dram_request_type);
      if (target == MemoryTarget::OFFCHIP_DRAM && device->offchip_mmap_controller) {
        auto offchip_mem_obj = MemoryObject::Create(
            tensor->getMMap(), tensor->getMemoryObject()->getLogicAddr(),
            tensor->getSize(), device->offchip_mmap_controller);
        dram_request->AddOperand(offchip_mem_obj, pim_operand_type);
      } else {
        dram_request->AddOperand(tensor->getMemoryObject(), pim_operand_type);
      }
      requests.push_back(dram_request);
    }
    
    device->run_ramulator_batch(requests, target);
    
    if (target == MemoryTarget::OFFCHIP_DRAM && device->offchip_dram_interface) {
      exec_status = device->offchip_dram_interface->getExecStatus(); 
    } else {
      exec_status = device->dram_interface->getExecStatus(); 
    }
    
    if (!bypass_cache) {
        device->addExecutionCache(exec_status, key);
    }
  }

  return exec_status;
};

ExecStatus getIdealMemoryStatus(Device_Ptr device, ProcessorType processor_type,
                          DRAMRequestType dram_request_type, Tensor_Ptr tensor) {
  
  ExecStatus exec_status;
  long total_size = tensor->getSize();
  device->run_ideal(dram_request_type, tensor);
  exec_status = device->dram_interface->getExecStatus(); 
  return exec_status;
};

}  // namespace llm_system