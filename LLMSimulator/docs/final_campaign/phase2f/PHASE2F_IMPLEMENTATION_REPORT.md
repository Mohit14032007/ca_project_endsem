# Phase 2F — Implementation Report: Compute-Time Multiplier, Overlap Architecture, and Clock Continuity

## 1. Executive Summary

This report documents the architectural modifications implemented for **Phase 2F: Synthetic Compute-Window Sensitivity Evaluation** in `LLMSimulator`. 

Phase 2F investigated whether training-free expert prediction and forward-lookahead prefetching can reduce exposed memory stalls when the available computation window is expanded. The evaluation required three core architectural enhancements:
1. **Configurable Compute-Time Multiplier:** Synthetically scales on-chip compute duration across $M \in \{1.0, 4.0, 8.0, 12.0, 16.0\}$ via YAML configuration (`simulation.optimization.compute_time_multiplier`) and environment override (`COMPUTE_MULTIPLIER`).
2. **Total Duration & Overlap Architecture Decoupling:** In the baseline simulator, `total_duration = max(compute_duration, memory_duration)` masked compute scaling behind memory durations. The decoupling scales on-chip execution duration while strictly preserving physical off-chip DDR5 transfer latencies ($3.90116\text{ ms}$ at $22.578\text{ GB/s}$) and unscaled memory stalls.
3. **Inter-Token Monotonic Clock Continuity:** Prevents backward simulated time jumps across decoding token boundaries in `Cluster::restartModuleGraph()`, ensuring that `DDR5TransferEngine` maintains an uninterrupted monotonic event queue.

All modifications strictly preserve the authoritative historical baseline anchor (`d5bf61ff0c066c06f72173189f9924329f613a88`) and contain zero ternary operators.

---

## 2. Architectural Modifications

### 2.1 Configurable Compute Multiplier in Hardware & System Configuration
In [`src/hardware/hardware_config.h`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/hardware_config.h):
- Added `double compute_time_multiplier = 1.0;` and `bool clock_continuity = true;` to `SystemConfig` constructor parameters, member fields, and default system profiles (`A100`, `H100`, `B100`, `B200`).
- Implemented environment variable override parsing in `SystemConfig::apply_environment_overrides()`:
  - `COMPUTE_MULTIPLIER`: Parses string to `double`, applies override if $> 0.0$.
  - `CLOCK_CONTINUITY`: Parses `1/0/true/false/on/off/yes/no` flags without ternary operators.
- In [`eval/test_deepseekv3_mixtral_style.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp), added hierarchical YAML extraction checking `simulation.optimization.compute_time_multiplier`, `system.optimization.compute_time_multiplier`, and `simulation.compute_time_multiplier`.

### 2.2 Device FLOP Scaling & Analytical Peak Adjustments
In [`src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp):
- In `Device::Device(SystemConfig config, int device_total_rank, Cluster_ptr cluster)`:
  ```cpp
  compute_peak_flops = config.compute_peak_flops;
  if (config.compute_time_multiplier > 0.0 && config.compute_time_multiplier != 1.0) {
    this->config.compute_peak_flops = this->config.compute_peak_flops / config.compute_time_multiplier;
    compute_peak_flops = this->config.compute_peak_flops;
  }
  ```
- This analytically scales `flops / compute_peak_flops` across all computational layers (LayerNorm, RoPE, Residual, Attention, and Router).

### 2.3 Overlap Decoupling in `LinearExecutionGPU`
In [`src/hardware/linear_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp):
Decode GEMMs at batch size 1 have arithmetic intensity $\approx 1.0\text{ FLOP/byte}$, which falls far to the left of the A100 roofline knee ($485\text{ FLOP/byte}$). In baseline code, reading weights from HBM takes $\approx 6.5\text{ \mu s}$, while raw FLOP compute duration is only $\approx 376\text{ ns}$. If only raw FLOP duration were multiplied by $M$, `max(compute_duration, memory_duration)` would continue selecting memory duration for $M \le 16$, advancing device time by 0 additional nanoseconds.

To allow compute duration scaling to genuinely advance simulated device time without scaling off-chip DDR5 miss stalls:
```cpp
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
```
**Invariants Preserved:**
1. At $M=1.0$: Executes `std::max(compute_duration, memory_duration)`, producing a bitwise identical match to historical baseline execution.
2. At $M > 1.0$: On-chip execution duration scales by $M$, expanding the inter-stage computation window available to `DDR5TransferEngine::run_until()`.
3. Off-chip DDR5 miss latency (`weight_memory_cost`) is strictly isolated and unscaled ($3.90116\text{ ms}$).

### 2.4 Monotonic Global Inter-Token Simulated Timebase
In [`src/hardware/cluster.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/cluster.cpp) and [`eval/test_deepseekv3_mixtral_style.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp):
- In `Cluster::restartModuleGraph()`:
  ```cpp
  time_ns prev_time = 0;
  if (device->config.clock_continuity) {
    prev_time = device->status.device_time;
  }
  if (device->top_module_graph && !device->top_module_graph->empty()) {
    device->restartGraph();
  }
  device->reset_status();
  if (device->config.clock_continuity) {
    device->status.device_time = prev_time;
    device->status.low_time = prev_time;
    device->status.high_time = prev_time;
  }
  ```
- In `eval/test_deepseekv3_mixtral_style.cpp`:
  ```cpp
  time_ns step_start_dev_time = device->status.device_time;
  cluster->run(metadata);
  time_ns step_device_time = device->status.device_time - step_start_dev_time;
  if (system_config.clock_continuity) {
    cumulative_device_time = device->status.device_time;
  } else {
    cumulative_device_time += step_device_time;
  }
  stat.latency = step_device_time;
  stat.time = cumulative_device_time;
  ```
- In `Cluster::setStat(Stat &stat)`:
  Protected `stat.latency` from being overwritten with monotonic cumulative time when `clock_continuity` is active.

---

## 3. Verification of Zero Ternary Operators

In accordance with strict project constraints, no ternary operators (`? :` in C++, `ast.IfExp` in Python) were introduced:
- Verified via AST parser in [`scripts/final_campaign/run_phase2f_sensitivity.py`](file:///home/mohman/ca_project/scripts/final_campaign/run_phase2f_sensitivity.py): 0 ternaries found.
- Verified via AST parser in [`scripts/final_campaign/generate_phase2f_plots.py`](file:///home/mohman/ca_project/scripts/final_campaign/generate_phase2f_plots.py): 0 ternaries found.
- Verified via `grep -rn "?"` across modified C++ source files: 0 ternary operators found.
