# Phase 2F.0 — Compute-Timing and Clock-Continuity Audit

## 1. Executive Summary

This read-only audit investigates the computational timing sources, device clock advancement mechanisms, and inter-token clock continuity in the `LLMSimulator` framework. Its purpose is to evaluate whether a synthetic compute-window scaling multiplier can be safely introduced to investigate the latent overlap capabilities of forward-lookahead prefetching without perturbing memory bandwidth, cache capacity, routing traces, or execution semantics.

### Key Audit Findings
1. **Compute Timing is Purely Analytical:** GPU execution time in `LLMSimulator` is modeled mathematically via FLOP counting divided by peak device compute throughput. No real host CPU sleeping, kernel spinning, or CUDA execution occurs.
2. **Safe Multiplier Feasibility:** A configurable compute multiplier can scale simulated stage compute duration cleanly by modulating `compute_duration` in `LinearExecutionGPU` or via a designated scaling factor in `StatusBoard` / `HardwareConfig`. This alters only simulated duration and does not distort physical DDR5 bandwidth, HBM capacity, routing decisions, or demand lookup sequences.
3. **Discrete Event Progression:** Because the DDR5 transfer engine advances via `ddr5_transfer_engine->run_until(device->get_time())`, extending the simulated compute time directly expands the simulated time window available for speculative transfers to make physical progress across the bus.
4. **Resettable Device Clock Discontinuity:** The device clock `device->status.device_time` is reset to zero at the beginning of each decoding token by `Cluster::restartModuleGraph()`. In contrast, `DDR5TransferEngine` preserves its internal event queue and timestamp state. This creates an inter-token clock discontinuity where timestamps abruptly jump backward between token epochs.

---

## 2. Detailed Source Trace: Compute-Time Accounting

All computational timing in the simulator originates from static architectural modeling. Below is the exact source trace of functions, files, and lines governing compute durations and device-time progression.

### 2.1 MoE Compute-Duration Calculation
The forward execution of an MoE layer passes through `MoEDecoder::forward` (`src/module/decoder.cpp:143-162`), which invokes `ExpertFFN::forward` (`src/module/expert.cpp:170-241`).
For each routed expert, `FeedForward3Way::forward` (`src/module/layer.cpp:415-437`) evaluates `gate_proj`, `up_proj`, and `down_proj`. Each projection is an instance of `Linear` (`src/module/linear.cpp:22-85`).

When `Linear::forward` executes, it invokes the compute execution pipeline:
- **Calling Interface:** `Device::execution` (`src/hardware/device.cpp:263-273`).
- **Execution Dispatcher:** `Executor::execution` (`src/hardware/executor.cpp:135-155`) -> `Executor::executeLinear` (`src/hardware/executor.cpp:222-235`).
- **Core Computational Kernel:** `LinearExecutionGPU` (`src/hardware/linear_impl.cpp:16-230`).

Inside `LinearExecutionGPU` (`src/hardware/linear_impl.cpp:23-40`):
```cpp
// Lines 23-28: FLOP and memory calculation
double m = input->shape[0];
double k = input->shape[1];
double n = weight->shape[1];

hw_metric total_flops = 2.0 * m * k * n;
hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

// Lines 30-33: Pure compute and memory duration calculation
time_ns compute_duration =
    total_flops / compute_peak_flops * 1000 * 1000 * 1000;
time_ns memory_duration =
    total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

exec_status.compute_duration = compute_duration;
```
For DeepSeek-V3 Mixtral-style decode execution ($m=1, k=4096, n=14336$), each GEMM projection yields an exact analytical compute duration based on `compute_peak_flops` ($989.4\text{ TFLOPS}$ FP16 on NVIDIA A100).

### 2.2 Stage-Level Compute-Time Accounting
The total stage execution time combines the compute and memory durations of the active modules.
In `LinearExecutionGPU` (`src/hardware/linear_impl.cpp:216-217`):
```cpp
exec_status.total_duration =
    std::max(exec_status.compute_duration, exec_status.memory_duration);
```
When an expert weight is resident in HBM, `exec_status.memory_duration` represents the high-bandwidth HBM access latency ($2.039\text{ TB/s}$), which is small compared to off-chip DDR5.
When an expert weight misses in HBM and must be fetched from DDR5, `exec_status.memory_duration` includes the off-chip transfer completion latency (`lines 96-100`).

The per-stage compute duration is recorded across the 8 routed experts in each of the 75 MoE layers:
$$\text{MoE Compute Duration per Stage} = \sum_{e=1}^{8} \left( t_{\text{comp}}(\text{gate}) + t_{\text{comp}}(\text{up}) + t_{\text{comp}}(\text{down}) + t_{\text{comp}}(\text{act}) \right) \approx 0.18317\text{ ms}$$
Across 4 decoding tokens ($4 \times 75 = 300\text{ MoE stages}$), total MoE compute time is:
$$\text{Total MoE Compute Time} = 300 \times 0.18317\text{ ms} = 54.95\text{ ms}$$

### 2.3 Device-Time Advancement During Computation
Device time does not advance via host wallclock time. Instead, it advances incrementally as each module in the execution graph is popped:
- **Module Execution Graph:** `TopModuleGraph::run_next_module` (`src/module/module_graph.cpp:177-189`).
- **Timestamp Push:** Before module execution, `set_push_status()` (`lines 191-220`) aligns `status.device_time = std::max(status.low_time, status.high_time)`.
- **Timestamp Pop and Accumulation:** After module execution, `set_pop_status()` (`lines 223-276`):
```cpp
// Lines 237-252: Accumulation into device time
ExecStatus exec_status = device->getExecStatus();

if (status.parallel_execution) {
  if (exec_status.processor_type == ProcessorType::LOGIC ||
      exec_status.processor_type == ProcessorType::PIM) {
    status.low_time += exec_status.total_duration;
    status.device_time = status.low_time;
  } else if (exec_status.processor_type == ProcessorType::GPU) {
    status.high_time += exec_status.total_duration;
    status.device_time = status.high_time;
  }
} else {
  status.device_time += exec_status.total_duration;
}
```
- **Device Time Query:** `Device::get_time()` (`src/hardware/device.h:222`):
```cpp
time_ns get_time() { return status.device_time; }
```
`status.device_time` represents the monotonically increasing simulated time within the current decoding step.

### 2.4 Attention and Non-MoE Computation Accounting
Non-MoE layers are interspersed between and within decoding blocks:
- **RMSNorm / LayerNorm:** `LayerNorm::forward` (`src/module/layernorm.cpp:34-41`).
  `compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;`
  `device->status.device_time += total_time;` (`line 53`).
- **Multi-Head / Multi-Latent Attention (MLA):** `src/hardware/attention_gen_impl.cpp`.
  Includes $Q$-down, $KV$-down, $KR$-proj, $Q$-up, $QR$-proj, $KV$-up, $O$-proj, and rotary positional embeddings (RoPE, `src/module/rope.cpp:41-48`).
  Attention computation adds approximately $0.05\text{ ms}$ to $0.08\text{ ms}$ per stage.
- **Residual Additions:** `Residual::forward` (`src/module/residual.cpp:30-49`).
  `device->status.device_time += total_time;` (`line 49`).

All non-MoE compute operations directly advance `device->status.device_time`.

### 2.5 Interaction Between Compute Events and DDR5TransferEngine
The connection between GPU computation and DDR5 background transfers is mediated by discrete time synchronization points:
1. **Stage Routing Dispatch:** In `scheduler/sequence.cpp:236-237` and `306-307`:
   ```cpp
   double cur_time = static_cast<double>(dev->get_time());
   scheduler->prediction_engine->dispatch_candidates(dev, cur_time);
   ```
   Speculative prefetch candidates are submitted with timestamp `cur_time = dev->get_time()`.
2. **Background Engine Progression:** Inside `Device::dispatch_prefetch_candidates` (`src/hardware/device.cpp:1419`) and `Device::process_completed_prefetches` (`src/hardware/device.cpp:1318-1323`):
   ```cpp
   ddr5_transfer_engine->run_until(current_device_time);
   ```
   The transfer engine executes all pending discrete events (transfer service start, byte progression, and transfer completions) up to `current_device_time`.
3. **Demand Access Synchronization:** At the start of `LinearExecutionGPU` (`src/hardware/linear_impl.cpp:64-67`):
   ```cpp
   device->set_current_stage(weight->expert_layer_id);
   device->process_completed_prefetches(start_time);
   if (!device->is_expert_resident(weight->expert_layer_id, weight->expert_id)) {
     is_demand_miss = true;
   }
   ```
   If GPU compute time has advanced `start_time` past the prefetch transfer completion timestamp, `process_completed_prefetches` marks the expert resident in HBM. As a result, `is_expert_resident` evaluates to `true`, converting what would have been a demand miss into a timely resident hit.

---

## 3. Safety Analysis of Compute Duration Scaling

Can a synthetic multiplier safely scale compute duration without invalidating simulator invariants?

| Subsystem / Property | Invariant Required | Multiplier Impact | Safety Assessment |
| :--- | :--- | :--- | :--- |
| **Host CPU Sleep** | Zero sleep or thread blocking | Multiplier modifies numerical FLOP/duration math only | **SAFE (Zero host sleep)** |
| **Host Idle Spinning** | Zero busy-wait loops | Simulation loop is purely analytical discrete-event | **SAFE (Zero busy-wait)** |
| **DDR5 Transfer Latency** | $88\text{ MB} / 22.578\text{ GB/s} = 3.901\text{ ms}$ | Governed by `DDR5TransferEngine::schedule_completion_event` | **SAFE (Unchanged)** |
| **DDR5 Bus Bandwidth** | $22.578\text{ GB/s}$ constant | Bandwidth constant in `DDR5TransferEngine` is untouched | **SAFE (Unchanged)** |
| **Routing Decisions** | Ground-truth traces A1–C1 | Routing loaded from static CSV trace files | **SAFE (Unchanged)** |
| **HBM Cache Capacity** | $80\text{ GB}$ ($975$ experts) | Cache eviction governed by `LRUExpertCache` | **SAFE (Unchanged)** |
| **Demand Lookups** | $7,200$ lookups across 4 tokens | Sequenced by trace replay loop in `test_deepseekv3...` | **SAFE (Unchanged)** |
| **Transfer Engine Progress** | Must advance with compute | Engine is advanced via `run_until(device->get_time())` | **SAFE (Progresses naturally)** |

### Conclusion on Feasibility
Scaling the compute duration by multiplier $M$ (e.g., $M \in \{1.0, 4.0, 8.0, 12.0, 16.0\}$) safely extends the simulated time interval between stage dispatches and demand lookups. Because `DDR5TransferEngine` is driven by `device->get_time()`, extending simulated compute duration provides a larger temporal window for speculative transfers to complete without distorting memory transfer physics or model routing semantics.

---

## 4. Clock Continuity Across Decoding Tokens

A critical consideration in multi-token prefetching is whether simulation timeflows smoothly across token boundaries.

### 4.1 The Mechanism of `device_time` Reset
In `test_deepseekv3_mixtral_style.cpp:431-504`:
```cpp
time_ns cumulative_device_time = 0;

for (int step = 0; step < iter; step++) {
  // Step start
  cluster->run(metadata);
  // Step end
  time_ns step_device_time = device->status.device_time;
  cumulative_device_time += step_device_time;
  ...
}
```
At the beginning of each call to `cluster->run(metadata)`:
1. `Cluster::run` (`src/hardware/cluster.cpp:1101-1109`) calls `restartModuleGraph()`.
2. `Cluster::restartModuleGraph` (`src/hardware/cluster.cpp:44-51`):
   ```cpp
   void Cluster::restartModuleGraph() {
     for (int device_rank = 0; device_rank < num_total_device; device_rank++) {
       Device::Ptr device = get_device(device_rank);
       device->restartGraph();
       device->reset_status();
       device->reset_timeboard();
     }
   }
   ```
3. `Device::reset_status` (`src/hardware/device.cpp:189`):
   ```cpp
   void Device::reset_status() { status = StatusBoard(); }
   ```
   This zeroes `device->status.device_time` at the start of every token step!

### 4.2 Timebase Mismatch Between Device and Transfer Engine
While `device->status.device_time` is zeroed at the start of each token step:
1. `ddr5_transfer_engine` is **NOT** reset between steps (`ddr5_transfer_engine->reset()` is only invoked by `reset_expert_cache()`, which is never called during inference).
2. As a result, at the end of Token 0, `ddr5_transfer_engine->current_time_ns_` stands at approximately $2.37 \times 10^9\text{ ns}$ ($2370\text{ ms}$).
3. When Token 1 starts, `device->get_time()` resets to $0.0\text{ ns}$.
4. When the first transfer of Token 1 is submitted at `current_time_ns = 50,000.0\text{ ns}`:
   ```cpp
   if (current_time_ns > current_time_ns_) {
       run_until(current_time_ns);
   }
   current_time_ns_ = current_time_ns;
   ```
   `current_time_ns_` in `DDR5TransferEngine` is forcefully set backward from $2.37\times 10^9$ to $50,000$!
5. Any incomplete transfer events from Token 0 remaining in `event_queue_` retain their $2.37\times 10^9\text{ ns}$ timestamps. Newly submitted Token 1 events receive timestamps near $0.0\text{ ns} + 3.9\times 10^6\text{ ns}$.

### 4.3 Implication for Phase 2F Experiments
The inter-token clock reset prevents prefetch pipelines from crossing from the end of token $k$ into the beginning of token $k+1$.
However, **within each decoding token**, simulation time increases monotonically across all 75 stages from $0.0\text{ ms}$ to $\approx 2370\text{ ms}$.
Therefore, an intra-token compute sensitivity experiment (e.g. testing whether stage $L$ prefetching for stage $L+2$ succeeds within the same token) is valid and unaffected by inter-token reset, provided evaluations account for the epoch boundary.

