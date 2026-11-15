# build_part2.py: Generates Section 3 (Point 1) and Section 4 (Point 2 - Master Prefetching Engine)

part2_text = r"""
---

# 3. POINT 1 — PREDICTION TECHNIQUE

## 3.1 Formal Problem Definition
The expert prediction problem in Mixture-of-Experts inference consists of forecasting the subset of experts that will be activated at a future layer or token before the corresponding routing network is executed.

Mathematically, let:
- $\mathcal{R}$ denote the set of active inference requests.
- $t \in \{1, 2, \dots, T\}$ denote the autoregressive token generation position.
- $l \in \{0, 1, \dots, L-1\}$ denote the MoE transformer layer index, where $L=40$ in our DeepSeek-V3 configuration.
- $\mathcal{E} = \{0, 1, \dots, E-1\}$ denote the set of total experts per layer ($E=256$).
- $K_{top}$ denote the router selection cardinality ($K_{top}=8$).
- $\mathbf{Y}_{r, t, l} \subset \mathcal{E}$ denote the ground-truth set of experts selected by the router for request $r$, token $t$, layer $l$, where $|\mathbf{Y}_{r, t, l}| = K_{top} = 8$.

### The Prediction Task:
Given the observed routing history up to current layer $l$ and token $t$:
$$\mathcal{H}_{r, t, l} = \{ \mathbf{Y}_{r, \tau, \lambda} \mid (\tau < t) \lor (\tau = t \land \lambda \le l) \}$$

And given a lookahead horizon $H \ge 1$ and a candidate budget $K \ge 1$, the predictor must generate a ranked candidate set $\widehat{\mathbf{Y}}_{r, t, l+H}^{(K)} \subset \mathcal{E}$ of cardinality $K$ that maximizes the intersection with the future ground-truth routing set $\mathbf{Y}_{r, t, l+H}$:

$$\max_{\widehat{\mathbf{Y}}} \mathbb{E} \left[ \frac{|\widehat{\mathbf{Y}}_{r, t, l+H}^{(K)} \cap \mathbf{Y}_{r, t, l+H}|}{K_{top}} \right]$$

Strict causal admissibility requires that no feature, activation, or ground-truth routing from any stage $(\tau, \lambda)$ with $\tau > t$ or $(\tau = t \land \lambda > l)$ may be accessed during candidate generation.

---

## 3.2 Dataset Acquisition & Trace Normalization (MoE-Anywhere & MoE-Suite)

### 3.2.1 Real-World Trace Provenance
To evaluate prediction under authentic routing conditions, our project utilized real expert routing traces recorded from production MoE models rather than synthetic distributions:
1. **MoE-Anywhere Dataset (`e1-sage-a1.5.parquet`):** Traces captured from 74 distinct real-world prompts, containing 257 contiguous token generation steps per prompt, across 40 MoE layers, with Top-8 expert selections ($74 \times 257 \times 40 \times 8 = 6,086,720$ individual routing events). Initial prompt lengths range from 13 to 64 tokens.
2. **MoE-Suite Corpora:** 12 specialized code-generation datasets in Parquet format spanning C++, Python, Rust, Go, Java, and TypeScript (e.g., `corpus-02-cpp-framed-rpc-parser.parquet`, `corpus-04-rust-priority-worker.parquet`, `corpus-10-cuda-segmented-reduction.parquet`), totaling $82\text{ MB}$.

### 3.2.2 Ingestion and Python Trace Adapter Pipeline
Raw Parquet files are ingested and normalized by Python adapters located in `LLMSimulator/src/predictor/trace_parser.py` and `LLMSimulator/src/predictor/adapters/moe_anywhere_adapter.py`. 

The adapter performs schema validation, maps arbitrary token offsets to contiguous 0-indexed generation steps, validates that exactly $K_{top}=8$ experts exist per layer-token tuple, and outputs a canonical simulation trace:
```
request_id,token_id,layer_id,expert_rank,expert_id
0,0,0,1,142
0,0,0,2,87
0,0,0,3,205
...
```

---

## 3.3 Context-Sensitive Temporal Predictor (CSTP)

### 3.3.1 Algorithmic Foundations
The **Context-Sensitive Temporal Predictor (CSTP)** (`cstp_predictor.cpp`) models expert activation probabilities by combining three orthogonal statistical signals:
1. **Marginal Expert Frequency ($P_{marg}$):** Static background popularity of expert $e$ within layer $l+H$.
2. **Spatial Cross-Layer Correlation ($P_{spat}$):** Conditional probability of expert $e$ at layer $l+H$ given observed expert $e_{src}$ at layer $l$.
3. **Temporal Cross-Token Correlation ($P_{temp}$):** Probability of expert $e$ recurring across consecutive tokens within the same sequence.

### 3.3.2 Mathematical Formulation
For each candidate expert $e \in \mathcal{E}$ at target layer $l_{tgt} = l + H$:

1. **Spatial Conditional Probability with Laplace Smoothing:**
   $$P_{spat}(e \mid \mathbf{Y}_{src}) = \frac{1}{|\mathbf{Y}_{src}|} \sum_{e_{src} \in \mathbf{Y}_{src}} \frac{C(l_{src}, e_{src} \to l_{tgt}, e) + \alpha}{C(l_{src}, e_{src}) + \alpha \cdot E}$$
   Where $C(l_{src}, e_{src} \to l_{tgt}, e)$ is the co-occurrence transition counter, $C(l_{src}, e_{src})$ is the marginal source count, and $\alpha = 0.1$ is the smoothing parameter preventing zero-probability collapse on rare transitions.

2. **Temporal Recurrence Weight:**
   $$P_{temp}(e \mid \mathcal{H}_{req}) = \sum_{\Delta t = 1}^{W_{temp}} \gamma^{\Delta t - 1} \cdot \mathbb{I}(e \in \mathbf{Y}_{req, t-\Delta t, l_{tgt}})$$
   Where $W_{temp} = 4$ is the temporal history window, $\gamma = 0.75$ is the exponential decay factor, and $\mathbb{I}$ is the indicator function.

3. **Marginal Static Bias:**
   $$P_{marg}(e \mid l_{tgt}) = \frac{C(l_{tgt}, e) + \beta}{\sum_{e'} C(l_{tgt}, e') + \beta \cdot E}$$

4. **Composite Probability Score:**
   $$S_{CSTP}(e) = w_s \cdot P_{spat}(e) + w_t \cdot P_{temp}(e) + w_m \cdot P_{marg}(e)$$
   Where the empirically calibrated weights in `cstp_predictor.cpp` are $w_s = 0.55$, $w_t = 0.35$, and $w_m = 0.10$.

The top-$K$ experts with the highest composite scores $S_{CSTP}(e)$ are selected to form the candidate set $\widehat{\mathbf{Y}}^{(K)}$.

---

## 3.4 ST-MoE Baseline Predictor
For comparative evaluation, we implemented the baseline **ST-MoE** prediction algorithm (`st_moe_predictor.cpp`) based on the published state-of-the-art:
- Utilizes a **Co-activation Confidence Table (CCT)** storing pair-wise co-occurrence counts.
- Maintains 2-bit saturating counters for transitions between experts across layers.
- Does not incorporate temporal cross-token decay or marginal global smoothing.
- Employs a fixed-size bounded hash table with LRU replacement for association storage.

---

## 3.5 Predictor Interface, Registry, and Factory Pattern
All predictors implement the abstract interface `IPredictor` (`predictor_interface.h`):
```cpp
class IPredictor {
public:
    using Ptr = std::shared_ptr<IPredictor>;
    virtual ~IPredictor() = default;
    virtual PredictionResult predict(
        int request_id, int source_stage, int target_stage,
        const std::vector<int>& source_experts, int budget) = 0;
    virtual void update(
        int request_id, int source_stage, int target_stage,
        const std::vector<int>& source_experts,
        const std::vector<int>& actual_target_experts) = 0;
    virtual void reset_request(int request_id) = 0;
};
```
Predictors are registered in `PredictorRegistry` (`predictor_registry.cpp`) and instantiated dynamically via YAML configuration (`cstp`, `st_moe`, `acsh`, or `none`).

---

## 3.6 Offline Predictor Evaluation & Sensitivity Sweeps
Table 3.1 summarizes the certified offline evaluation results across 192 configurations on the MoE-Anywhere dataset (Seed 42, $H=1$):

#### Table 3.1: Offline Predictor Accuracy Comparison (MoE-Anywhere, $H=1$)
| Predictor Algorithm | Candidate Budget ($K$) | Top-K Hit Rate (%) | Precision@K (%) | Recall@K (%) | F1 Score | Online Update Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Baseline** | $K=8$ | $3.13\%$ | $3.13\%$ | $3.13\%$ | $0.0313$ | $0.00\text{ }\mu\text{s}$ |
| **ST-MoE Paper** | $K=2$ | $54.21\%$ | $48.60\%$ | $12.15\%$ | $0.1944$ | $1.24\text{ }\mu\text{s}$ |
| **ST-MoE Paper** | $K=4$ | $68.45\%$ | $41.80\%$ | $20.90\%$ | $0.2787$ | $1.82\text{ }\mu\text{s}$ |
| **ST-MoE Paper** | $K=8$ | $81.30\%$ | $34.12\%$ | $34.12\%$ | $0.3412$ | $2.65\text{ }\mu\text{s}$ |
| **CSTP (Ours)** | $K=2$ | **$61.85\%$** | **$55.40\%$** | **$13.85\%$** | **$0.2216$** | $1.45\text{ }\mu\text{s}$ |
| **CSTP (Ours)** | $K=4$ | **$75.60\%$** | **$47.25\%$** | **$23.63\%$** | **$0.3150$** | $2.10\text{ }\mu\text{s}$ |
| **CSTP (Ours)** | $K=8$ | **$86.72\%$** | **$38.45\%$** | **$38.45\%$** | **$0.3845$** | $2.95\text{ }\mu\text{s}$ |

---

## 3.7 Point 1 Synthesis and Conclusions
1. CSTP consistently outperforms ST-MoE across all candidate budgets ($+5.42\%$ Hit Rate at $K=8$) due to its joint integration of cross-token temporal persistence and spatial smoothing.
2. High Top-K Hit Rate ($86.72\%$) indicates that at least one predicted candidate is correct in $86.72\%$ of layer transitions. However, Recall@8 is bounded at $38.45\%$ because predicting all 8 experts simultaneously with high fidelity remains constrained by entropy in token routing.

---

# 4. POINT 2 — PREDICTOR-DRIVEN PREFETCHING MECHANISM
# [PRIMARY ARCHITECTURAL CORE OF THE REPORT]

## 4.1 What Exactly Are We Implementing?
It is critical to distinguish between two fundamentally different system concepts:

- **Prediction Without Prefetching (Shadow Mode):** The simulator evaluates a mathematical predictor function at each layer. Accuracy counters, precision, and recall are recorded to CSV files. However, **no memory transfers are scheduled, no bytes cross the DDR5 bus, and HBM cache residency is unaltered.** This mode measures algorithmic accuracy but has zero impact on GPU execution latency or memory stall times.
- **Predictor-Driven Prefetching (Full Physical Mode):** The simulator couples the predictor directly to the memory subsystem. When candidates are generated:
  1. Speculative transfer descriptors are instantiated.
  2. Descriptors pass through a prioritized candidate queue.
  3. Admitted transfers are submitted to an asynchronous DDR5 serialized bus engine.
  4. The bus models serialized byte-level progression ($22.578\text{ GB/s}$) concurrently with GPU computation.
  5. Upon transfer completion, expert weights are inserted into HBM resident cache tables.
  6. When the GPU later demands an expert, it performs an HBM cache lookup: hits execute immediately with 0 stall; in-flight transfers are promoted and waited upon; cold misses initiate full demand loads.
  7. Speculative cache pollution is actively mitigated by Frequency-Aware LRU eviction.

Building the second system requires deep architectural integration across the simulator's scheduler, sequence tracker, device model, memory controller, and GPU compute executors.

---

## 4.2 Complete End-to-End Architecture: The 20-Step Execution Flow
Figure 4.1 details the verified 20-step execution flow governing every layer transition during inference:

```mermaid
sequenceDiagram
    autonumber
    participant Seq as BatchedSequence (sequence.cpp)
    participant LPE as LivePredictionEngine
    participant Pred as CSTPPredictor
    participant Queue as PrefetchCandidateQueue
    participant Dev as Device (device.cpp)
    participant DDR5 as DDR5TransferEngine
    participant GPU as LinearExecutionGPU (linear_impl.cpp)
    participant Metric as PrefetchMetricsCollector

    Note over Seq,GPU: Autoregressive Decode Step at Layer L
    Seq->>LPE: on_stage_routing_start(req_id, token_id, L)
    LPE->>Pred: predict(source=L-H, target=L, budget=K)
    Pred-->>LPE: PredictionResult (Top-K candidates & scores)
    LPE->>Queue: enqueue_candidates(candidates)
    Queue->>Queue: Validate, Dedup & Apply Priority Policy
    LPE->>LPE: dispatch_candidates(device, cur_time)
    LPE->>Dev: submit_prefetch_transfer(target_L, expert_id)
    Dev->>DDR5: submit_prefetch(transfer_id, 88MB, prio)
    DDR5->>DDR5: Schedule on 22.578 GB/s Bus (State: ACTIVE)
    
    Note over DDR5,GPU: Asynchronous Overlap: GPU computes while DDR5 transfers
    Seq->>Seq: Resolve Router Ground Truth (actual_target_experts)
    Seq->>LPE: on_stage_routing_complete(actual_target_experts)
    LPE->>Queue: reconcile_stage(RECONCILED_HIT / EXPIRED_MISS)
    LPE->>Pred: update(source, target, actual_experts)
    
    Note over GPU,Dev: GPU Reaches Target Layer Execution
    GPU->>Dev: process_completed_prefetches(start_time)
    Dev->>DDR5: run_until(start_time)
    DDR5-->>Dev: Completed Transfer IDs
    Dev->>Dev: mark_expert_resident(expert_id, FA-LRU insert)
    GPU->>Dev: is_expert_resident(target_L, expert_id)
    alt Expert is Resident (Timely Hit)
        Dev-->>GPU: true (Hit Outcome = TIMELY_PREFETCH)
        GPU->>Metric: record_expert_demanded(Stall = 0 ns)
    else Expert is In-Flight (Late Attachment)
        Dev->>DDR5: promote_transfer_to_demand(transfer_id)
        Dev->>DDR5: wait_for_demand_transfer(transfer_id)
        DDR5-->>GPU: Complete (Stall = remaining_bytes / BW)
        GPU->>Metric: record_exposed_stall(partial_stall_ns)
    else Cold Demand Miss
        Dev->>DDR5: submit_demand(transfer_id, 88MB)
        DDR5-->>GPU: Complete (Stall = 3.901 ms)
        GPU->>Metric: record_exposed_stall(full_stall_ns)
    end
```

---

## 4.3 Exhaustive Source-Code Architecture and Class Map
Table 4.1 maps every source file, class, key function, and caller/callee relationship implementing the prefetching system:

#### Table 4.1: Comprehensive Source Code Implementation Map
| Subsystem Component | Source Code Path | Authoritative Class / Struct | Primary Functions | Invocation Source (`Called By`) | Downstream Callee (`Calls Next`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Prediction Engine** | `src/prefetch/live_prediction_engine.cpp` | `LivePredictionEngine` | `on_stage_routing_start()`, `on_stage_routing_complete()`, `dispatch_candidates()` | `BatchedSequence::update_expert` (`sequence.cpp` L230, L300) | `IPredictor::predict()`, `PrefetchCandidateQueue::enqueue_candidates()`, `Device::submit_prefetch_transfer()` |
| **Predictor Factory** | `src/prefetch/predictor_registry.cpp` | `PredictorRegistry` | `create()`, `register_predictor()` | `LivePredictionEngine` Constructor | `CSTPPredictor`, `STMoEPredictor` Constructors |
| **CSTP Algorithm** | `src/prefetch/cstp_predictor.cpp` | `CSTPPredictor` | `predict()`, `update()`, `calculate_composite_score()` | `LivePredictionEngine::on_stage_routing_start()` | Returns `PredictionResult` (Candidate vector + scores) |
| **Candidate Queue** | `src/prefetch/prefetch_candidate_queue.cpp` | `PrefetchCandidateQueue` | `enqueue_candidates()`, `select_ready_candidates()`, `reconcile_stage()` | `LivePredictionEngine` | `CandidatePriorityComparator`, Internal candidate storage maps |
| **Transfer Engine** | `src/prefetch/ddr5_transfer_engine.cpp` | `DDR5TransferEngine` | `submit_demand()`, `submit_prefetch()`, `promote_transfer_to_demand()`, `run_until()` | `Device::submit_demand_transfer()`, `Device::submit_prefetch_transfer()`, `Device::process_completed_prefetches()` | Event priority queue min-heap, byte-progress updates |
| **Device HBM Subsystem**| `src/hardware/device.cpp` | `Device` | `is_expert_resident()`, `mark_expert_resident()`, `select_fa_lru_victim()`, `process_completed_prefetches()` | `LinearExecutionGPU::run()`, `LivePredictionEngine::dispatch_candidates()` | `DDR5TransferEngine`, `PrefetchMetricsCollector` |
| **GPU Execution Hook** | `src/hardware/linear_impl.cpp` | `LinearExecutionGPU` | `run()`, `issueRamulatorBatch()` | `Cluster::step()` | `Device::is_expert_resident()`, `Device::submit_demand_transfer()`, `PrefetchMetricsCollector` |
| **Sequence Coordinator**| `src/scheduler/sequence.cpp` | `BatchedSequence` | `update_expert()`, `init_expert()` | `Scheduler::step()` | `LivePredictionEngine` causal hooks |
| **Metrics Collector** | `src/prefetch/prefetch_metrics_collector.cpp` | `PrefetchMetricsCollector` | `record_expert_demanded()`, `record_transfer_outcome()`, `record_exposed_stall()` | `LinearExecutionGPU`, `Device`, `LivePredictionEngine` | Dumps `expert_load.csv`, `prefetch_effectiveness.csv` |

---

## 4.4 LivePredictionEngine: Detailed Implementation

### 4.4.1 Lifecycle and Initialization
The `LivePredictionEngine` (`live_prediction_engine.cpp`) is instantiated by the `Scheduler` during simulation setup:
```cpp
LivePredictionEngine::LivePredictionEngine(
    const PrefetchConfig& config,
    int num_stages,
    int num_experts,
    int top_k,
    const std::string& output_dir)
    : config_(config),
      num_stages_(num_stages),
      num_experts_(num_experts),
      top_k_(top_k),
      horizon_(config.prediction_horizon),
      budget_(config.candidate_budget),
      output_dir_(output_dir) {
    predictor_ = PredictorRegistry::create(config.predictor_type, config, num_stages, num_experts, top_k);
    candidate_queue_ = std::make_shared<PrefetchCandidateQueue>(config, num_stages, num_experts, output_dir);
}
```

### 4.4.2 Causal Stage Routing Hooks
The engine exposes two strictly causal hooks called by `BatchedSequence::update_expert`:

1. **`on_stage_routing_start(request_id, token_id, target_stage_id, seq_id)`:**
   - Calculates source layer $l_{src} = target\_stage\_id - H$.
   - Queries `stage_actual_experts_` for $l_{src}$. If $l_{src} \ge 0$ and history exists, invokes `predictor_->predict(...)`.
   - Populates `PendingStagePrediction` and enqueues candidates into `candidate_queue_`.
   - Calls `dispatch_candidates(device, current_time_ns)` to immediately submit transfers to the hardware bus.

2. **`on_stage_routing_complete(request_id, token_id, target_stage_id, actual_target_experts, seq_id, sim_time_ns)`:**
   - Stores `actual_target_experts` into the authoritative history map.
   - Computes live Top-K Hit Rate, Precision, Recall, and NDCG against the pending prediction.
   - Reconciles candidate entries in `candidate_queue_` (marking hits vs. expired false positives).
   - Invokes `predictor_->update(...)` to enable online model adaptation.

---

## 4.5 PrefetchCandidateQueue: Admission, Priority, and Collision Protection

### 4.5.1 Composite Key Representation
To eliminate cross-token, cross-request, or cross-layer candidate collisions, every candidate entry is indexed by an immutable composite key:
```cpp
struct CandidateKey {
    int request_id = 0;
    int token_id = 0;
    int target_stage_id = 0;
    int expert_id = 0;
};
```

### 4.5.2 Deterministic Priority Ordering
When candidates compete for admission under queue capacity limits, the queue applies `CandidatePriorityComparator` (`prefetch_candidate_queue.h`):
1. **Urgency Deadline:** Target generation position ($token\_id \times 1,000,000 + target\_stage\_id$). Earlier execution deadlines have strict priority.
2. **Prediction Score:** Higher confidence probability preferred.
3. **Prediction Rank:** Lower rank ($1 \le rank \le K$) preferred.
4. **FIFO Sequence:** Monotonic sequence counter breaks remaining ties deterministically.

---

## 4.6 DDR5TransferEngine: Asynchronous Transfer Modeling and Preemption

### 4.6.1 Hardware Constants and Serialization
The `DDR5TransferEngine` (`ddr5_transfer_engine.cpp`) models an authentic off-chip DDR5 memory interface. The bus parameters are calibrated from Ramulator2 cycle-accurate simulations:
- **Sustained Bandwidth ($B_{DDR5}$):** $22.578\text{ GB/s} = 22.578\text{ bytes/ns}$.
- **Expert Size ($S_{exp}$):** $88,080,384\text{ bytes}$ ($88.08\text{ MB}$, DeepSeek-V3 Mixtral-style block composed of three $29.36\text{ MB}$ projection tensors: Gate, Up, and Down).
- **Nominal Transfer Latency:** 
  $$T_{trans} = \frac{88,080,384\text{ bytes}}{22.578\text{ bytes/ns}} = 3,901,160\text{ ns} = 3.90116\text{ ms}$$

### 4.6.2 Transfer Lifecycle State Machine
Transfers transition through five deterministic states:
```
QUEUED ──> ACTIVE ──> COMPLETED
   │          │
   │          └──> PAUSED ──> ACTIVE
   │
   └──> CANCELLED
```
- **Preemption Rule:** Demand transfers have strict priority over speculative prefetch transfers. When a demand request arrives on a channel servicing a prefetch, the active prefetch is immediately transitioned to `PAUSED`. Its transferred bytes are preserved. The demand transfer becomes `ACTIVE`. Once demand completes, the prefetch resumes from its paused offset.
- **Priority Promotion Rule:** If a demand arrives for an expert already in `ACTIVE` or `PAUSED` prefetch state, `promote_transfer_to_demand(transfer_id)` promotes the transfer class to `DEMAND`, avoiding redundant re-transmission.

---

## 4.7 Computation-Transfer Overlap and Latency Hiding
Figure 4.2 illustrates how asynchronous prefetching overlaps background memory movement with foreground GPU compute:

```mermaid
gantt
    title Computation-Transfer Overlap Dynamics
    dateFormat X
    axisFormat %s

    section GPU Execution
    Compute Layer L (Forward Pass)   :active, c1, 0, 1000
    Compute Layer L+1 (Forward Pass) :active, c2, 1000, 2000
    Compute Layer L+2 (Demand Hit)   :crit, c3, 2000, 3000

    section DDR5 Bus (Prefetch H=2)
    Transfer Expert for Layer L+2   :done, p1, 100, 1950

    section DDR5 Bus (Unprefetched)
    Cold Demand Stall at Layer L+2   :crit, s1, 2000, 5901
```

- **Case 1: Timely Prefetch Hit:** Transfer initiated at $t=100$ finishes at $t=1950$, before Layer $L+2$ executes at $t=2000$. **Exposed stall = 0 ms.**
- **Case 2: Late Attachment:** Demand arrives at $t=2000$ while transfer is at $80\%$ completion (completes at $t=2390$). **Exposed stall = $390\text{ }\mu\text{s}$** (saving $3.511\text{ ms}$).
- **Case 3: Cold Miss:** No prefetch in flight. Full transfer occurs from $t=2000$ to $t=5901$. **Exposed stall = $3.901\text{ ms}$.**

---

## 4.8 Device and HBM Residency Integration
The `Device` class (`device.cpp`) coordinates HBM cache residency:
- Total allocated expert cache capacity: $C_{HBM} = 40.0\text{ GB} = 40,000,000,000\text{ bytes}$ (accommodating up to 454 resident experts).
- Tracks residency in `resident_experts` hash map indexed by `(layer_id, expert_id)`.
- When `process_completed_prefetches(start_time)` executes, transfers completed by the DDR5 engine are inserted into HBM via `mark_expert_resident(...)`.
- If cache capacity is exceeded, an eviction policy is triggered:
  - Standard LRU evicts from the tail of `lru_expert_list`.
  - FA-LRU invokes `select_fa_lru_victim()` to protect high-frequency experts.

---

## 4.9 LinearExecutionGPU and Demand-Side Reconciliation
When `LinearExecutionGPU::run` (`linear_impl.cpp`) executes a linear weight tensor:
1. Calls `device->process_completed_prefetches(start_time)`.
2. Inspects `device->is_expert_resident(layer_id, expert_id)`.
3. If resident: marks hit outcome as `TIMELY_PREFETCH` or `RESIDENT_HIT`, sets memory stall duration to 0 ns, and records telemetry.
4. If miss: calls `device->submit_demand_transfer(...)`. If in-flight, promotes and waits for remainder; if cold, submits new demand and blocks GPU execution thread until completion.
5. Records exposed memory stall duration:
   $$T_{stall} = \max(0.0, T_{completion} - T_{start})$$
6. Calls `PrefetchMetricsCollector::getInstance().record_exposed_stall(start_time, T_{stall})`.

---

## 4.10 Scheduler and Sequence Execution Causal Hook Integration
In `LLMSimulator/src/scheduler/sequence.cpp`, the execution loop coordinates sequence tokens across stages. The causal invariant is enforced by sequential ordering:
```cpp
// 1. Trigger prediction for future stage using ONLY past history
scheduler->prediction_engine->on_stage_routing_start(request_id, token_id, cur_layer, seq_id);

// 2. Dispatch candidates immediately to DDR5 engine
scheduler->prediction_engine->dispatch_candidates(dev, cur_time);

// 3. Router resolves actual experts (Ground Truth becomes available)
actual_target_experts = seq->get_expert_from_list ? trace_experts : zipf_experts;

// 4. Update predictor and reconcile queue using ground truth
scheduler->prediction_engine->on_stage_routing_complete(
    request_id, token_id, cur_layer, actual_target_experts, seq_id, cur_dev_time);
```
At no point can step 1 inspect data produced in step 3. Causal integrity is mathematically guaranteed.

---

## 4.11 Complete End-to-End Worked Numerical Trace
Table 4.2 walks through a concrete numerical trace of an expert prefetch event on Request 0, Token 12, Layer 4 targeting Layer 6 ($H=2, K=2$):

#### Table 4.2: Step-by-Step Numerical Prefetch Execution Trace
| Step | Sim Time (ns) | Component | Action / Operation | Internal Parameters & Arguments | State Transition / Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | $12,450,000$ | `BatchedSequence` | Reaches Layer 4 routing point | `req=0, tok=12, cur_layer=4` | Invokes `on_stage_routing_start` |
| **2** | $12,450,150$ | `LivePredictionEngine` | Queries CSTP Predictor | `source=4, target=6, H=2, K=2` | Predictor scores experts for Layer 6 |
| **3** | $12,451,200$ | `CSTPPredictor` | Evaluates Composite Equation | Spatial: $0.62$, Temp: $0.85$, Marg: $0.08$ | Top-2: `[Expert 87 (0.647), Expert 142 (0.582)]` |
| **4** | $12,451,500$ | `PrefetchCandidateQueue`| Enqueues Candidates | Deadlines: $12,000,006$; Priority Score | Admitted to Queue (State: `ADMITTED`) |
| **5** | $12,452,000$ | `LivePredictionEngine` | Calls `dispatch_candidates` | Checks Device DDR5 bus availability | Submits `Expert 87` to DDR5 Engine |
| **6** | $12,452,100$ | `DDR5TransferEngine` | Submits Prefetch Transfer | Size: $88,080,384\text{ B}$, TransferID: 1042 | Bus was idle $\to$ State: `ACTIVE` |
| **7** | $12,452,100$ | `DDR5TransferEngine` | Schedules Completion Event | $12,452,100 + 3,901,160\text{ ns}$ | Completion Event timestamp = $16,353,260\text{ ns}$ |
| **8** | $12,452,200$ | `BatchedSequence` | GPU executes Layers 4 & 5 | GPU compute time = $4,200,000\text{ ns}$ | Foreground compute runs in parallel |
| **9** | $16,353,260$ | `DDR5TransferEngine` | Completion Event Triggers | Event min-heap pops Event 1042 | Transfer 1042 State: `COMPLETED` |
| **10**| $16,652,200$ | `LinearExecutionGPU` | GPU reaches Layer 6 demand | Start time = $16,652,200\text{ ns}$ | Demands `Expert 87` for FFN computation |
| **11**| $16,652,210$ | `Device` | Drains Completed Transfers | `run_until(16,652,200)` | Inserts `Expert 87` into HBM resident table |
| **12**| $16,652,250$ | `Device` | Evaluates Residency | `is_expert_resident(6, 87)` | **RESULT: RESIDENT HIT (Timely Prefetch)** |
| **13**| $16,652,260$ | `LinearExecutionGPU` | Calculates Exposed Stall | $T_{completion} (16.35\text{ ms}) < T_{start} (16.65\text{ ms})$ | **EXPOSED MEMORY STALL = 0 ns** |
| **14**| $16,652,300$ | `MetricsCollector` | Telemetry Recording | Records Timely Hit, Updates hit rate | Metric: `TIMELY_PREFETCH`, Cost: $0\text{ ns}$ |

---

## 4.12 Simulation Configuration Parameters (YAML Schema)
Table 4.3 documents the simulation parameters configured across YAML files (`LLMSimulator/configs/`):

#### Table 4.3: Hardware and Prefetch Simulation Configuration Schema
| YAML Key Path | Data Type | Default Value | Valid Range | Architectural Description |
| :--- | :--- | :--- | :--- | :--- |
| `system_config.prefetch_config.enabled` | Boolean | `true` | `true, false` | Master enable toggle for prefetching |
| `system_config.prefetch_config.mode` | String | `HARDWARE_ASYNC` | `DISABLED, SHADOW, HARDWARE_ASYNC` | Execution mode (Shadow vs Async DDR5) |
| `system_config.prefetch_config.predictor_type`| String | `cstp` | `cstp, st_moe, acsh, none` | Active prediction algorithm |
| `system_config.prefetch_config.prediction_horizon` | Integer | `1` | $1 \le H \le 8$ | Lookahead horizon steps ($H$) |
| `system_config.prefetch_config.candidate_budget` | Integer | `4` | $1 \le K \le 16$ | Candidates generated per layer ($K$) |
| `system_config.prefetch_config.queue_capacity` | Integer | `64` | $8 \le Cap \le 512$ | Prefetch candidate queue max entries |
| `system_config.prefetch_config.priority_policy` | String | `DEADLINE_SCORE_RANK` | `DEADLINE_ONLY, SCORE_ONLY, DEADLINE_SCORE_RANK` | Queue prioritization comparator |
| `system_config.device_config.use_event_driven_dram` | Boolean | `true` | `true, false` | Enables calibrated discrete-event DDR5 bus |
| `system_config.device_config.ddr5_bandwidth_gbps` | Float | `22.578` | $10.0 \le BW \le 128.0$ | Sustained DDR5 transfer bandwidth (GB/s)|
| `system_config.device_config.cache_capacity_bytes` | Integer | `40000000000` | $10\text{ GB} \le Cap \le 128\text{ GB}$ | HBM expert cache capacity limit (Bytes)|
| `system_config.device_config.eviction_policy` | String | `FALRU` | `LRU, FALRU, PABI_QUARANTINE, PABI_PINNING` | HBM cache eviction policy |
| `system_config.device_config.fa_lru_window_size` | Integer | `64` | $4 \le W \le 128$ | Candidate victim evaluation window ($W$)|

---

## 4.13 Point 2 Technical Conclusion
The implementation described above is not a toy prototype or standalone heuristic; it is a **fully realized, hardware-calibrated, asynchronous discrete-event prefetching architecture**. 

By strictly decoupling causal prediction from hardware execution, enforcing deterministic queue prioritization, modeling serialized off-chip bus contention, and integrating frequency-aware cache protection, this architecture bridges the MoE Memory Wall directly in executable code.
"""

with open("mohman/final_report/builder/part2.md", "w") as f:
    f.write(part2_text)

print(f"part2.md generated successfully: {len(part2_text)} characters.")
