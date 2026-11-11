#!/usr/bin/env python3
import os
import ast

WORKSPACE_DIR = "/home/mohman/ca_project"
SUBMISSION_DIR = os.path.join(WORKSPACE_DIR, "final_submission")
MASTER_DIR = os.path.join(SUBMISSION_DIR, "00_MASTER")

def check_no_ternary():
    with open(__file__, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=__file__)
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            raise AssertionError("Ternary operator found!")
    print("[PRE-FLIGHT] Zero ternary operators verified.")

def main():
    check_no_ternary()
    readme_path = os.path.join(MASTER_DIR, "ARCHITECTURE_EVOLUTION_AND_IMPLEMENTATION_README.md")

    content = """# Architecture Evolution and Implementation Manual
**MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference**

---

## 1. Original Simulator Architecture & Baseline Operation

### 1.1 Original Execution Pipeline
In the authoritative baseline commit (`d5bf61ff0c066c06f72173189f9924329f613a88`), the simulator executed Mixture-of-Experts inference as follows:
1. **Routing Stage (`src/module/route.cpp:RouteModule::run`):** Evaluated top-K routing tokens or replayed recorded routing indices from CSV traces.
2. **Expert Demand Lookup (`src/hardware/device.cpp:Device::check_expert_cache`):** Checked the unified HBM cache for required expert weights.
3. **Demand Fetch & Stall (`src/hardware/linear_impl.cpp:LinearExecutionGPU::run`):**
   - On a cache miss, stalled the GPU pipeline while fetching expert weights from off-chip DRAM.
   - DRAM request duration was computed via Ramulator2 cycle-accurate DRAM simulation or analytical peak bandwidth.
   - Total stage duration was evaluated as `total_duration = max(compute_duration, memory_duration)`.
4. **Cache Insertion (`src/hardware/device.cpp:Device::update_lru`):** The demanded expert was inserted into the unified LRU cache. If capacity was exceeded, the least recently used expert was evicted.

### 1.2 Original Baseline Limitations
- **Demand-Only Transfers:** Expert loading was purely reactive. The GPU remained completely stalled during memory fetches.
- **No Predictive Prefetching:** The simulator possessed no mechanisms for tracking inter-expert spatial co-activation or temporal transition probabilities.
- **Unified Cache Vulnerability:** Speculative transfers, if introduced naively, could displace verified demand experts, inducing severe cache pollution.
- **Coupled Execution Timing:** Compute and off-chip DDR5 miss latencies were entangled inside `max(compute, memory)`, preventing independent analysis of latency-hiding overlap.

---

## 2. New Prediction Architecture: ST-MoE & CSTP

### 2.1 ST-MoE Predictor Formulation
ST-MoE combines frequency-based historical prior with spatial co-activation tracking:
$$S_{\\text{ST-MoE}}(e) = \\alpha \\cdot \\text{Freq}(e) + (1 - \\alpha) \\cdot \\sum_{e' \\in \\mathcal{E}_{\\text{src}}} C(e', e)$$
Where:
- $\\text{Freq}(e)$ is the historical activation count of expert $e$.
- $C(e', e)$ is the co-activation count between source-stage expert $e'$ and candidate $e$.
- Updates are guarded by 4-bit saturating counters ($0 \\le c \\le 15$) with threshold $T = 2$.

### 2.2 CSTP Predictor Formulation
CSTP formulates spatio-temporal expert transitions with Laplace smoothing:
$$S_{\\text{CSTP}}(e) = \\alpha \\cdot P_{\\text{spatial}}(e \\mid \\mathcal{E}_{\\text{src}}) + (1 - \\alpha) \\cdot P_{\\text{temporal}}(e \\mid \\mathcal{E}_{\\text{prev}})$$
Where:
- Transition probabilities incorporate Laplace smoothing:
  $$P(e \\mid e') = \\frac{N(e', e) + \\lambda}{\\sum_{e''} N(e', e'') + \\lambda \\cdot |\\mathcal{E}|}$$
- Multi-horizon transition matrices $T^{(h)}$ model transitions across horizon lead time $H \\in \\{1, 2, 3\\}$.

---

## 3. Forward-Lookahead Prefetch Scheduling

### 3.1 Temporal Decoupling: Source $L \\to$ Target $L+H$
In earlier co-located prefetching (evaluated in Phase 2B Tier 1), prefetch requests for layer $L$ were dispatched at the beginning of layer $L$. Because computation within layer $L$ lasts only $0.183\\text{ ms}$ while DDR5 transfer takes $3.901\\text{ ms}$, the prefetch had zero time to overlap and always arrived late.

Under **Forward-Lookahead Scheduling**:
- At source stage $L=10$ (with horizon $H=2$):
  1. Router determines top-8 experts for stage 10.
  2. Prediction engine observes stage 10 routing and predicts top-$B$ candidates for stage $L+2 = 12$.
  3. Prefetch candidate queue admits candidates and dispatches them to the DDR5 transfer engine.
  4. While stages 10 and 11 execute their compute on-chip, the speculative transfer proceeds across the DDR5 bus.
  5. When stage 12 arrives, prefetched experts already resident in HBM become **Timely Resident Hits**, eliminating the $3.901\\text{ ms}$ stall entirely.

### 3.2 Causal Isolation & Router Authority
The prediction engine observes only routing events from stage $\\le L$. It has zero visibility into future routing choices. The actual MoE router remains 100% authoritative in `src/module/route.cpp`.

---

## 4. DDR5 Transfer Engine & Transfer Invariants

### 4.1 Bus Modeling
- Dual-channel DDR5-4800 interface calibrated at $22.578\\text{ GB/s}$.
- Isolated expert parameter transfer duration:
  $$\\tau_{\\text{xfer}} = \\frac{88,080,384\\text{ bytes}}{22.578 \\times 10^9\\text{ bytes/sec}} = 3.90116\\text{ ms}$$
- Serialized dispatch of $B=4$ candidates requires $4 \\times 3.90116\\text{ ms} = 15.605\\text{ ms}$.

### 4.2 Demand Preemption & Attachment
- **Demand Priority:** When an actual demand miss occurs, active speculative transfers are paused, and the bus is reassigned to the demand fetch.
- **In-Flight Attachment:** If demand requests an expert that is currently in flight on the bus, the demand request attaches to the existing transfer, avoiding duplicate bandwidth consumption.

---

## 5. HBM Management: PABI Quarantine & Belády Pinning

### 5.1 PABI Quarantine
- **Physical Partitioning:** HBM is partitioned into Demand-Verified (80% capacity) and Speculative Quarantine (20% capacity cap = $16.0\\text{ GB} = 195\\text{ experts}$).
- **Admission & Promotion:** Speculative prefetches are admitted exclusively into quarantine. Upon an actual demand lookup, the expert is promoted to the demand partition.
- **Eviction Isolation:** Speculative mispredictions are evicted strictly within the quarantine pool, completely immunizing demand experts from eviction.

### 5.2 Belády-Style Pinning
- Pinned speculative experts are granted temporary immunity from eviction until their target-stage deadline ($S_{\\text{deadline}} = S_{\\text{target}}$).
- Upon target stage retirement, pins expire monotonically.
- **Empirical Negative Finding:** On Workload A1 with an 80 GB cache, 35,563 pin intents were created and cleanly expired, but because speculative evictions were rare, pinning yielded zero incremental hits over PABI without pinning.

---

## 6. Chronological Bug Discovery & Remediation

| Issue ID | Chronological Defect | Root Cause | Remediation | Verification Result |
| :---: | :--- | :--- | :--- | :--- |
| **BUG-1** | Post-increment layer mismatch | Router prefetch evaluated stage $L$, but `cur_layer++` caused demand lookup at $L+1$. | Introduced logical `current_moe_stage_id`. | Stage IDs 100% aligned. |
| **BUG-2** | Queue retention of completed transfers | `active_candidates_` retained dispatched candidates indefinitely. | Filtered dispatch to `ADMITTED` candidates only. | Transfer conservation verified. |
| **BUG-3** | Candidate queue congestion | Dispatched budget $B=4$ injected $15.6\\text{ ms}$ of work per stage. | Implemented candidate throttling ($B=1, 2$). | Queue delay dropped to $0.0\\text{ ms}$ at $M=12$. |
| **BUG-4** | Remaining transfer logging | Active transfers recorded $0.0\\text{ ms}$ remaining time. | Fixed elapsed duration calculation. | Exact remaining times logged. |
| **BUG-5** | Provenance propagation | Speculative records lacked `source_stage_id`. | Added explicit source stage propagation. | 100% traceable transfer records. |
| **BUG-6** | Monotonic device clock | Module graph restart reset cumulative clock. | Preserved monotonic clock in `restartModuleGraph()`. | Clock continuity validated. |
| **BUG-7** | Predictor metrics parser bug | Summary runner looked for `prediction_records.csv`. | Parsed authentic `live_prediction_summary.json`. | Real precision/recall restored. |
| **BUG-8** | Traffic unit discrepancy | CSV headers labelled binary MiB as decimal `_mb`. | Disambiguated columns into `_mib` and `_mb`. | Dimensional consistency verified. |
| **BUG-9** | Multiplier semantic clarification | Parameter was labelled pure compute multiplier. | Renamed **Synthetic On-Chip Multiplier**. | Rigorous analytical attribution. |
| **BUG-10** | Energy model coverage | Native Ramulator excluded DDR5 bus transport. | Calculated calibrated transport ($28\\text{ pJ/byte}$). | Full disclosure of native vs estimated energy. |

---

## 7. End-to-End System Execution Flow (Mermaid Diagram)

```mermaid
flowchart TD
    A["Authoritative MoE Router"] -->|"Routing Event (Stage L)"| B["Routing History Buffer"]
    B --> C["ST-MoE / CSTP Prediction Engine"]
    C -->|"Top-B Predictions for L+H"| D["Prefetch Candidate Queue"]
    D -->|"ADMITTED Candidates"| E["DDR5 Transfer Engine"]
    E -->|"Asynchronous DDR5 Transfer"| F["PABI Speculative Quarantine (20% Cap)"]
    
    G["Demand Lookup (Stage L+H)"] --> H{"HBM Cache State?"}
    H -->|"Resident in Demand Pool"| I["Natural Hit (Zero Stall)"]
    H -->|"Resident in Quarantine"| J["Timely Resident Hit (Promote to Demand)"]
    H -->|"Active on DDR5 Bus"| K["Late In-Flight Attachment (Wait Remaining Time)"]
    H -->|"Not Prefetched"| L["Demand Miss (High Priority DDR5 Fetch)"]
    
    I --> M["LinearExecutionGPU Compute"]
    J --> M
    K --> M
    L --> M
```

---

## 8. Original Baseline vs Final Architecture Comparison

| Architectural Subsystem | Original Baseline (`d5bf61f`) | Final Implementation | Technical Advantage |
| :--- | :--- | :--- | :--- |
| **Expert Loading** | Purely reactive demand fetch | Asynchronous forward prefetching | Enables DDR5-compute overlap |
| **Prediction Engine** | None | ST-MoE and CSTP engines | Exploits spatial & temporal locality |
| **Lookahead Scheduling** | None | Forward lookahead ($S_{\\text{target}} = S_{\\text{source}} + H$) | Maximizes available transfer lead time |
| **DDR5 Transfer Engine** | Synchronous demand blocking | Asynchronous engine with preemption | Eliminates bus starvation & deadlocks |
| **HBM Partitioning** | Unified LRU cache pool | PABI Quarantine (20% cap) | Immunizes demand working set |
| **Cache Pinning** | None | Belady-style score deadline pinning | Protects high-confidence candidates |
| **Execution Modeling** | Coupled `max(compute, mem)` | Decoupled on-chip window scaling | Preserves baseline exactness at $M=1.0$ |
| **Metric Accounting** | Basic hit/miss counts | Dual conservation laws ($Lookups, Transfers$) | 100% mathematical auditability |
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[SUCCESS] Wrote {readme_path}")

if __name__ == "__main__":
    main()

