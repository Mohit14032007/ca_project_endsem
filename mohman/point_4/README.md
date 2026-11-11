# Point 4: HBM Management Beyond Standard LRU

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Project Objective and Overview

Point 4 investigates **HBM Management** for Mixture-of-Experts (MoE) inference:
> *Implement an HBM management or eviction policy beyond standard LRU. Compare it against LRU and analyse its impact on performance and prefetch effectiveness.*

In MoE architectures such as DeepSeek-V3 (256 routed experts per layer across 40 MoE stages, totaling 10,240 routed experts and ~902 GB of weights), GPU High Bandwidth Memory (HBM) cannot hold all experts simultaneously. The system relies on a two-tier memory hierarchy:
1. **On-Chip / On-Board HBM** (e.g., 10 GB to 80 GB dedicated expert cache).
2. **Off-Chip Host Memory** (DDR5 DRAM over PCIe/CXL interconnect).

Under standard Least Recently Used (LRU) eviction, newly loaded experts displace the least recently accessed experts regardless of their global or recent activation frequency. Because MoE expert routing exhibits pronounced non-uniformity ("heavy hitters" and power-law distribution), LRU suffers from cache thrashing: frequently demanded experts are evicted during bursty token phases, resulting in excessive DDR5 demand traffic and severe GPU execution stalls.

In Point 4, we design, implement, and evaluate a **Frequency-Aware LRU (FA-LRU)** eviction policy to protect frequently demanded experts, reduce demand misses, suppress DDR5 memory traffic, and enhance prefetch timeliness.

---

## 2. Status & Phase Progress

**Current Status:** `PHASE 2 IMPLEMENTATION & SMOKE TESTS COMPLETE — AUTHORIZATION: READY FOR PHASE 3 FULL CAMPAIGN`

In accordance with strict project rules:
- **FA-LRU Implemented:** Candidate window victim selection ($W=16$), shift-safe lazy epoch discounting, and primary projection frequency isolation completed.
- **Unit Tests:** 12/12 Point 4 unit tests (`test_fa_lru`) passed cleanly with 0 errors. All 48 existing regression unit tests passed cleanly.
- **Smoke Tests:** 4-run smoke test campaign (`s1_baseline_lru`, `s2_baseline_fa_lru`, `s3_cstp_lru`, `s4_cstp_fa_lru`) executed cleanly on Prompt 60 (exit code 0, 0 crashes, 0 memory leaks).
- **All 8 Invariants Verified:** Deterministic routing integrity, strict capacity upper bound ($4.0\text{ GB}$), demand conservation ($112 - 67 = 45$), and active evictions ($67 \ge 50$) verified.
- **Zero Ternary Operators:** 100% compliance verified across C++ and Python ASTs.
- All Phase 2 documentation and artifacts are indexed below.

---

## 3. Directory Structure

```text
mohman/point_4/
├── README.md                                 # Top-level Point 4 roadmap and audit status
├── audit/
│   ├── HBM_CACHE_ARCHITECTURE_AUDIT.md       # Comprehensive audit of existing HBM implementation
│   ├── EVICTION_POLICY_EXTENSION_AUDIT.md   # Extension points for plugging in FA-LRU
│   └── AUDIT_GAPS.md                         # Uninstrumented hooks, config discrepancies, and gaps
├── design/
│   ├── FA_LRU_POLICY_DESIGN.md               # Phase 1: Initial mathematical formulation
│   ├── EXPERIMENT_DESIGN.md                  # Phase 1: Initial experiment matrix and plots
│   ├── SMOKE_TEST_PLAN.md                    # Phase 1: Minimal smoke test specification
│   ├── FINAL_POLICY_SPECIFICATION.md         # Phase 1.5: Finalized policy specification & coalescing rules
│   ├── FINAL_EXPERIMENT_MATRIX.md            # Phase 1.5: Finalized 120-run matrix & paired invariants
│   ├── FINAL_METRIC_DEFINITIONS.md           # Phase 1.5: Finalized denominators & anti-inflation accounting
│   ├── FINAL_WORKLOAD_SELECTION.md           # Phase 1.5: Finalized P4-W1..W4 workloads & trace provenance
│   ├── PHASE_1_5_DECISION_REPORT.md          # Phase 1.5: Comprehensive decision report & GO recommendation
│   └── phase_1_6/                            # Phase 1.6: Final consistency corrections & trace replay
│       ├── DEMAND_ACCOUNTING_CORRECTIONS.md  # Resolves D_routed vs D_coalesced vs D_tensor with trace data
│       ├── FA_LRU_FREQUENCY_CORRECTIONS.md   # W1 primary dispatch, explicit flags, state lifecycle
│       ├── LAZY_DECAY_CORRECTIONS.md         # Shift-safe lazy decay materialization (UB prevention)
│       ├── EVICTION_AND_TRANSFER_ACCOUNTING.md # Mutually exclusive eviction taxonomy & transfer deduplication
│       ├── SMOKE_TEST_TRACE_VALIDATION.md    # Exact trace replay of P4-SMOKE across capacities (4.0 GB selected)
│       └── FINAL_DESIGN_CONSISTENCY_REPORT.md # Final consistency audit report and Phase 2 GO authorization
├── configs/                                  # (Reserved for Phase 2: YAML configuration files)
├── implementation/                           # (Reserved for Phase 2: patch files and implementation notes)
├── scripts/                                  # (Reserved for Phase 2: run harnesses and plotting scripts)
├── smoke_test/                               # (Reserved for Phase 2: smoke test logs and outputs)
├── experiments/                              # (Reserved for Phase 3: full campaign runs and manifests)
├── results/                                  # (Reserved for Phase 3: raw CSVs, metrics, and reports)
└── plots/                                    # (Reserved for Phase 3: generated publication figures)
```

---

## 4. Deliverables Index

### Phase 1 Deliverables
| Deliverable | Path | Summary |
| :--- | :--- | :--- |
| **HBM Cache Architecture Audit** | [`audit/HBM_CACHE_ARCHITECTURE_AUDIT.md`](file:///home/mohman/ca_project/mohman/point_4/audit/HBM_CACHE_ARCHITECTURE_AUDIT.md) | Ground-truth analysis of `Device`, residency hash map, LRU lists, capacity checks, and demand vs prefetch flows. |
| **Eviction Extension Audit** | [`audit/EVICTION_POLICY_EXTENSION_AUDIT.md`](file:///home/mohman/ca_project/mohman/point_4/audit/EVICTION_POLICY_EXTENSION_AUDIT.md) | Exact extension points in `prefetch_config.h/cpp` and `device.h/cpp` for non-invasive policy injection. |
| **FA-LRU Policy Design** | [`design/FA_LRU_POLICY_DESIGN.md`](file:///home/mohman/ca_project/mohman/point_4/design/FA_LRU_POLICY_DESIGN.md) | Initial algorithm, frequency metadata, epoch decay, bottom-W candidate window victim selection, and complexity. |
| **Experiment Design** | [`design/EXPERIMENT_DESIGN.md`](file:///home/mohman/ca_project/mohman/point_4/design/EXPERIMENT_DESIGN.md) | Initial 6-way comparison matrix, workloads, 3 HBM capacities, and 10 publication plots. |
| **Smoke Test Plan** | [`design/SMOKE_TEST_PLAN.md`](file:///home/mohman/ca_project/mohman/point_4/design/SMOKE_TEST_PLAN.md) | Constrained-capacity smoke test plan, invariant assertions, verification scripts, and strict GO / INVESTIGATE / NO-GO rules. |
| **Audit Gaps & Ambiguities** | [`audit/AUDIT_GAPS.md`](file:///home/mohman/ca_project/mohman/point_4/audit/AUDIT_GAPS.md) | Detailed enumeration of configuration overrides, no-op collector stubs, and metrics instrumentation requirements. |

### Phase 1.5 Deliverables
| Deliverable | Path | Summary |
| :--- | :--- | :--- |
| **Final Policy Specification** | [`design/FINAL_POLICY_SPECIFICATION.md`](file:///home/mohman/ca_project/mohman/point_4/design/FINAL_POLICY_SPECIFICATION.md) | Resolves sub-projection counting ($W_1$ vs $W_3, W_2$), stage coalescing, lazy epoch discounting, and $W=16$ victim selection. |
| **Final Experiment Matrix** | [`design/FINAL_EXPERIMENT_MATRIX.md`](file:///home/mohman/ca_project/mohman/point_4/design/FINAL_EXPERIMENT_MATRIX.md) | Formal specification of all 120 runs across 6 strategies, 5 capacities, and 4 workloads. |
| **Final Metric Definitions** | [`design/FINAL_METRIC_DEFINITIONS.md`](file:///home/mohman/ca_project/mohman/point_4/design/FINAL_METRIC_DEFINITIONS.md) | Exact mathematical definitions, denominators, reload tracking, traffic counters, and double-counting prevention. |
| **Final Workload Selection** | [`design/FINAL_WORKLOAD_SELECTION.md`](file:///home/mohman/ca_project/mohman/point_4/design/FINAL_WORKLOAD_SELECTION.md) | Provenance of P4-W1 ($B=1$), P4-W2 ($B=4$), P4-W3 ($B=14$), P4-W4 ($B=32$) with 0 prompt duplicates. |
| **Phase 1.5 Decision Report** | [`design/PHASE_1_5_DECISION_REPORT.md`](file:///home/mohman/ca_project/mohman/point_4/design/PHASE_1_5_DECISION_REPORT.md) | Comprehensive synthesis of resolved decisions, risk mitigations, implementation sequence, and GO recommendation. |

### Phase 1.6 Consistency Audit Deliverables
| Deliverable | Path | Summary |
| :--- | :--- | :--- |
| **Demand Accounting Corrections** | [`design/phase_1_6/DEMAND_ACCOUNTING_CORRECTIONS.md`](file:///home/mohman/ca_project/mohman/point_4/design/phase_1_6/DEMAND_ACCOUNTING_CORRECTIONS.md) | Distinguishes $D_{\text{routed}}$, $D_{\text{coalesced}}$, and $D_{\text{tensor}}$. Derives exact numbers from parquet trace data. |
| **FA-LRU Frequency Corrections** | [`design/phase_1_6/FA_LRU_FREQUENCY_CORRECTIONS.md`](file:///home/mohman/ca_project/mohman/point_4/design/phase_1_6/FA_LRU_FREQUENCY_CORRECTIONS.md) | Formalizes explicit primary-demand flag (`is_primary_expert_weight`), eliminating fragile string comparisons. |
| **Lazy Decay Corrections** | [`design/phase_1_6/LAZY_DECAY_CORRECTIONS.md`](file:///home/mohman/ca_project/mohman/point_4/design/phase_1_6/LAZY_DECAY_CORRECTIONS.md) | Solves ISO C++ shift undefined behavior hazard with $\ge 31$ clamp. Details progression hook wrap-around. |
| **Eviction & Transfer Accounting** | [`design/phase_1_6/EVICTION_AND_TRANSFER_ACCOUNTING.md`](file:///home/mohman/ca_project/mohman/point_4/design/phase_1_6/EVICTION_AND_TRANSFER_ACCOUNTING.md) | Mutually exclusive primary eviction taxonomy ($E_{\text{unconsumed}}, E_{\text{demand}}, E_{\text{pabi\_spec}}$) and DDR5 byte conservation. |
| **Smoke Test Trace Validation** | [`design/phase_1_6/SMOKE_TEST_TRACE_VALIDATION.md`](file:///home/mohman/ca_project/mohman/point_4/design/phase_1_6/SMOKE_TEST_TRACE_VALIDATION.md) | Exact LRU replay of P4-SMOKE. Refines capacity to 4.0 GB to produce 48 hits and 67 evictions. |
| **Final Consistency Report** | [`design/phase_1_6/FINAL_DESIGN_CONSISTENCY_REPORT.md`](file:///home/mohman/ca_project/mohman/point_4/design/phase_1_6/FINAL_DESIGN_CONSISTENCY_REPORT.md) | Comprehensive consistency report and final Phase 2 GO authorization. |

### Phase 2 Implementation & Smoke Test Deliverables
| Deliverable | Path | Summary |
| :--- | :--- | :--- |
| **Implementation Baseline** | [`implementation/IMPLEMENTATION_BASELINE.md`](file:///home/mohman/ca_project/mohman/point_4/implementation/IMPLEMENTATION_BASELINE.md) | Initial baseline git diff audit, untracked files catalog, and plan of execution. |
| **Implementation Report** | [`implementation/IMPLEMENTATION_REPORT.md`](file:///home/mohman/ca_project/mohman/point_4/implementation/IMPLEMENTATION_REPORT.md) | Complete narrative on FA-LRU architecture, data structures, lazy decay math, and primary weight isolation. |
| **Source Changelog** | [`implementation/SOURCE_CHANGELOG.md`](file:///home/mohman/ca_project/mohman/point_4/implementation/SOURCE_CHANGELOG.md) | Contiguous diff breakdown across all modified and newly created C++ files. |
| **Build & Test Report** | [`implementation/BUILD_AND_TEST_REPORT.md`](file:///home/mohman/ca_project/mohman/point_4/implementation/BUILD_AND_TEST_REPORT.md) | Full compilation report, 12/12 unit test execution log (`test_fa_lru`), and 48-case regression suite log. |
| **Smoke Test Results** | [`implementation/SMOKE_TEST_RESULTS.md`](file:///home/mohman/ca_project/mohman/point_4/implementation/SMOKE_TEST_RESULTS.md) | Detailed 4-run smoke test metrics table (S1..S4), performance comparison, and tie-breaking analysis. |
| **Correctness Invariants** | [`implementation/CORRECTNESS_INVARIANTS.md`](file:///home/mohman/ca_project/mohman/point_4/implementation/CORRECTNESS_INVARIANTS.md) | Mathematical proofs of residency conservation, eviction mutual exclusion, shift safety, and zero-ternary AST audit. |

---

## 5. Next Steps

1. Review Phase 2 Implementation and Smoke Test Results.
2. Formulate Phase 3 Execution Strategy for the 120-run controlled experimental matrix across W1–W4 workloads and 5 HBM capacities (10 GB to 80 GB).
3. Validate Phase 3 automation runner and slurm/batch harness prior to launch.
