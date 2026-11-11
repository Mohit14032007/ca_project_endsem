# Point 4 Phase 3: Metric Validation and Conservation Report

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Harness Executable:** `run_deepseekv3_mixtral_style`  
**Evaluation Scope:** All 24 Focused Campaign Runs  
**Verification Result:** 100% PASSED (All conservation identities strictly satisfied)  

---

## 1. Mathematical Conservation Invariants

To ensure physical realism and algorithmic validity across all 24 simulation runs, the telemetry engine enforces five conservation identities. Every run was audited against these mathematical requirements:

### Invariant 1: Physical Capacity Conservation
At every execution timestamp $t$ and at simulation termination $t_{\text{end}}$:

$$M_{\text{resident}}(t) \le M_{\text{capacity}}$$

### Invariant 2: Eviction Classification Mutual Exclusion
Total cache evictions must equal the sum of unconsumed prefetch evictions, demand resident evictions, and PABI quarantine evictions:

$$E_{\text{total}} = E_{\text{unconsumed}} + E_{\text{demand}} + E_{\text{PABI}}$$

### Invariant 3: Prefetch-Induced Eviction Upper Bound
Prefetch-induced evictions represent the subset of demand-resident lines evicted specifically while accommodating incoming speculative prefetch lines. By definition, they cannot exceed total demand-resident evictions:

$$E_{\text{prefetch-induced}} \le E_{\text{demand}}$$

### Invariant 4: Physical DDR5 Transfer Traffic Conservation
Total DDR5 physical traffic must equal the exact sum of demand transfer bytes and speculative transfer bytes:

$$B_{\text{total}} = B_{\text{demand}} + B_{\text{speculative}}$$

### Invariant 5: Multi-Level Demand Denominator Consistency
For DeepSeek-V3 execution with 3 GEMM projections (Gate, Up, Down) per expert evaluation:

$$D_{\text{tensor}} = 3 \times D_{\text{routed}}$$

$$D_{\text{coalesced}} \le D_{\text{routed}}$$

---

## 2. Invariant Audit Table Across All 24 Runs

The table below reports the verified numerical telemetry for all 24 runs.

| Run ID | Capacity Budget (Bytes) | Final Resident (Bytes) | Capacity Check | $E_{\text{total}}$ | $E_{\text{uncons}}$ | $E_{\text{demand}}$ | $E_{\text{PABI}}$ | Eviction Check | $E_{\text{induced}}$ | Induced Check | Total Traffic (Bytes) | Traffic Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w2_cap1_c1` | 10,000,000,000 | 9,953,083,392 | PASS | 8,162 | 0 | 8,162 | 0 | PASS | 0 | PASS | 728,865,177,600 | PASS |
| `w2_cap1_c2` | 10,000,000,000 | 9,953,083,392 | PASS | 8,162 | 0 | 8,162 | 0 | PASS | 0 | PASS | 728,865,177,600 | PASS |
| `w2_cap1_c3` | 10,000,000,000 | 9,953,083,392 | PASS | 8,392 | 227 | 8,165 | 0 | PASS | 232 | PASS | 755,553,534,464 | PASS |
| `w2_cap1_c4` | 10,000,000,000 | 9,953,083,392 | PASS | 8,392 | 227 | 8,165 | 0 | PASS | 231 | PASS | 755,553,534,464 | PASS |
| `w2_cap2_c1` | 37,580,963,840 | 37,522,243,584 | PASS | 6,665 | 0 | 6,665 | 0 | PASS | 0 | PASS | 624,578,002,944 | PASS |
| `w2_cap2_c2` | 37,580,963,840 | 37,522,243,584 | PASS | 6,665 | 0 | 6,665 | 0 | PASS | 0 | PASS | 624,578,002,944 | PASS |
| `w2_cap2_c3` | 37,580,963,840 | 37,522,243,584 | PASS | 7,296 | 217 | 7,079 | 0 | PASS | 227 | PASS | 684,384,584,192 | PASS |
| `w2_cap2_c4` | 37,580,963,840 | 37,522,243,584 | PASS | 7,296 | 217 | 7,079 | 0 | PASS | 225 | PASS | 684,384,584,192 | PASS |
| `w2_cap3_c1` | 80,000,000,000 | 79,976,988,672 | PASS | 3,895 | 0 | 3,895 | 0 | PASS | 0 | PASS | 423,050,084,352 | PASS |
| `w2_cap3_c2` | 80,000,000,000 | 79,976,988,672 | PASS | 3,895 | 0 | 3,895 | 0 | PASS | 0 | PASS | 423,050,084,352 | PASS |
| `w2_cap3_c3` | 80,000,000,000 | 79,976,988,672 | PASS | 4,142 | 171 | 3,971 | 0 | PASS | 198 | PASS | 446,919,868,416 | PASS |
| `w2_cap3_c4` | 80,000,000,000 | 79,976,988,672 | PASS | 4,142 | 171 | 3,971 | 0 | PASS | 198 | PASS | 446,919,868,416 | PASS |
| `w3_cap1_c1` | 10,000,000,000 | 9,953,083,392 | PASS | 13,051 | 0 | 13,051 | 0 | PASS | 0 | PASS | 1,159,490,174,976 | PASS |
| `w3_cap1_c2` | 10,000,000,000 | 9,953,083,392 | PASS | 13,051 | 0 | 13,051 | 0 | PASS | 0 | PASS | 1,159,490,174,976 | PASS |
| `w3_cap1_c3` | 10,000,000,000 | 9,953,083,392 | PASS | 13,364 | 311 | 13,053 | 0 | PASS | 314 | PASS | 1,192,432,238,592 | PASS |
| `w3_cap1_c4` | 10,000,000,000 | 9,953,083,392 | PASS | 13,364 | 311 | 13,053 | 0 | PASS | 315 | PASS | 1,192,432,238,592 | PASS |
| `w3_cap2_c1` | 37,580,963,840 | 37,522,243,584 | PASS | 12,738 | 0 | 12,738 | 0 | PASS | 0 | PASS | 1,159,490,174,976 | PASS |
| `w3_cap2_c2` | 37,580,963,840 | 37,522,243,584 | PASS | 12,738 | 0 | 12,738 | 0 | PASS | 0 | PASS | 1,159,490,174,976 | PASS |
| `w3_cap2_c3` | 37,580,963,840 | 37,522,243,584 | PASS | 13,038 | 291 | 12,747 | 0 | PASS | 302 | PASS | 1,191,199,112,704 | PASS |
| `w3_cap2_c4` | 37,580,963,840 | 37,522,243,584 | PASS | 13,038 | 292 | 12,746 | 0 | PASS | 302 | PASS | 1,191,199,112,704 | PASS |
| `w3_cap3_c1` | 80,000,000,000 | 79,976,988,672 | PASS | 9,078 | 0 | 9,078 | 0 | PASS | 0 | PASS | 879,570,714,624 | PASS |
| `w3_cap3_c2` | 80,000,000,000 | 79,976,988,672 | PASS | 9,078 | 0 | 9,078 | 0 | PASS | 0 | PASS | 879,570,714,624 | PASS |
| `w3_cap3_c3` | 80,000,000,000 | 79,976,988,672 | PASS | 9,500 | 249 | 9,251 | 0 | PASS | 299 | PASS | 920,175,771,648 | PASS |
| `w3_cap3_c4` | 80,000,000,000 | 79,976,988,672 | PASS | 9,500 | 249 | 9,251 | 0 | PASS | 299 | PASS | 920,175,771,648 | PASS |

---

## 3. Demand Accounting Denominator Audit

Following the Phase 2.6 accounting correction, all three demand granularities were recorded and verified:

### Workload W2 ($B=4, T=16, L=40, \text{Top-}K=8$)
- $D_{\text{coalesced}} = 8,275$
- $D_{\text{routed}} = 8,275$
- $D_{\text{tensor}} = 24,825$ ($= 3 \times 8,275$)
- Coalescing ratio: $D_{\text{routed}} / D_{\text{coalesced}} = 1.0000$ (across 4 sequences, routing overlap per stage is low).
- Projection factor: $D_{\text{tensor}} / D_{\text{routed}} = 3.0000$ (Gate, Up, Down projections).
- Invalid decode-step demands: Exactly 0 across all 12 W2 runs.

### Workload W3 ($B=14, T=16, L=40, \text{Top-}K=8$)
- $D_{\text{coalesced}} = 13,164$
- $D_{\text{routed}} = 13,164$
- $D_{\text{tensor}} = 39,492$ ($= 3 \times 13,164$)
- Coalescing ratio: $D_{\text{routed}} / D_{\text{coalesced}} = 1.0000$.
- Projection factor: $D_{\text{tensor}} / D_{\text{routed}} = 3.0000$.
- Invalid decode-step demands: Exactly 0 across all 12 W3 runs.

---

## 4. Verification Conclusion

Every single run satisfied all conservation identities and denominator requirements. The telemetry data across the 24 runs is numerically consistent, causally sound, and certified for comparative performance analysis.

