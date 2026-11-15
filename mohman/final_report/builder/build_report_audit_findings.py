#!/usr/bin/env python3
"""
build_report_audit_findings.py:
Generates the comprehensive forensic audit document:
mohman/final_report/REPORT_AUDIT_FINDINGS.md
Enforces strict zero-ternary rule.
"""

import os
import sys

def build_findings():
    content = """# FORENSIC AUDIT FINDINGS AND TECHNICAL RECONCILIATION REPORT

**Course:** CSE 511/ECE 511: Computer Architecture  
**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `https://github.com/Mohit14032007/ca_project_endsem`  
**Authors:** Mohit Gupta (2024354), Mrityunjai Poddar (2024358), Saksham Madan (2024493)  
**Date:** November 2026  
**Audit Destination:** `mohman/final_report/REPORT_AUDIT_FINDINGS.md`

---

## 1. Executive Summary of Forensic Investigation

A complete, exhaustive technical audit was conducted across the 58-page project report (`MoE_Memory_Wall_Final_Report.docx`), the underlying C++ simulator implementation (`LLMSimulator/src/`), experiment configuration manifests, raw trace files, consolidated result CSVs, and visualization scripts.

The audit verified that while the overarching architectural thesis—leveraging spatial and temporal correlation to prefetch Mixture-of-Experts weights from host DDR5 to accelerator HBM under offloaded inference—is solidly supported by the experimental data, several critical discrepancies existed between the written descriptions and the actual implemented simulator mechanisms.

Specifically:
1. **Algorithmic Discrepancies:** The mathematical scoring models attributed to both the proposed predictor (CSTP) and the literature baseline (ST-MoE) did not match the C++ source code. CSTP was described as using continuous exponential decay over a correlation matrix dot-product, whereas it actually implements Dirichlet-smoothed marginal priors, Dirichlet-smoothed spatial conditional transitions, and Dirichlet-smoothed temporal transitions. ST-MoE was described as a generic transition frequency Markov model, whereas it actually implements a Cross-Layer Correlation Table (CCT) with 2-bit saturating counters ($c \\in \\{0, 1, 2, 3\\}$) and a History Table (+2 bonus) with reinforcement update mechanics.
2. **Implementation Locations:** The Frequency-Aware LRU (FA-LRU) policy was attributed to a non-existent file `mohman/point_4/src/fa_lru_policy.cpp` with a continuous mathematical penalty formula, whereas it is actually integrated directly into `LLMSimulator/src/hardware/device.cpp` as a windowed bottom-$W$ candidate scan with lazy bit-shift epoch decay (`decayed = access_frequency >> epoch_diff`).
3. **Numerical and Table Reconciliation:** In Table 8.1, stall savings from Point 3 Workload W7 (6.717s) were mistakenly inserted into the Point 6 lookahead sweep table for Workload W3. In Table 9.2, the GPU MAC compute energy column ($71.25\\text{ J}$) was omitted from the subsystem breakdown.
4. **Energy Optimum Identification:** Prior text implied that prefetching always increases energy or that Config D at $K=4$ was optimal. Forensic ledger analysis proves that $K=2$ (`k2_c4`) is the authentic system energy and Energy-Delay Product (EDP) sweet spot, delivering a net energy reduction of $-27.68\\text{ J}$ ($-0.29\\%$) and an EDP reduction of $-1.72\\%$ ($7,079.32\\text{ kJ}\\cdot\\text{s}$ vs $7,202.85\\text{ kJ}\\cdot\\text{s}$).

All discrepancies have been resolved, backed by direct C++ source code references, verified CSV rows, and mathematical derivations.

---

## 2. Master Forensic Audit Table

| Audit ID | Report Location | Original Claim / Text | Actual Repository Evidence | Audit Status | Discrepancy Nature | Correction Applied | Severity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AUD-01** | Chapter 3, Sec 3.2, Table 3.1 | CSTP uses continuous exponential temporal decay with correlation matrix dot-product `score = (corr * 0.70) + (exp(-decay * dt) * 0.30)` | `LLMSimulator/src/prefetch/cstp_predictor.cpp` lines 70–260; `cstp_predictor.h` lines 40–110 | **INCORRECT** | Formula was fabricated/placeholder. Reality: Dirichlet-smoothed spatial transitions, temporal transitions, and marginal priors with convex parameter $\\alpha$. Dynamic `history_table_` updates per token even in frozen mode. | Derived and replaced with exact C++ Dirichlet smoothing equations, pseudocode, and C++ source excerpt in Chapter 3. | **Critical** |
| **AUD-02** | Chapter 3, Sec 3.1, Table 3.1 | ST-MoE uses generic Markov transition frequency matrix with static popularity fallback | `LLMSimulator/src/prefetch/st_moe_predictor.cpp` lines 30–220; `st_moe_predictor.h` (Zhao et al. arXiv:2606.15453v1) | **INCORRECT** | Overly simplified. Reality: 2-bit saturating counters ($c \\in \\{0, 1, 2, 3\\}$, init=2) in Cross-Layer Correlation Table (CCT), History Table (+2 bonus), and reinforcement update (match +1, mismatch -1, candidate replacement). | Replaced text, mathematical equations, and code snippet in Chapter 3 with exact CCT and HT reinforcement logic. | **Critical** |
| **AUD-03** | Chapter 4 (Table 4.1) & Chapter 6 | FA-LRU implemented in `mohman/point_4/src/fa_lru_policy.cpp` using formula `(F / delta_t) * penalty` | `LLMSimulator/src/hardware/device.cpp` lines 682–800 (`Device::select_fa_lru_victim` and `materialize_lazy_decay`) | **INCORRECT** | File path did not exist; formula was synthetic. Reality: Windowed bottom-$W$ candidate scan ($W=64$), lazy epoch halving via right-shift, fallback beyond window, and divergence tracking. | Replaced path with `device.cpp`; replaced formula with windowed bottom-$W$ minimum-frequency scan and lazy epoch halving equations. | **Critical** |
| **AUD-04** | Chapter 8, Table 8.1 | Table 8.1 reported 6.717s stall hidden for lookahead $H=1$ in Workload W3 | `mohman/point_6/final_results/consolidated_data/lookahead_comparison.csv` vs `mohman/point_3/results/all_experiment_results.csv` | **DISCREPANCY** | Mixed data campaigns. 6.717s belonged to Workload W7 ($B=128, T=16$) in Point 3. In Point 6 ($B=32, P=2048, T=128, K=4$), stall hidden is $3,568.90\\text{ ms}$ ($H=1$), $3,873.24\\text{ ms}$ ($H=2$), and $4,005.17\\text{ ms}$ ($H=4$). | Replaced Table 8.1 with verified Point 6 sensitivity data from `lookahead_comparison.csv`. | **Major** |
| **AUD-05** | Chapter 9, Table 9.2 | Table 9.2 omitted GPU MAC compute energy column and presented partial breakdown | `mohman/point_7/extracted_data/energy_component_breakdown.csv` | **DISCREPANCY** | GPU MAC compute energy ($71.25\\text{ J}$, $0.76\\%$ share) was discussed in text but omitted from table columns. | Added dedicated GPU MAC energy column ($71.25\\text{ J}$) and accurate percentage shares to Table 9.2. | **Major** |
| **AUD-06** | Chapter 9, Sec 9.3 & Executive Summary | Ambiguous statements regarding whether prefetching saves energy, or claiming Config D at $K=4$ was optimal | `mohman/point_7/extracted_data/unified_energy_results.csv` row `k2_c4` | **DISCREPANCY** | At $K=4$ (`anc_c4`), speculative traffic causes $+0.31\\%$ energy overhead ($9,416.66\\text{ J}$ vs $9,387.61\\text{ J}$). However, at $K=2$ (`k2_c4`), static DRAM savings ($-49.18\\text{ J}$) exceed speculative traffic ($+18.63\\text{ J}$), yielding a net saving of $-27.68\\text{ J}$ and lowest EDP ($7,079.32\\text{ kJ}\\cdot\\text{s}$). | Explicitly designated $K=2$ as the tested system energy and EDP optimum in Chapter 9, Executive Summary, and tables. | **Major** |
| **AUD-07** | Chapters 2, 4, 5, 8 | Multiple transfer times quoted without campaign context (1.376 ms, 3.720 ms, 3.901 ms, 4.096 ms) | Configuration files `sys_config.json`, `per_run_metrics.csv`, `lookahead_comparison.csv` | **DISCREPANCY** | Transfer durations depended on simulated link bandwidth and tensor sizing across different experimental phases. | Explicitly annotated each duration: 1.376 ms (88 MB @ 64 GB/s in Point 3); 3.901 ms (88 MB @ 22.578 GB/s in Point 5/6/7); 3.720 ms (84 MiB @ 22.578 GB/s); 4.096 ms (serialized uncoalesced expert fetch). | **Major** |
| **AUD-08** | Chapter 2 (Sec 2.3) & Chapter 8 | Standard Multi-Head Attention (MHA) KV cache equation cited for DeepSeek-V3 | DeepSeek-V3 Technical Report & repository architectural profiles | **DISCREPANCY** | DeepSeek-V3 uses Multi-Head Latent Attention (MLA), which compresses KV cache into a low-rank latent vector ($d_c = 512, d_{c,\\text{rope}} = 64$), drastically reducing KV footprint from 60.4 GiB to 4.06 GiB. | Added full MLA KV cache mathematical derivation and compared standard MHA vs MLA footprint, explaining why 35 GiB of HBM is available for expert cache. | **Major** |
| **AUD-09** | Chapter 4, Table 4.2 | Worked numerical execution trace lacked microsecond mathematical derivation | Hardware timing constants: $84\\text{ MiB} / 22.578\\text{ GB/s} = 3.720\\text{ ms}$; GEMM compute overlap | **REFINED** | Timestamps in earlier versions were rough approximations rather than clock-exact derivations. | Rebuilt Table 4.2 with mathematically derived sequence of events showing exact dispatch, arrival, demand check, GEMM overlap, and late attachment stall hiding. | **Minor** |
| **AUD-10** | Chapter 3, Sec 3.3 | Precision, Recall, Hit Rate, and F1 used without explicit multi-label top-K set-theoretic definitions | Evaluation scripts in `mohman/point_1/scripts/` | **REFINED** | Top-K routing requires explicit set-theoretic definitions to distinguish top-K recall from hit rate and exact set accuracy. | Added rigorous definitions for $\\text{Recall}@K$, $\\text{Precision}@K$, $\\text{HitRate}@K$, and $\\text{ExactSetAccuracy}$. | **Minor** |

---

## 3. Deep-Dive Forensic Analyses and Code Verifications

### 3.1. CSTP Predictor Implementation vs. Literature Claims

In `LLMSimulator/src/prefetch/cstp_predictor.cpp`, the Correlation-based Spatio-Temporal Predictor (CSTP) does not compute an exponential decay over a static correlation matrix. Instead, it maintains empirical co-occurrence and transition frequency counts smoothed by Dirichlet priors.

#### Mathematical Formulation in C++ Source:
1. **Marginal Prior Probability:**
   For any stage $L$ and candidate expert $v \\in \\{0, \\dots, E-1\\}$:
   $$P_{\\text{prior}}(v \\mid L) = \\frac{M[L, v] + \\epsilon}{\\sum_{j=0}^{E-1} M[L, j] + E \\cdot \\epsilon}$$
   where $M[L, v]$ is the marginal activation count of expert $v$ in stage $L$, and $\\epsilon = 10^{-6}$ (`prior_epsilon_`).

2. **Spatial Conditional Probability:**
   Given active source experts $U \\subseteq \\{0, \\dots, E-1\\}$ at source stage $L_{\\text{src}}$:
   $$P_s(v \\mid u, L_{\\text{src}}, L_{\\text{tgt}}) = \\frac{C_s[L_{\\text{src}}, u, v] + \\lambda P_{\\text{prior}}(v \\mid L_{\\text{tgt}})}{\\sum_{j=0}^{E-1} C_s[L_{\\text{src}}, u, j] + \\lambda}$$
   where $C_s[L_{\\text{src}}, u, v]$ is the spatial co-activation count matrix and $\\lambda = 1.0$ (`smoothing_lambda_`). The aggregated spatial score is:
   $$S_s(v) = \\frac{1}{|U|} \\sum_{u \\in U} P_s(v \\mid u, L_{\\text{src}}, L_{\\text{tgt}})$$

3. **Temporal Conditional Probability:**
   Given the actual experts $W$ routed at target stage $L_{\\text{tgt}}$ during the *previous token* (stored in `history_table_[request_id][target_stage_id]`):
   $$P_t(v \\mid w, L_{\\text{tgt}}) = \\frac{C_t[L_{\\text{tgt}}, w, v] + \\lambda P_{\\text{prior}}(v \\mid L_{\\text{tgt}})}{\\sum_{j=0}^{E-1} C_t[L_{\\text{tgt}}, w, j] + \\lambda}$$
   The aggregated temporal score is:
   $$S_t(v) = \\frac{1}{|W|} \\sum_{w \\in W} P_t(v \\mid w, L_{\\text{tgt}})$$

4. **Composite Score & Top-K Admission:**
   $$S(v) = \\alpha S_s(v) + (1 - \\alpha) S_t(v), \\quad \\alpha = 0.5$$
   The top-$K$ candidates are selected by descending composite score:
   $$\\mathcal{P}_K = \\operatorname{TopK}_{v \\in \\{0, \\dots, E-1\\}} S(v)$$

#### Behavior in Frozen Profiled Mode:
When `online_adaptation_ == false`, the count matrices $M$, $C_s$, and $C_t$ remain static (frozen from profiling). Crucially, however, `history_table_[request_id][target_stage_id]` *is still updated dynamically* after each token's routing decisions (`cstp_predictor.cpp` line 328). This allows temporal context tracking across consecutive tokens even without modifying the profiled transition matrices.

---

### 3.2. ST-MoE Baseline Model and 2-Bit Saturating Counters

The literature baseline ST-MoE (Zhao et al., arXiv:2606.15453v1, June 2026) is implemented in `LLMSimulator/src/prefetch/st_moe_predictor.cpp`.

#### Core Mechanism:
1. **Cross-Layer Correlation Table (CCT):**
   Stores a 2-bit saturating counter $c_{L, u, v} \\in \\{0, 1, 2, 3\\}$ (initialized to 2) with capacity $C = 8$ entries per source expert.
2. **History Table (HT):**
   Caches the target layer experts from the immediately preceding token. If candidate $v \\in \\text{HT}$, it receives a $+2.0$ confidence bonus.
3. **Scoring Function:**
   $$S_{\\text{ST-MoE}}(v) = \\sum_{u \\in U} c_{L_{\\text{src}}, u, v} + 2 \\cdot \\mathbf{1}[v \\in \\text{HT}[L_{\\text{tgt}}]]$$
4. **Reinforcement Learning Update (Algorithm 3):**
   - For all candidate entries in CCT: if the candidate matches an actual target expert, $c \\leftarrow \\min(3, c+1)$; if it mismatches, $c \\leftarrow c-1$.
   - Any actual target expert not present in CCT replaces an entry with $c = 0$, initializing the new entry with $c = 2$.

---

### 3.3. Frequency-Aware LRU (FA-LRU) Implementation

In `LLMSimulator/src/hardware/device.cpp`:
- **Function:** `Device::select_fa_lru_victim()` (lines 701–785) and `Device::materialize_lazy_decay()` (lines 682–699).
- **Algorithm:**
  1. Inspects the bottom-$W$ candidates ($W = \\text{fa\\_lru\\_window\\_size} = 64$) of `lru_expert_list` (which is ordered by recency, LRU at head).
  2. For each unpinned candidate, applies lazy epoch halving:
     $$\\Delta e = e_{\\text{global}} - e_{\\text{last}}$$
     $$\\text{access\\_frequency} = \\max\\left(1, \\text{access\\_frequency} \\gg \\Delta e\\right) \\quad (\\text{if } \\Delta e < 31, \\text{ else } 1)$$
  3. Selects victim candidate $k^* = \\arg\\min_{k \\in \\mathcal{C}_W} \\text{access\\_frequency}(k)$.
  4. Ties are broken by oldest recency (since the list is scanned from LRU head).
  5. If all bottom-$W$ candidates are pinned, falls back to the first unpinned candidate beyond $W$.
  6. Tracks divergent decisions: when $k^* \\neq k_{\\text{LRU}}$, increments `fa_lru_diag.divergent_decisions`. If the standard LRU victim was demand-verified, increments `fa_lru_diag.protected_demand_experts_count`.

---

### 3.4. Sensitivity Analysis & Table 8.1 Discrepancy Resolution

Table 8.1 in earlier drafts listed $6.717\\text{ s}$ stall hidden for lookahead $H=1$ in Workload W3.
Forensic tracing revealed that $6.717\\text{ s}$ came from Workload W7 ($B=128, T=16$) in Point 3 large workloads (`all_experiment_results.csv`).
In Point 6 sensitivity sweeps (`lookahead_comparison.csv`), the parameters are $B=32, P=2048, T=128, K=4$. Under this workload, the authentic metrics are:
- $H=1$: Execution time $761.63\\text{ s}$, Exposed stall $747,583.37\\text{ ms}$, Stall hidden $3,568.90\\text{ ms}$, Late attachment $21.29\\%$, Wasted traffic $665.36\\text{ GB}$.
- $H=2$: Execution time $760.91\\text{ s}$, Exposed stall $746,864.77\\text{ ms}$, Stall hidden $3,873.24\\text{ ms}$, Late attachment $19.92\\%$, Wasted traffic $646.25\\text{ GB}$.
- $H=4$: Execution time $760.20\\text{ s}$, Exposed stall $746,152.06\\text{ ms}$, Stall hidden $4,005.17\\text{ ms}$, Late attachment $18.15\\%$, Wasted traffic $637.88\\text{ GB}$.

Table 8.1 has been updated with these exact numbers.

---

### 3.5. Energy Component Breakdown & Sweet Spot Identification

From `mohman/point_7/extracted_data/energy_component_breakdown.csv` and `unified_energy_results.csv`:
- Physical coefficients: $P_{\\text{static}} = 4.5\\text{ W}$, $E_{\\text{ddr5}} = 0.1\\text{ nJ/byte}$, $E_{\\text{hbm}} = 3.48\\text{ pJ/bit}$, $E_{\\text{mac}} = 0.23\\text{ pJ/FLOP}$.
- For Baseline (`anc_c1`, $767.27\\text{ s}$):
  - HBM Energy: $1,042.87\\text{ J}$ ($11.11\\%$)
  - GPU MAC Energy: $71.25\\text{ J}$ ($0.76\\%$)
  - DDR5 Active Energy: $4,820.77\\text{ J}$ ($51.35\\%$)
  - Static DRAM Energy: $3,452.72\\text{ J}$ ($36.78\\%$)
  - Total System Energy: $9,387.61\\text{ J}$
  - System EDP: $7,202.85\\text{ kJ}\\cdot\\text{s}$
- For Config D ($K=4$, `anc_c4`, $761.63\\text{ s}$):
  - HBM: $1,044.18\\text{ J}$, MAC: $71.25\\text{ J}$, DDR5: $4,873.91\\text{ J}$, Static DRAM: $3,427.32\\text{ J}$.
  - Total Energy: $9,416.66\\text{ J}$ ($+0.31\\%$ overhead vs baseline).
  - System EDP: $7,171.98\\text{ kJ}\\cdot\\text{s}$ ($-0.43\\%$ reduction).
- For Tested Optimum ($K=2$, `k2_c4`, $756.34\\text{ s}$):
  - Execution time improves by $10.93\\text{ s}$ ($-1.42\\%$).
  - Static DRAM energy drops from $3,452.72\\text{ J}$ to $3,403.54\\text{ J}$ (saving $-49.18\\text{ J}$).
  - Speculative DDR5 energy adds only $+18.63\\text{ J}$ ($4,839.40\\text{ J}$ vs $4,820.77\\text{ J}$).
  - Total System Energy drops to $9,359.93\\text{ J}$ (net saving of $-27.68\\text{ J}$, $-0.29\\%$).
  - System EDP drops to $7,079.32\\text{ kJ}\\cdot\\text{s}$ ($-1.72\\%$ reduction).

$K=2$ is mathematically and experimentally confirmed as the optimal prefetching budget.

---

## 4. Conclusion & Verification Sign-Off

Every discrepancy cataloged in this forensic audit has been reconciled in the final deliverables:
- `mohman/final_report/MASTER_RESULT_LEDGER.csv` (191 runs)
- `mohman/final_report/FIGURE_TABLE_AUDIT.csv` (47 items)
- `mohman/final_report/MoE_Memory_Wall_Final_Report_CORRECTED.docx`
- `mohman/final_report/MoE_Memory_Wall_Final_Report_CORRECTED.pdf`
- `mohman/final_report/FINAL_CORRECTION_SUMMARY.md`
"""

    os.makedirs("mohman/final_report", exist_ok=True)
    out_path = "mohman/final_report/REPORT_AUDIT_FINDINGS.md"
    with open(out_path, "w") as f:
        f.write(content)
    print(f"Successfully generated {out_path} ({len(content)} bytes).")

if __name__ == "__main__":
    build_findings()
