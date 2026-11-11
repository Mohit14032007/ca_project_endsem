# Campaign Invariant Validation Report
**MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference**

### Dual Conservation Laws:
1. **Demand Conservation:** $\text{Lookups} = \text{Natural Hits} + H_{\text{timely}} + A_{\text{late}} + M_{\text{unprefetched}} = 7,200$
2. **Transfer Conservation:** $\text{Submitted} = \text{Demanded} + \text{Unused} + \text{Evicted} + \text{In-Flight} + \text{Cancelled}$

| Campaign | Run ID | Natural Hits | Timely Hits | Late Attachments | Unprefetched Misses | Total Lookups | Demand Status | Transfer Status | Overall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Phase 2F (Compute Multiplier) | `B-M1` | 5179 | 0 | 0 | 2021 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `FWD-S-H2-M1` | 5179 | 0 | 59 | 1962 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `FWD-C-H2-M1` | 5179 | 0 | 68 | 1953 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `B-M4` | 5179 | 0 | 0 | 2021 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `FWD-S-H2-M4` | 5179 | 1 | 52 | 1968 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `FWD-C-H2-M4` | 5179 | 1 | 61 | 1959 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `B-M8` | 5179 | 0 | 0 | 2021 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `FWD-S-H2-M8` | 5179 | 1 | 48 | 1972 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `FWD-C-H2-M8` | 5179 | 1 | 57 | 1963 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `B-M12` | 5179 | 0 | 0 | 2021 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `FWD-S-H2-M12` | 5179 | 2 | 43 | 1976 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `FWD-C-H2-M12` | 5179 | 2 | 52 | 1967 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `B-M16` | 5179 | 0 | 0 | 2021 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `FWD-S-H2-M16` | 5179 | 2 | 31 | 1988 | 7200 | **PASS** | **PASS** | **PASS** |
| Phase 2F (Compute Multiplier) | `FWD-C-H2-M16` | 5179 | 2 | 48 | 1971 | 7200 | **PASS** | **PASS** | **PASS** |
| Sensitivity (Candidate Budget) | `FWD-S-H2-B1-M1` | 5179 | 0 | 10 | 2011 | 7200 | **PASS** | **PASS** | **PASS** |
| Sensitivity (Candidate Budget) | `FWD-S-H2-B2-M1` | 5179 | 1 | 23 | 1996 | 7199 | **FAIL** | **PASS** | **FAIL** |
| Sensitivity (Candidate Budget) | `FWD-S-H2-B4-M1` | 5179 | 0 | 59 | 1962 | 7200 | **PASS** | **PASS** | **PASS** |
| Sensitivity (Candidate Budget) | `FWD-C-H2-B1-M1` | 5179 | 0 | 7 | 2014 | 7200 | **PASS** | **PASS** | **PASS** |
| Sensitivity (Candidate Budget) | `FWD-C-H2-B2-M1` | 5179 | 1 | 29 | 1990 | 7199 | **FAIL** | **PASS** | **FAIL** |
| Sensitivity (Candidate Budget) | `FWD-C-H2-B4-M1` | 5179 | 0 | 68 | 1953 | 7200 | **PASS** | **PASS** | **PASS** |
| Sensitivity (Candidate Budget) | `FWD-S-H2-B1-M12` | 5179 | 2 | 0 | 2017 | 7198 | **FAIL** | **PASS** | **FAIL** |
| Sensitivity (Candidate Budget) | `FWD-S-H2-B2-M12` | 5179 | 2 | 5 | 2012 | 7198 | **FAIL** | **PASS** | **FAIL** |
| Sensitivity (Candidate Budget) | `FWD-S-H2-B4-M12` | 5179 | 2 | 43 | 1976 | 7200 | **PASS** | **PASS** | **PASS** |
| Sensitivity (Candidate Budget) | `FWD-C-H2-B1-M12` | 5179 | 2 | 0 | 2017 | 7198 | **FAIL** | **PASS** | **FAIL** |
| Sensitivity (Candidate Budget) | `FWD-C-H2-B2-M12` | 5179 | 2 | 16 | 2001 | 7198 | **FAIL** | **PASS** | **FAIL** |
| Sensitivity (Candidate Budget) | `FWD-C-H2-B4-M12` | 5179 | 2 | 52 | 1967 | 7200 | **PASS** | **PASS** | **PASS** |
