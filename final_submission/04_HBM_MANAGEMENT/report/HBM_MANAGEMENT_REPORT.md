# Requirement 4: HBM Management Beyond LRU (PABI & Belady Pinning)

## 1. Policy Formulation
- **Standard LRU:** Unified 80 GB cache pool. Speculative prefetches compete freely with demand-verified experts, risking speculative pollution during mispredictions.
- **PABI Quarantine:** Enforces strict physical isolation. Speculative region is capped at $20\%$ of total capacity ($16.0\text{ GB} = 195\text{ experts}$). Speculative experts remain in quarantine until validated by an actual demand lookup, completely immunizing the demand working set from eviction.
- **Belady-Inspired Pinning:** In-flight and completed speculative candidates receiving high predictor confidence are pinned until their target-stage execution deadline ($S_{\text{deadline}} = S_{\text{target}}$). Pins expire monotonically upon stage retirement.

## 2. Experimental Verification
Unit tests in [`test_pabi_quarantine.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_pabi_quarantine.cpp) (20 tests) and [`test_pabi_pinning.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_pabi_pinning.cpp) (25 tests) pass with 100% success rate.
