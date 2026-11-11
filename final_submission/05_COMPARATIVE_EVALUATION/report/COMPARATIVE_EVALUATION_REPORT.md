# Requirement 5: Complete Comparative Evaluation and Component Ablation

## 1. Controlled Attribution
The nine-configuration evaluation isolates the exact contribution of each architectural subsystem:
- **P vs B:** Evaluates pure predictor computational overhead ($< 0.16\%$ host overhead) with zero speculative traffic or cache disturbance.
- **PF vs P:** Evaluates prefetching contribution; CSTP (`PF-C`) achieves $945.92\text{ ms}$ stall reduction, outperforming ST-MoE (`PF-S`, $614.03\text{ ms}$) due to superior spatio-temporal transition modeling.
- **QUAR vs PF:** Evaluates PABI quarantine protection. Eliminates risk of speculative eviction of hot demand experts.
- **FULL vs QUAR:** Evaluates Belady-inspired pinning. Pinning successfully retains high-priority candidates across multi-stage lookahead.
