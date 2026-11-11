# MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference
## End-Semester Capstone Project Defense

---

### Slide 1: The MoE Memory Bottleneck
- Mixture-of-Experts (MoE) decouples parameter capacity from FLOPs.
- Moving expert weights from off-chip DDR5 (22.578 GB/s) to on-chip HBM (2 TB/s) introduces massive memory stall (up to 98% of runtime).
- Isolated expert transfer latency: **3.90116 ms** for 88.08 MB (FP16).
- Computation window per MoE stage: only **0.183 ms** on nominal hardware!

---

### Slide 2: Proposed Solution & Architecture
- **Training-Free Prediction:** ST-MoE (frequency + spatial co-activation) and CSTP (Laplace-smoothed spatio-temporal Markov model).
- **Forward-Lookahead Prefetching:** Schedules prefetch at stage $L$ for target stage $L+H$ ($H=2$).
- **DDR5 Transfer Engine:** Asynchronous transfers with demand preemption, pause/resume, and in-flight demand attachment.
- **PABI Quarantine & Belady Pinning:** 20% capacity cap isolates speculative prefetches, preventing cache pollution.

---

### Slide 3: Core Empirical Findings
- **Emergence of Timely Hits:** At $M \ge 4\times$, timely prefetch hits emerge, eliminating exposed memory stall.
- **Queue Throttling Advantage:** Reducing candidate budget from $B=4$ to $B=1$ or $B=2$ eliminates transfer queue waiting time (from 2171 ms to 0.0 ms) and cuts speculative traffic by 77.7%!
- **100% Invariant Conservation:** Verified across all runs without exception.
