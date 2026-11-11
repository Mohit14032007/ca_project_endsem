# Bonus Objective: Evaluation on Real MoE Routing Traces

## 1. Real Routing Trace Provenance
Rather than relying exclusively on synthetic random routing, the project evaluated authentic, recorded expert activation sequences:
1. **MoE-Suite (Task 12):** GLM-5.2 architecture (75 MoE layers, 256 experts/layer, Top-8 routing). Authentic multi-token software engineering workload traces.
2. **MoE-Anywhere (Prompt 60):** Sage-A1.5 architecture (40 MoE layers, 256 experts/layer, Top-8 routing). Real autoregressive inference execution trace.
3. **Negative Control Baseline:** Dynamic pseudo-random trace (MT19937 seed 777) used to demonstrate zero correlation and validate that prediction advantages stem exclusively from genuine spatial/temporal patterns.
