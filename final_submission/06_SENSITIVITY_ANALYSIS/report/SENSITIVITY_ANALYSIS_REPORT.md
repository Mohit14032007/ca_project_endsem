# Requirement 6: Comprehensive Sensitivity Analysis Across 5 Architectural Dimensions

## 1. Summary of Sensitivity Dimensions
1. **Batch Concurrency ($B \in \{1, 2, 4\}$):** Evaluated on MoE-Anywhere ($T=128$). As batch size increases, per-stage computation scales proportionally, providing linear aggregate throughput while maintaining consistent HBM hit rates ($82.8$--$83.1\%$).
2. **Context Length ($L_{\text{in}} \in \{128, 512, 1024, 2048\}$):** Evaluated on MoE-Suite. Demonstrates that prefill context length expands attention KV-cache memory footprints without altering MoE expert locality or routing traces.
3. **Lookahead Horizon ($H \in \{1, 2, 3\}$):** Evaluated on Workload A1. Increasing horizon from $H=1$ to $H=3$ extends lead time but increases prediction difficulty; at nominal hardware, DDR5 bandwidth remains the gating factor.
4. **Candidate Budget ($B \in \{1, 2, 4\}$):** Evaluated on Workload A1. Proves that lower candidate budget ($B=1, 2$) avoids transfer queue congestion (queue wait drops from $2,171\text{ ms}$ to $0.0\text{ ms}$ at $M=12$), reduces speculative DDR5 traffic by $77.7\%$, and achieves superior latency hiding.
5. **Synthetic On-Chip Multiplier ($M \in \{1.0, 4.0, 8.0, 12.0, 16.0\}$):** Evaluated on Workload A1 across 15 runs. Demonstrates that timely resident hits emerge once the available computation window exceeds DDR5 transfer latency ($3.901\text{ ms}$), validating the underlying forward prefetch mechanism.
