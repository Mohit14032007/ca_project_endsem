# Requirement 3: Prefetching Effectiveness Evaluation

## 1. Conversion and Demand Breakdown
Evaluation across the 15-run compute-window matrix confirms the dual conservation laws:
- **Demand Conservation:** $5,179\text{ Natural Hits} + H_{\text{timely}} + A_{\text{late}} + M_{\text{unprefetched}} = 7,200\text{ Lookups}$ held for 100% of runs.
- **Transfer Lifecycle Conservation:** $\text{Submitted} = \text{Demanded} + \text{Unused} + \text{Evicted} + \text{In-Flight} + \text{Cancelled}$ held for 100% of runs.

## 2. Key Findings
1. **Nominal Hardware ($M=1.0\times$):** Transfer duration ($3.901\text{ ms}$) exceeds available computation window ($0.366\text{ ms}$) by $10.7\times$, yielding 0 timely hits and $52$--$59$ late attachments.
2. **Emergence of Timely Hits ($M \ge 4.0\times$):** Timely resident hits emerge at $M=4.0$ (1 hit, $3.90\text{ ms}$ stall reduction) and double to 2 hits at $M=12.0$ ($7.80\text{ ms}$ stall reduction).
3. **Queue Decongestion:** At $M=16.0$, queue waiting time dropped by $56\%$ (from $3,075\text{ ms}$ to $1,349\text{ ms}$), and 100% of speculative transfers completed.
