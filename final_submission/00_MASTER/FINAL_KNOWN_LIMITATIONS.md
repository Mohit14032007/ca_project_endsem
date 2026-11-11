# Final Project Known Limitations and Methodological Distinctions
**MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference**

### 1. Simulated Device Time vs Host Wall-Clock Time
All reported latencies, execution times, exposed memory stalls, and speedups represent cycle-accurate and event-driven **simulated GPU device time** computed by the simulator's hardware models (`LinearExecutionGPU`, `Cluster`, and Ramulator2/DRAM models). Host wall-clock time is strictly the host machine duration required to run the simulation process and is separately logged in run manifests.

### 2. Synthetic On-Chip Execution-Window Multiplier vs Real GPU Benchmarking
The sensitivity evaluation across multiplier $M \in \{1.0, 4.0, 8.0, 12.0, 16.0\}$ represents a **synthetic on-chip execution-window multiplier**. It scales the analytical on-chip duration $\max(\text{compute\_duration}, \text{onchip\_memory\_duration})$ while cleanly isolating off-chip DDR5 transfer miss stalls (`weight_memory_cost`). This is an architectural sensitivity tool designed to characterize latency hiding under expanded computation windows; it must not be cited as physical GPU hardware measurements.

### 3. Native DRAM Energy Model vs Estimated DDR5 Transport Energy
The simulator's native energy model (`Cluster::exportToCSV`) tracks HBM DRAM row activations, reads, writes, and on-chip MAC compute operations. It does not natively account for off-chip DDR5 bus transfers or PCIe/CXL link transport energy. To prevent omissions, DDR5 transport energy is separately reported as a calibrated estimate ($28\text{ pJ/byte}$, or $2.466\text{ mJ}$ per $88.08\text{ MB}$ expert) and clearly distinguished from native simulator energy counters.

### 4. Demand Conservation and Transfer Invariants
All prefetch and demand events are strictly governed by conservation laws:
1. $\text{Lookups} = \text{Natural Hits} + \text{Timely Hits} + \text{Late Attachments} + \text{Unprefetched Misses} = 7,200$ (Workload A1).
2. $\text{Submitted} = \text{Demanded} + \text{Unused} + \text{Evicted} + \text{In-Flight} + \text{Cancelled}$.
Late in-flight attachments are never conflated with timely resident hits.
