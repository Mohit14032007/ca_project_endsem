# ST-MoE Figures and Visualizations

**Directory:** [`mohman/point_1/st_moe/plots/`](file:///home/mohman/ca_project/mohman/point_1/st_moe/plots/)  
**Generator:** [`LLMSimulator/scripts/generate_plots.py`](file:///home/mohman/ca_project/mohman/point_1/st_moe/scripts/generate_plots.py)  
**Formats:** Both high-resolution PNG (300 DPI) and publication-ready vector PDF are generated for every figure.

---

## Figure Catalog

### 1. `recall_comparison.png` / `.pdf`
- **Title:** ST-MoE Expert Prediction Recall (Primary Standard: Online Within-Request)
- **Data Source:** [`results/csv/online_policy_comparison.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/online_policy_comparison.csv)
- **Content:** Compares micro-recall@8 for the two authoritative workloads (`MoE Anywhere` / Sage-A1.5 and `MoE Suite` / GLM-5.2) across lookahead horizons $H=1$ (next adjacent layer) and $H=2$ (skip layer).
- **Key Finding:** ST-MoE achieves **47.91%** recall on `moe_anywhere` ($H=1$) and **26.53%** on `moe_suite` ($H=1$). Degradation at $H=2$ is modest (-1.71 pp on Sage, -0.63 pp on GLM).

---

### 2. `frozen_vs_online.png` / `.pdf`
- **Title:** ST-MoE Recall Scaling Across Prediction Budgets (Frozen vs. Online)
- **Data Source:** [`results/csv/fixed_budget_summary.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/fixed_budget_summary.csv)
- **Content:** Dual-panel curve comparing recall as prediction budget $B$ increases from 1 to 8. Compares static offline profiling (`frozen`) against continuous online adaptation (`online_across_requests`), with the authoritative isolated benchmark (`online_within_request`) highlighted at $B=8$.
- **Key Finding:** Intra-request online adaptation accounts for >78% of total online gain over the frozen baseline (+5.52 pp on Sage-A1.5).

---

### 3. `horizon_comparison.png` / `.pdf`
- **Title:** Lookahead Horizon Degradation ($H=1$ vs. $H=2$)
- **Data Source:** [`results/csv/online_policy_comparison.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/online_policy_comparison.csv)
- **Content:** Side-by-side grouped bar chart illustrating recall retention across 1-step and 2-step lookahead horizons under both frozen and online adaptation regimes.
- **Key Finding:** Spatial correlation across adjacent layers is stronger than across 2-step skips, but ST-MoE retains >96% of its relative predictive accuracy across horizons.

---

### 4. `hit_rate_comparison.png` / `.pdf`
- **Title:** ST-MoE Hit Rate vs. Prediction Budget ($H=1$)
- **Data Source:** [`results/csv/fixed_budget_summary.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/fixed_budget_summary.csv)
- **Content:** Plots the hit rate ($\mathbb{P}(\text{Hits} \ge 1)$) across budgets $B \in \{1, 2, 4, 6, 8\}$, with the reported paper claim anchor of $\sim 85\%$ indicated as a reference line.
- **Key Finding:** Hit rate reaches **96.94%** on `moe_anywhere` ($B=8$) and **89.84%** on `moe_suite` ($B=8$). This strongly indicates that the paper's colloquial "85% prediction accuracy" corresponds to a hit rate metric rather than full Top-$k$ expert recall (47.91%).

---

### 5. `cold_vs_warm_recall.png` / `.pdf`
- **Title:** Cold-Start vs. Steady-State Recall
- **Data Source:** [`results/csv/online_policy_comparison.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/online_policy_comparison.csv)
- **Content:** Breaks down prediction accuracy during initial prompt tokens (cold-start, tokens 0–4) versus subsequent generation tokens (steady-state, tokens 5+) across Frozen, Online Within-Request, and Online Across-Requests.
- **Key Finding:** In cold-start, temporal history is sparse, dropping recall to ~31.85%. Once history populates, steady-state recall rises to 48.47%–50.01%.

---

### 6. `native_threshold_tradeoff.png` / `.pdf`
- **Title:** ST-MoE Native Thresholding Tradeoff: Set-Size Dilation & Precision Collapse
- **Data Source:** [`results/csv/native_threshold_results.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/native_threshold_results.csv)
- **Content:** Directly visualizes the mathematical pathology of the paper's native thresholding rule ($\text{Score} \ge \tau = 2$). Compares the additive equation 1 interpretation ($\text{Score} = \text{CCT} + \text{HT}$) against the intersection interpretation ($\mathbb{I}(\text{CCT}) + \mathbb{I}(\text{HT})$).
- **Key Finding:** Additive thresholding produces severe set-size dilation (mean predicted set size $|P| = 20.88$ to $32.02$ experts), diluting precision to 18.9%–27.8%. The intersection interpretation controls set size ($|P| \approx 4.1$ to $5.4$) but suffers from lower recall (27.7%–31.1%).

---

### 7. `profiling_sensitivity.png` / `.pdf`
- **Title:** Offline Profiling Sensitivity (MoE Anywhere, $H=1$, $B=K=8$)
- **Data Source:** [`results/csv/profiling_sensitivity.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/profiling_sensitivity.csv)
- **Content:** Evaluates predictor recall as the offline profiling data fraction is varied from $10\%$ (1,285 tokens) to $100\%$ (13,364 tokens).
- **Key Finding:** Under static `frozen` evaluation, recall improves from 34.13% to 42.39% with more profiling data. However, when online adaptation is enabled, even $10\%$ profiling achieves **48.75%** recall, demonstrating that dynamic online CCT updates compensate rapidly for sparse offline profiling data.

