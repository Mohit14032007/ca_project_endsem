# Reference Materials: ST-MoE Framework

**Directory:** [`mohman/point_1/st_moe/references/`](file:///home/mohman/ca_project/mohman/point_1/st_moe/references/)  
**Primary Paper:** [`Zhao_et_al_STMoE_2606.15453v1.pdf`](file:///home/mohman/ca_project/mohman/point_1/st_moe/references/Zhao_et_al_STMoE_2606.15453v1.pdf)

---

## Primary Citation

```bibtex
@article{zhao2024stmoe,
  title={A Spatio-Temporal Expert Prefetching Framework for Efficient MoE-based LLM Inference},
  author={Zhao, et al.},
  journal={arXiv preprint arXiv:2606.15453v1},
  year={2024}
}
```

---

## Critical Paper Sections and Reproduction Mapping

| Section in Paper | Concept / Algorithm | Implementation in Codebase |
| :--- | :--- | :--- |
| **Section 3** | Spatio-Temporal Expert Correlations | Analyzed in [`ST_MOE_METHODOLOGY.md`](file:///home/mohman/ca_project/mohman/point_1/st_moe/ST_MOE_METHODOLOGY.md) |
| **Section 4.2** | Correlation Confidence Table (CCT) & History Table (HT) | [`LLMSimulator/src/predictor/st_moe_paper.py:25-95`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py) |
| **Equation 1** | Dynamic Prediction Scoring: $\text{Score} = \text{CCT} + \text{HT}$ | Implemented with dual interpretations (`confidence_additive` vs `indicator_intersection`) |
| **Algorithm 1** | Offline CCT Profiling Calibration | Implemented in `STMoEPaperPredictor.profile()` |
| **Algorithm 2** | Dynamic Expert Prediction Strategy | Implemented in `STMoEPaperPredictor.predict()` |
| **Algorithm 3** | Dynamic Update Strategy (2-bit saturating counters) | Implemented in `STMoEPaperPredictor.update()` |
| **Section 5.3 & Fig. 6** | Claimed Prediction Accuracy ($\sim 85\%$) | Audited in [`PAPER_COMPARISON.md`](file:///home/mohman/ca_project/mohman/point_1/st_moe/PAPER_COMPARISON.md) |

---

## Key Mathematical Ambiguity Resolved

The paper does not provide an explicit mathematical equation for its reported "prediction accuracy" of approximately 85%. Our empirical investigation proves:
1. When evaluated against full Top-$k$ expert set recall ($\frac{|P \cap G|}{|G|}$), ST-MoE achieves **47.91%** on `MoE Anywhere` and **26.53%** on `MoE Suite`.
2. When evaluated under the Hit Rate metric ($\mathbb{P}(|P \cap G| \ge 1)$), ST-MoE achieves **96.94%** on `MoE Anywhere` and **89.84%** on `MoE Suite`.
3. The paper's claimed $\sim 85\%$ accuracy aligns closely with the hit-rate definition rather than complete set recall. Full details are presented in [`PAPER_COMPARISON.md`](file:///home/mohman/ca_project/mohman/point_1/st_moe/PAPER_COMPARISON.md).

