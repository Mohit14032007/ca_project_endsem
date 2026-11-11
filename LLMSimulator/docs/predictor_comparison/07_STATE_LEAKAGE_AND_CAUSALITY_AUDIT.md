# Forensic Audit of State Isolation, Contamination, and Causality

**Audit Scope**: Verification of data integrity, cross-request isolation, budget independence, and temporal causality across the offline evaluation harness and predictor classes.

---

## 1. Systematic Investigation of the 12 Forensic Questions

### Q1. Can training requests contribute to global CSTP statistics?
- **Finding**: **YES (Intended)**.
- **Code Evidence**: In `scripts/eval_predictors_offline.py:143`, `cstp_default.fit_profile(train_records)` is called. The method iterates through all training records, incrementing `stage_marginal_counts`, `spatial_counts`, and `temporal_counts`.
- **Verdict**: Valid offline training procedure.

### Q2. Can test requests contribute to global CSTP statistics?
- **Finding**: **NO (Default)** / **YES (Only if `online_adaptation=True`)**.
- **Code Evidence**: In `cstp_predictor.py:238`, the block that increments counts inside `update()` is gated by `if self.online_adaptation:`. In the default configuration, `online_adaptation` is `False`. During standard evaluation, test records never modify `stage_marginal_counts`, `spatial_counts`, or `temporal_counts`.
- **Verdict**: Zero test-to-model statistical leakage under standard offline evaluation.

### Q3. If online adaptation is enabled, are test observations used to predict later test tokens?
- **Finding**: **YES**.
- **Code Evidence**: When `online_adaptation=True`, each call to `runner.update()` after token $t$ increments transition counts. The prediction for token $t+1$ will then reflect these updated counts.
- **Verdict**: Causal online learning.

### Q4. Is online adaptation intended continual learning or unintended leakage?
- **Finding**: **Intended Causal Online Adaptation**.
- **Code Evidence**: The update occurs strictly **after** prediction metrics for token $t$ are logged (`evaluation.py:246-252`). The model never observes token $t$ before predicting token $t$. It mimics an online serving system adapting to incoming query traffic.
- **Verdict**: Valid causal adaptation, provided it is explicitly designated as "online learning" rather than "pure offline prediction".

### Q5. Can Request A's temporal history affect Request B?
- **Finding**: **NO**.
- **Code Evidence**: All temporal history is indexed by `request_id`:
  ```python
  self.history_table[request_id][stage_id] = set(actual_target_experts)
  ```
  When predicting for Request B, the predictor queries `self.history_table.get(request_id_B, {})`.
- **Verdict**: Strictly isolated. Verified by unit test `test_interleaved_requests_preserve_isolation` in [`tests/predictor_comparison/test_predictor_request_isolation.py`](file:///home/mohman/ca_project/LLMSimulator/tests/predictor_comparison/test_predictor_request_isolation.py).

### Q6. Are global statistics shared between requests?
- **Finding**: **YES (Intended)**.
- **Code Evidence**: `spatial_counts` and `cct` tables represent model architecture routing affinities across layers. These are properties of the neural network weights and tasks, which are constant across requests.
- **Verdict**: Correct architectural design. Spatial co-activations are model-level, whereas temporal histories are sequence-level.

### Q7. Are predictor objects recreated between datasets?
- **Finding**: **YES**.
- **Code Evidence**: In `eval_predictors_offline.py:103,135`, fresh instances of `STMoEPredictor` and `CSTPPredictor` are instantiated inside the dataset evaluation function. State from one dataset trace does not carry over to another.
- **Verdict**: Clean dataset isolation.

### Q8. Are predictor objects deep-copied between budgets?
- **Finding**: **YES**.
- **Code Evidence**: In `evaluation.py:206`:
  ```python
  for budget in budgets:
      runner = copy.deepcopy(predictor_instance)
  ```
- **Verdict**: Guaranteed budget isolation.

### Q9. Does changing budget alter learning state?
- **Finding**: **NO**.
- **Code Evidence**: Because each budget receives a distinct `copy.deepcopy()` clone of the post-profiled master predictor, mutations (such as ST-MoE CCT counter adjustments or history updates) occurring during the $B=1$ evaluation run have zero effect on the runner instantiated for $B=2, 4, 8, 16$.
- **Verdict**: Completely independent budget evaluations.

### Q10. Is the first prediction of a new request genuinely cold-started?
- **Finding**: **YES**.
- **Code Evidence**: In `evaluation.py:213`, `runner.reset_request_state(req)` is invoked before iterating through tokens of that request. At token $t=0$, `self.history_table.get(req, {})` is empty.
  - CSTP falls back to the target-stage marginal prior $P_{\text{prior}}(v)$.
  - ST-MoE adds $0$ temporal bonus.
- **Verdict**: True cold start.

### Q11. Does reset clear all required state?
- **Finding**: **YES**.
- **Code Evidence**: `reset_request_state(request_id)` deletes the temporal entry for that specific sequence ID. It does not erase the offline-trained spatial matrices or CCT tables.
- **Verdict**: Correctly scoped reset.

### Q12. Does any predictor retain state across separate evaluation runs?
- **Finding**: **NO**.
- **Code Evidence**: Predictor instances are local Python objects created inside execution functions. They are not stored in module-level global variables or serialized to disk unless explicitly saved.
- **Verdict**: Clean process-level lifecycle.

---

## 2. Risk, Severity, and Recommendation Matrix

| Category | Potential Contamination Path | Observed Severity | Forensic Verification | System Recommendation |
| :--- | :--- | :---: | :--- | :--- |
| **Train/Test Separation** | Training on test records | **NONE** | Disjoint request-level splits via `split_by_request()` | Maintain minimum 4 distinct requests per trace |
| **Causal Order** | Peeking at future target labels | **NONE** | Prediction precedes `update()` by 20 lines of code; tested in unit test suite | Preserve causal evaluation harness structure |
| **Interleaved Requests** | Leaking token history between sequences | **NONE** | State partitioned by dictionary key `request_id` | Keep `reset_request_state(req)` in evaluation loop |
| **Budget Coupling** | Low-budget run polluting high-budget run | **NONE** | `copy.deepcopy()` executed at outer budget loop | Maintain deep-copy cloning policy |
| **Single-Request Traces** | Attempting request splits on single-request traces | **MEDIUM** | `split_by_request()` detects $N=1$ and enforces temporal split with warning | Flag single-request datasets in reporting |

