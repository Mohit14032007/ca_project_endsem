# Phase B Implementation Report: Paper-Faithful ST-MoE, Common Interface, and Validation

## 1. Executive Summary
Phase B of the expert predictor comparison framework has been successfully executed. We implemented a dedicated, paper-faithful reproduction of **ST-MoE** (*Zhao et al., arXiv:2606.15453v1*), established an abstract common predictor interface (`BasePredictor`) with standardized adapters for all predictors, verified strict causality and request isolation, proved exact numerical equivalence on a hand-calculated verification example, and established a 49-point regression test suite with 100% passing status.

---

## 2. Inventory of Files Created and Modified

### 2.1 Newly Created Files
1. `src/predictor/base_predictor.py`:
   - Defines abstract base class `BasePredictor`.
   - Implements standardized wrappers: `STMoELegacyWrapper`, `CSTPWrapper`.
   - Implements baseline predictors: `RandomPredictor`, `FrequencyPredictor`.
2. `src/predictor/st_moe_paper.py`:
   - Faithful reproduction of ST-MoE (Zhao et al., 2026).
   - Implements Algorithms 1, 2, 3 and Equation 1 with 2-bit saturating counters initialized to decimal 2.
3. `tests/predictor_comparison/__init__.py`: Package initialization for comparison tests.
4. `tests/predictor_comparison/test_st_moe_paper.py`: 10 comprehensive unit tests for CCT, HT, Algorithm 2, and Algorithm 3.
5. `tests/predictor_comparison/test_predictor_hand_example.py`: 4 tests validating the deterministic 6-expert hand calculation.
6. `tests/predictor_comparison/test_predictor_interface.py`: 5 tests validating polymorphism, budget handling, and CSTP score preservation.
7. `tests/predictor_comparison/test_predictor_request_isolation.py`: 2 tests verifying independent temporal state across interleaved requests.
8. `tests/predictor_comparison/test_predictor_causality.py`: 3 tests verifying strict causal sequencing and idempotence.
9. `docs/predictor_comparison/INITIAL_GIT_STATE.txt`: Initial commit and working-tree status.
10. `docs/predictor_comparison/00_REPOSITORY_AUDIT.md`: Phase A repository audit and component reusability matrix.
11. `docs/predictor_comparison/PHASE_B_INITIAL_STATE.md`: Phase B starting baseline record.
12. `docs/predictor_comparison/01_ST_MOE_PAPER_FIDELITY.md`: Deep technical fidelity analysis, equations, ambiguities, hand calculation, and coverage matrix.
13. `docs/predictor_comparison/PHASE_B_IMPLEMENTATION_REPORT.md`: This comprehensive implementation report.
14. `docs/predictor_comparison/PHASE_B_TEST_REPORT.md`: Detailed test execution report.

### 2.2 Modified Files
1. `src/predictor/__init__.py`: Updated exports to expose `BasePredictor`, `STMoEPaperPredictor`, `STMoELegacyWrapper`, `CSTPWrapper`, `RandomPredictor`, `FrequencyPredictor`, and underlying classes.

---

## 3. Predictor Architecture & Common Interface

### 3.1 Common Predictor Interface (`BasePredictor`)
All predictors expose a unified lifecycle:
```python
class BasePredictor(ABC):

    @abstractmethod
    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Clears request-specific temporal tracking state."""
        pass

    @abstractmethod
    def predict(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        budget: int,
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> List[int]:
        """Returns at most budget candidate expert IDs for target_stage_id."""
        pass

    @abstractmethod
    def update(
        self,
        request_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int],
        token_id: Optional[int] = None
    ) -> None:
        """Updates internal state strictly post-routing revelation."""
        pass
```

### 3.2 Predictor Hierarchy and Separation of Identities
The evaluation pipeline recognizes three distinct primary predictors:
- `st_moe_paper`: Faithful to Zhao et al. (arXiv:2606.15453v1) Algorithms 1–3 and Equation 1.
- `st_moe_legacy`: Preserved Phase 1 implementation (tiered fallback priority).
- `cstp`: Proposed Dirichlet-smoothed spatio-temporal predictor with marginal prior.

---

## 4. Key Implementation Decisions

1. **Resolution of Algorithm 2 Line 7 vs. Equation 1**:
   - In Algorithm 2 Line 7, the pseudocode writes `ConfScore[h] <- 1; ConfScore[h] <- ConfScore[h] + 1`, which as a literal assignment would overwrite accumulated spatial scores from Line 5.
   - Section 4.2 text and Equation 1 explicitly define the combined score as $Score(e) = CCT(e) + HT(e)$.
   - We implement Equation 1 as primary (`ConfScore[h] += 2`), preserving both spatial and temporal contributions. A configurable parameter `literal_alg2_overwrite = False` allows testing the literal overwrite.
2. **CCT Counter Initialization**:
   - `st_moe_paper` initializes all new profiling entries and online replacement entries to decimal `2` (binary `10`, preferred state), matching Algorithm 1 Line 15 and Algorithm 3 Line 14.
3. **Deterministic Tie-Breaking**:
   - When ranking candidates for budget truncation or replacement, identical confidence scores are broken deterministically by ascending expert ID.
4. **Preservation of CSTP**:
   - `CSTPWrapper` passes predictions and updates directly to `CSTPPredictor` without modifying any Dirichlet smoothing parameters, marginal priors, or weighting formulas.
5. **Zero Ternary Operators**:
   - AST validation confirms 0 ternary operators (`ast.IfExp`) across all newly written Python files.

---

## 5. Preservation and Regression Checks

- **Phase 1 Predictors & Tests**: All 25 Phase 1 tests in `tests/test_cstp.py`, `tests/test_st_moe.py`, `tests/test_evaluation.py`, and `tests/test_trace_parser.py` continue to pass without modification.
- **Phase 2A C++ Execution Cache**: All 9 unit test checks in `./build/test_configurable_execution_cache` pass with exit code 0.
- **Historical Traces & Results**: `results/` directory remains completely untouched.

