# Phase B Initial State Record

## 1. Environment & Repository Information
- **Repository Root**: `/home/mohman/ca_project`
- **Simulation/Predictor Subdirectory**: `/home/mohman/ca_project/LLMSimulator`
- **Date & Timestamp**: `2026-11-01T11:08:00Z`
- **Active Git Branch**: `main`
- **HEAD Commit**: `d5bf61ff0c066c06f72173189f9924329f613a88`
- **HEAD Commit Message**: `Finalize LLMSimulator: Add Group 13 master reference documentation and code changes guide, clean up intermediate manifests and experimental runs`

---

## 2. Working Tree Status

### 2.1 Tracked Modified Files (Preserved Phase 2A Work)
The working tree contains verified Phase 2A changes (configurable off-chip DRAM execution cache):
- `LLMSimulator/config.yaml`: Added `bypass_offchip_execution_cache: false`.
- `LLMSimulator/eval/CMakeLists.txt`: Added target executable `test_configurable_execution_cache`.
- `LLMSimulator/eval/test.cpp`: Added YAML parsing & env var check for off-chip cache toggle.
- `LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp`: Added YAML parsing & env var check for off-chip cache toggle.
- `LLMSimulator/src/hardware/hardware_config.h`: Added `bool bypass_offchip_execution_cache = false;` to `SystemConfig`.
- `LLMSimulator/src/hardware/layer_impl.cpp`: Implemented conditional execution cache bypass in `issueRamulator()` and `issueRamulatorBatch()`.

### 2.2 Untracked Files and Directories
- `LLMSimulator.zip`
- `LLMSimulator/eval/test_configurable_execution_cache.cpp` (Phase 2A 9-point unit test)
- `LLMSimulator/results/` (Protected Phase 1 experiment logs and summaries)
- `LLMSimulator/scripts/` (Evaluation and trace processing scripts)
- `LLMSimulator/src/predictor/` (Predictor implementations: CSTP, legacy ST-MoE, trace parser, evaluation harness)
- `LLMSimulator/tests/` (Phase 1 unit tests: 25 passing tests)

---

## 3. Existing Predictor Modules (Phase 1 Baseline)
- `LLMSimulator/src/predictor/__init__.py`
- `LLMSimulator/src/predictor/cstp_predictor.py`: Proposed Dirichlet-smoothed spatio-temporal predictor with marginal prior.
- `LLMSimulator/src/predictor/st_moe_predictor.py`: Legacy ST-MoE implementation (preserved under identifier `st_moe_legacy`).
- `LLMSimulator/src/predictor/trace_parser.py`: Robust parser with drift detection and integrity validation.
- `LLMSimulator/src/predictor/evaluation.py`: Causal evaluation framework with request-level isolation.

---

## 4. Existing Test Commands and Baselines

### 4.1 Python Unit Test Suite (25 Tests)
```bash
cd /home/mohman/ca_project/LLMSimulator
python3 -m unittest discover tests -v
```
**Status**: 25 tests passing in ~0.07s.

### 4.2 C++ Execution Cache Test (Phase 2A)
```bash
cd /home/mohman/ca_project/LLMSimulator
./build/test_configurable_execution_cache
```
**Status**: All 9 validation checks passing with exit code 0.

---

## 5. Preservation Commitments
1. **Preserve Phase 2A Work**: Do not revert or alter `bypass_offchip_execution_cache` logic or tests.
2. **Preserve Legacy ST-MoE**: Keep `st_moe_predictor.py` intact and accessible as `st_moe_legacy`.
3. **Preserve CSTP Mathematics**: Do not alter equations, Dirichlet smoothing, or composite scoring in `cstp_predictor.py`.
4. **Preserve Historical Artifacts**: Do not touch `results/` or historical simulation traces.
5. **Strict No-Ternary Rule**: Enforce zero ternary operators across all newly written Python and C++ code.

