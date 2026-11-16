<USER_REQUEST>
# MASTER ANTIGRAVITY PROMPT — FINAL REPORT NUMERICAL, MATHEMATICAL, GRAPH–TABLE AND TEXT CONSISTENCY CORRECTION

## Project
**MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference**

## Repository
`https://github.com/Mohit14032007/ca_project_endsem`

## Input report
- `MoE_Memory_Wall_Final_Report_CORRECTED.docx`
- `MoE_Memory_Wall_Final_Report_CORRECTED.pdf`

The currently uploaded versions are the documents to correct. Preserve the existing report structure, technical depth, formatting style, and chapter organization wherever possible.

---

# 1. PRIMARY OBJECTIVE — READ CAREFULLY

Your task is to eliminate **all numerical, mathematical, graph-versus-table, caption-versus-graph, and narrative-versus-result discrepancies** in the final report.

The experiments, source implementations, data extraction, and verification work have already been performed. Treat the existing verified experiment outputs as authoritative unless a clear internal inconsistency is discovered in the stored result artifacts themselves.

This is primarily a **final report consistency correction task**, not a new experimental research task.

The most important requirement is:

> Every numerical value displayed in a graph must match the corresponding stored result table/CSV. Every value stated in the report must match the graph and stored result table. Every formula must calculate correctly and use consistent units and definitions.

The report currently contains some situations where:
- A graph shows one set of values but its caption describes another.
- The graph's X-axis represents different parameters from the values discussed in the paragraph.
- A table and the graph appear to come from different experiment campaigns.
- A percentage is described as recall in one place and hit rate or coverage elsewhere.
- The numerical result is correct but its interpretation or denominator is wrong.
- A mathematical calculation does not reproduce the displayed numerical result.
- Values belonging to different workloads, bandwidth configurations, or experiment campaigns have been combined.

Fix these discrepancies systematically.

## Critical restrictions

### Do not redo the verification or experiment audit

Do NOT:
- Re-run the complete research verification process.
- Re-audit the underlying research claims from scratch.
- Re-run all 191 experiments.
- Reconstruct the entire dataset from raw traces unless a specific stored result file is demonstrably inconsistent.
- Search the report for evidence that its own numbers are correct.
- Treat old report text, existing captions, or existing figure interpretations as evidence for numerical values.
- Change verified numbers just to make them agree with an existing paragraph.
- Invent replacement numbers.
- Estimate values visually from charts when the exact values exist in the stored CSV.
- Copy values from an older report version when a current result table is available.
- Mix values from different experimental campaigns.

### What you must use

Use this source-of-truth priority:

1. **Authoritative stored result CSVs and reconciled numerical tables** for each experiment campaign.
2. Existing figure-generation scripts and their actual input datasets.
3. Existing experiment-specific result summaries or manifests, where needed to identify the correct campaign and metric.
4. The existing report only as the document to be corrected.

The report itself must never be used as the source for correcting its own numerical values.

If a stored CSV is already designated as the consolidated or reconciled master dataset, use that dataset instead of an older intermediate CSV.

Do not overwrite raw experiment outputs or established result tables merely to make the document consistent. Correct the report-generation pipeline, plot scripts, and report content as necessary.

---

# 2. REQUIRED WORKFLOW

Perform the work in the following phases. Complete every phase before declaring the task finished.

## PHASE 0 — Identify the final report and establish the correction workspace

1. Locate the uploaded corrected DOCX and PDF.
2. Identify the report-generation scripts and templates in the repository.
3. Identify the existing final report output directory.
4. Identify the current figure files used by the report.
5. Identify the authoritative result tables and existing plot scripts for Points 1–7.
6. Determine which document builder currently generates the supplied report.
7. Preserve a backup of the input report and existing report-generation files before making changes.

Likely repository areas include:

- `mohman/final_report/`
- `mohman/final_report_v2/`
- `mohman/point_1/`
- `mohman/point_3/`
- `mohman/point_4/`
- `mohman/point_5/`
- `mohman/point_6/`
- `mohman/point_7/`
- `LLMSimulator/`

Use the actual repository structure. Do not assume that paths mentioned in the report are necessarily current paths.

### Important

There may be multiple report builders and older report versions. Identify the builder responsible for the supplied corrected report. Do not accidentally edit an obsolete report builder while leaving the actual final report unchanged.

---

# 3. PHASE 1 — CREATE A MASTER NUMERICAL RECONCILIATION MATRIX

Before editing the DOCX, create a complete reconciliation matrix.

Create:

`mohman/final_report_v2/audit/REPORT_NUMERICAL_RECONCILIATION.csv`

The matrix must contain one record for every numerical claim, table value, graph series, and mathematical result that appears in the report and is relevant to Points 1–7.

At minimum, use these columns:

| Column | Description |
|---|---|
| `report_location` | Chapter, section, table/figure, and page |
| `item_type` | Table, graph, equation, caption, paragraph, executive-summary claim |
| `item_id` | Figure/table/equation identifier |
| `metric_name` | Exact metric name |
| `experiment_point` | Point 1–7 |
| `experiment_campaign` | Exact campaign or dataset |
| `workload_id` | W1–W7 or relevant workload identifier |
| `configuration` | Baseline, ST-MoE, CSTP, Config A/B/C/D, etc. |
| `parameters` | B, P, T, H, K, W, bandwidth, seed, and other relevant configuration |
| `source_csv` | Authoritative result table |
| `source_column` | Exact column or calculation source |
| `source_row_filter` | Row identification or filtering rule |
| `authoritative_value` | Numerical value from source |
| `report_value` | Value currently written or plotted |
| `unit` | Exact unit |
| `difference` | Report value minus authoritative value |
| `status` | MATCH / MISMATCH / WRONG_CAMPAIGN / WRONG_UNIT / WRONG_METRIC / NEEDS_LABEL |
| `corrected_location` | Updated output file or report location |
| `notes` | Explanation of correction |

This should be a real machine-readable audit artifact, not a manually written summary.

## 3.1 Separate experiment campaigns

This is extremely important.

Never combine values across Points 3, 5, 6, and 7 just because they involve the same model or predictor.

Each point has its own experiment scope and must retain its own configuration.

Examples:

- Point 3 large-workload experiments use the Point 3 dataset and its own configured bandwidth.
- Point 5 four-way comparisons use the Point 5 comparison campaign.
- Point 6 lookahead, prompt-length, batch-size, and K-sensitivity experiments use their own consolidated sensitivity results.
- Point 7 energy/EDP results use the Point 7 energy ledger and component tables.

Do not copy a value from one point into another point's table unless the underlying experiment ID and configuration prove that they represent the same run.

### Explicit known example: 6.717 seconds

The report must distinguish:

- Point 3 W7 result: the large-workload experiment reports a 6.717-second execution-time difference between baseline and CSTP.
- Point 6 lookahead experiment: H=1, H=2 and H=4 have their own results for the Point 6 anchor workload.

Do not use the Point 3 W7 number as a Point 6 lookahead result.

For Point 6, use the actual rows in:

`mohman/point_6/final_results/consolidated_data/lookahead_comparison.csv`

or the current authoritative Point 6 master dataset if the repository designates a newer consolidated file.

---

# 4. PHASE 2 — FIX ALL FIGURE–TABLE–TEXT DISCREPANCIES

This is the highest-priority phase.

Perform a complete audit of **every graph and every result table across the entire report**, not only the figures that were identified in earlier feedback.

The report advertises 32 figure artifacts. Check every one.

## 4.1 The mandatory three-way consistency rule

For each figure:

**Stored numerical data → Graph → Caption and explanatory text**

All three must agree.

For each figure:

1. Identify the authoritative result CSV.
2. Identify the actual source columns used.
3. Identify the exact filtering/grouping/aggregation performed.
4. Independently calculate the plotted values.
5. Compare those values against the plotted bars, lines, markers, and labels.
6. Correct the plotting script if the chart is using the wrong subset, wrong column, wrong aggregation, wrong unit, or stale hardcoded values.
7. Regenerate the figure.
8. Update the caption.
9. Update the explanatory paragraph.
10. Check that the title, X-axis, Y-axis, legend, units, and plotted workload all agree.

Do not manually edit chart screenshots to hide inconsistent numbers. Fix the source plotting data or script and regenerate the original chart.

## 4.2 Every plotted series needs traceable values

For every graph, record its complete plotted data in a machine-readable form.

Create:

`mohman/final_report_v2/audit/figure_data/`

Store one CSV or JSON data export for every figure, with the exact values used to generate the graph.

For example:

- `figure_3_1_data.csv`
- `figure_3_2_data.csv`
- `figure_3_3_data.csv`
- `figure_5_1_data.csv`
- `figure_5_7_data.csv`
- `figure_7_3_data.csv`
- `figure_8_3_data.csv`
- `figure_9_1_data.csv`

Continue for every report figure.

Each figure-data file should preserve:
- X-axis values
- Series names
- Y-axis values
- Error bars or confidence intervals, if present
- Units
- Workload and configuration identifiers
- Source dataset reference
- Aggregation rule

This enables the reader to reproduce the exact graph from its underlying data.

## 4.3 Use exact values, not visual approximations

Do not use the image pixels to estimate the numerical value of a graph when the underlying CSV is available.

The graph must be generated from the authoritative data.

If a chart contains hardcoded values, replace them with values read from the correct source table.

Avoid maintaining one set of values inside the plotting code and a different set inside the report text.

---

# 5. PHASE 3 — POINT-WISE MATHEMATICAL AND NUMERICAL CORRECTIONS

Perform the following checks point by point.

## POINT 1 — TRAINING-FREE EXPERT PREDICTION

Relevant report sections:
- Chapter 3
- Figures 3.1–3.5
- Tables 3.1 and 3.2
- Appendices containing predictor equations

### A. Check the worked CSTP numerical example

The report uses:
- E = 4 experts
- U = {1, 3}
- M = [10, 30, 40, 20]
- Spatial counts for source expert 1: [5, 0, 15, 0]
- Spatial counts for source expert 3: [0, 10, 10, 0]
- Temporal counts: [2, 4, 18, 6]
- λ = 1
- α = 0.5

Recalculate every intermediate probability and final score directly.

Check:
- Prior normalization
- Dirichlet denominator
- Every conditional probability
- Mean spatial probability
- Mean temporal probability
- Composite score
- Ranking
- Top-K selection

Use adequate numerical precision internally. Round only for final presentation.

If an intermediate number rounds differently, ensure the final result is consistent with the displayed precision.

### B. Distinguish prediction metrics

The report defines:
- Recall@K
- Precision@K
- HitRate@K
- ExactSetAccuracy

Ensure every table and graph uses the actual metric produced by its source.

Never call a hit rate recall.

Never call an HBM resident hit predictor accuracy.

Clearly distinguish:
- Top-K prediction recall
- Top-K prediction precision
- Hit rate
- Exact set accuracy
- Prefetch coverage
- Timely prefetch hit rate

They are not interchangeable.

### C. Check Table 3.2

Reconcile all dataset-level values and the aggregate result.

The aggregate must be calculated using the actual underlying aggregation method.

Do not assume that taking the simple arithmetic average of three dataset-level percentages reproduces the aggregate. Check the actual denominator and whether the calculation is micro-averaged, macro-averaged, prompt-averaged, or demand-averaged.

If 82.02% and 68.12% are retained, verify their exact definition and ensure all references use that same metric.

### D. Figures 3.1–3.5

For every graph:
- Verify X-axis profiling percentages.
- Verify Y-axis metric.
- Verify K and H labels.
- Verify benchmark names.
- Verify online/frozen evaluation modes.
- Verify cold-start/warm-state definitions.
- Verify the ST-MoE threshold experiment's actual plotted variables.

In particular, remove stale descriptions that do not match the displayed axis or plotted data.

If the graph is not actually a profiling sweep, do not describe it as a profiling sweep.

If the graph is not a threshold sweep, do not call its X-axis a confidence threshold.

### E. Remove metric and algorithm terminology conflicts

Use one consistent predictor name:

**Correlation-based Spatio-Temporal Predictor (CSTP)**

Do not alternate it with "Context-Sensitive Temporal Predictor".

The written temporal mechanism must match the formula and implementation presented in the report.

If the displayed formula uses previous-token transition history, do not claim exponential temporal decay unless that is genuinely part of the implemented formula and the current result artifacts.

---

## POINT 2 — PREFETCH IMPLEMENTATION AND TIMING MATHEMATICS

Relevant report sections:
- Chapter 4
- Tables 4.1–4.3
- Table 4.2 worked timing example

Do not start another implementation audit. This phase is concerned with the consistency of the presented equations and numerical example.

### A. Transfer-duration calculation

Check all occurrences of:
- Expert tensor size
- Bytes
- MiB/MiB conversion
- GB versus GiB
- Bandwidth
- Transfer duration
- Completion timestamp

Apply:

\[
T_{\text{transfer}} =
\frac{S_{\text{bytes}}}{BW_{\text{bytes/s}}}
\]

Convert units explicitly.

The report currently quotes:
- 84 MiB
- 88,080,384 bytes
- 22.578 GB/s
- 64 GB/s

Check every combination actually used in each campaign.

For the stated example:

\[
\frac{88,080,384}{22.578 \times 10^9}
\approx 0.003901\text{ s}
=3.901\text{ ms}
\]

Therefore, do not retain a 3.720 ms result for this exact byte count and bandwidth.

However, do not replace every occurrence of 3.720 ms with 3.901 ms indiscriminately. A different expert size, byte convention, or bandwidth configuration may legitimately produce a different duration.

Use the exact parameters of each example.

### B. Table 4.2 timing sequence

Recalculate every timestamp in the worked execution trace.

Check:
- Prefetch issue time
- Bus availability
- Transfer start time
- Transfer duration
- Transfer completion time
- Second transfer start time
- Second transfer completion time
- Demand arrival time
- Remaining transfer time
- Compute overlap
- Exposed stall
- Net latency saved

Use consistent time units throughout.

The timeline must satisfy:

\[
t_{\text{completion}} =
t_{\text{start}}+T_{\text{transfer}}
\]

For a late-attached transfer:

\[
T_{\text{remaining}} =
\max(0,t_{\text{completion}}-t_{\text{demand}})
\]

If compute overlaps an in-flight transfer, describe that overlap mathematically and do not subtract it twice.

### C. Correct all demand-outcome labels

Clearly distinguish:
1. Ordinary HBM resident hit.
2. Completed timely prefetch hit.
3. Late in-flight attachment.
4. Demand miss.

An ordinary resident hit must not automatically be labelled as a timely speculative prefetch hit.

Captions and metrics must use the same classification.

### D. Correct the final timing calculation

Ensure the displayed total duration, exposed stall, and latency saved are mathematically consistent.

If:

\[
T_{\text{baseline}} =
T_{\text{compute}}+T_{\text{demand stall}}
\]

and

\[
T_{\text{prefetch}} =
T_{\text{compute}}+T_{\text{exposed stall}}
\]

then the latency saving is:

\[
\Delta T =
T_{\text{baseline}}-T_{\text{prefetch}}
\]

Do not claim that the entire transfer time was hidden when only part of the transfer was overlapped.

---

## POINT 3 — PREFETCHING EFFECTIVENESS

Relevant report sections:
- Chapter 5
- Table 5.1
- Figures 5.1–5.14

Authoritative campaign:
- Point 3 consolidated results
- Point 3 original and reconciled result tables
- Point 3 figure scripts and their corresponding datasets

Use the current repository's authoritative consolidated dataset, expected under:

`mohman/point_3/final_results/data/consolidated_csv/`

### A. Table 5.1

Recalculate every derived column:
- Baseline time
- CSTP time
- Stall reduction
- CSTP traffic overhead
- ST-MoE traffic overhead
- Percentage improvement

For execution time:

\[
\Delta T=T_{\text{baseline}}-T_{\text{CSTP}}
\]

For runtime reduction:

\[
R_{\text{runtime}} =
\frac{T_{\text{baseline}}-T_{\text{CSTP}}}
{T_{\text{baseline}}}\times100
\]

For exposed-stall reduction:

\[
R_{\text{stall}} =
\frac{S_{\text{baseline}}-S_{\text{CSTP}}}
{S_{\text{baseline}}}\times100
\]

Do not mix runtime reduction and exposed-stall reduction.

If the table lists the 4.91% as total runtime reduction, label it as such.

### B. Workload consistency

For each W1–W7 record, ensure:
- Batch size matches the experiment.
- Sequence length matches the experiment.
- Baseline, ST-MoE and CSTP rows are correctly paired.
- The graph plots the same workload and configuration.
- The paragraph refers to the same experiment.
- The plotted execution time and stall reductions match the source table.

Check especially that titles do not use the wrong workload parameters.

### C. Figures 5.1–5.6

For each workload graph, compare all displayed bars/series with the corresponding table values.

For Figure 5.6, calculate the aggregate from the same W1–W7 rows. Ensure the meaning of each axis is correct.

### D. Figures 5.7–5.14

These figures are especially important because captions, narrative interpretations and plotted values can become disconnected.

For each one:
- Verify the actual X-axis values.
- Verify every plotted series.
- Verify the Y-axis metric.
- Verify the legend.
- Verify the unit.
- Verify workload filtering.
- Verify the aggregation.
- Verify the paragraph's endpoints and quoted percentages.

Specific metrics requiring careful treatment:

**Figure 5.7 — Coverage vs batch size**

Use the actual batch-size sweep values in the source dataset. Do not state B=1 to 128 unless those are the actual plotted values.

**Figure 5.8 — Coverage vs sequence length**

Use the actual T values present in the source result table. Do not narrate a 32–512 sweep unless the chart contains those sequence lengths.

**Figure 5.9 — Unused prefetch rate**

Ensure the denominator is precisely defined.

For example, completed speculative transfers that never served a demand are not necessarily the same as all issued prefetches that were unused. Follow the existing stored metric's exact definition.

**Figure 5.10 — Wasted speculative traffic**

Ensure byte conversion is correct:

\[
1\text{ GB}=10^9\text{ bytes}
\]

\[
1\text{ GiB}=2^{30}\text{ bytes}
\]

Do not label GB values as GiB.

**Figure 5.11 — Demand versus speculative DDR5 traffic**

Check:
- Demand traffic
- Speculative traffic
- Total traffic
- Traffic overhead
- Percentage denominator

Ensure the graph and written explanation distinguish reduced demand traffic from total bus traffic.

**Figure 5.12 — Stall reduction**

Ensure the plot actually shows the same metric described in the caption.

Do not describe a percentage of total runtime saved as a percentage of exposed stall eliminated.

**Figure 5.13 — Expert-load latency**

Check whether the chart plots ns, µs or ms and ensure the explanatory text uses the same unit.

If values are normalized per expert, state the exact denominator.

**Figure 5.14 — Decode-step latency**

Ensure the chart title, caption, plotted data and paragraph refer to the same workload scope.

Do not label a single-workload plot as an all-workload comparison.

### E. Timely versus late prefetching

Make the definitions consistent across:
- Executive summary
- Point 3 tables
- Figures
- Point 6 sensitivity
- Conclusion

Keep separate counters for:
- Ordinary resident hits
- Completed timely prefetches
- Late in-flight attachments
- Uncovered demand misses
- Unused speculative transfers
- Evicted or cancelled speculative transfers, if applicable

Do not combine them into one generic "timely coverage" value.

---

## POINT 4 — FA-LRU CACHE MANAGEMENT

Relevant report sections:
- Chapter 6
- Table 6.1
- Figure 6.1
- FA-LRU comparisons elsewhere in the report

Focus on numerical consistency.

### A. Table 6.1

Verify:
- Cache capacity
- Candidate window size W
- LRU hit rate
- FA-LRU hit rate
- Hit-rate delta
- Protected-event counts
- Reuse counts
- Traffic reduction

The percentage difference must be calculated correctly.

If FA-LRU performs worse for a particular workload, retain that negative result. Do not change the result or force a positive interpretation.

### B. Figure 6.1

Ensure every plotted data point comes from the corresponding table/source CSV.

If the chart displays hit-rate improvement, use:

\[
\Delta H=H_{\text{FA-LRU}}-H_{\text{LRU}}
\]

Do not confuse absolute percentage-point differences with relative percentage improvements.

### C. Interpret diagnostic counters accurately

Keep the meaning of:
- Protected decisions
- Protected demand-verified entries
- Unique protected expert IDs
- Reuses
- Avoided reloads

Do not present a diagnostic counter as a number of unique experts or avoided DDR5 reloads unless that is exactly how it is computed.

---

## POINT 5 — FOUR-WAY ARCHITECTURAL COMPARISON

Relevant report sections:
- Chapter 7
- Table 7.1
- Figures 7.1–7.3

Authoritative source:

`mohman/point_5/extracted_data/unified_comparison.csv`

Use associated reconciled Point 5 data and figure-source tables where the master dataset has derived fields.

### A. Four configurations

Preserve the exact architectural meanings of:
- Config A
- Config B
- Config C
- Config D

Make sure every graph and table maps each configuration consistently.

### B. Execution-time calculations

For each configuration calculate:

\[
\Delta T_A=T_A-T_X
\]

\[
R_X=\frac{T_A-T_X}{T_A}\times100
\]

State the chosen baseline explicitly.

### C. DDR5 traffic reconciliation

For every configuration:

\[
T_{\text{total traffic}} =
T_{\text{demand}}+T_{\text{speculative}}
\]

Where applicable, check whether additional traffic categories exist and should also be included.

Compare:
- Table 7.1
- Figure 7.3
- Explanatory text
- Any executive-summary traffic claims

The sum of the demand and speculative components must match the reported total under the dataset's stated accounting convention.

Do not combine a demand-only total with a demand-plus-speculative total.

Do not mix bytes, GB and GiB.

### D. FA-LRU incremental contribution

Figure 7.1 must isolate the incremental effect of cache replacement from the prefetching effect itself.

Ensure the comparison is between matched configurations and identical campaign settings.

Do not describe a very small improvement as a large architectural performance gain.

---

## POINT 6 — SENSITIVITY ANALYSIS

Relevant report sections:
- Chapter 8
- Tables 8.1 and 8.2
- Figures 8.1–8.6

Authoritative results should come from:

`mohman/point_6/final_results/consolidated_data/`

Use the actual master CSV and corresponding reconciled sweep tables.

### A. Keep all sensitivity parameters distinct

Correctly distinguish:
- B = batch size
- P = prompt/context length
- T = decode sequence length
- H = prediction lookahead horizon
- K = prefetch candidate budget
- W = FA-LRU inspection window

Never substitute one parameter for another.

### B. Table 8.1 — Lookahead sensitivity

Reconcile the H=1, H=2 and H=4 rows.

Ensure each row belongs to the same anchor workload.

Check:
- Execution time
- Exposed stall
- Stall hidden
- Late attachments
- Wasted speculative traffic
- Prediction horizon
- Traffic overhead

Use the Point 6 experiment rows only.

The known Point 6 anchor is described by the corresponding sweep configuration, but the exact authoritative rows and values must be loaded from the current consolidated dataset.

### C. Table 8.2 — Candidate budget

Recalculate every traffic and latency metric across K values.

Ensure:
- Demand traffic and speculative traffic remain separate.
- Total traffic equals the relevant component sum.
- Prefetch budget is not confused with the number of actually submitted transfers.
- Percentages use the same baseline and denominator.
- Figures reflect the exact stored rows.

### D. Figures 8.1–8.6

**Figure 8.1:** Verify exposed stall versus batch size.

**Figure 8.2:** Verify per-token generation latency, including whether the denominator is batch tokens, generated tokens, requests, or decode steps.

**Figure 8.3:** Recalculate KV-cache footprint from the exact model assumptions and configuration. Check bytes, GiB/GB conversion, sequence length, layer count and batch dimension.

**Figure 8.4:** Ensure the plotted attention-computation curve really matches the stated mathematical scaling law. Do not claim quadratic growth if the data or model formula does not show that behavior.

**Figure 8.5:** Distinguish HBM resident hits from completed timely prefetches and late attachments. Do not label the wrong metric as timely prefetch coverage.

**Figure 8.6:** Ensure the graph represents the quantity described. If it plots demand and speculative traffic, do not describe it as bus utilization unless the data actually measures utilization.

### E. KV-cache equations and units

For each formula:
- Define every variable.
- Define byte precision.
- Define batch dimension.
- Define number of layers.
- Define whether KV cache is MHA or MLA.
- Use model-specific latent dimensions where applicable.
- Show the conversion to GB or GiB.

If the report uses a simplified analytical approximation, label it clearly as such and ensure the corresponding plotted data is generated from the same formula.

---

## POINT 7 — ENERGY AND ENERGY-DELAY PRODUCT

Relevant report sections:
- Chapter 9
- Tables 9.1 and 9.2
- Figures 9.1–9.3

Authoritative source:

`mohman/point_7/extracted_data/unified_energy_results.csv`

Use the reconciled component breakdown and energy traffic reconciliation files as appropriate.

### A. Energy component summation

For every reported configuration:

\[
E_{\text{total}}=
E_{\text{static DRAM}}+
E_{\text{DDR5}}+
E_{\text{HBM}}+
E_{\text{GPU MAC}}
\]

Check every component and total.

The total must equal the sum of the components within a clearly defined rounding tolerance.

If CPU predictor energy is separately modeled, explicitly state how it is treated. Do not silently omit it from the energy equation or imply it is included when it is not.

### B. Energy percentage shares

For every component:

\[
E_{\text{share},i}=
\frac{E_i}{E_{\text{total}}}\times100
\]

All component shares should sum to approximately 100%, with small differences only due to rounding.

### C. Energy savings and overhead

For every comparison:

\[
\Delta E=E_{\text{configuration}}-E_{\text{baseline}}
\]

\[
R_E=\frac{\Delta E}{E_{\text{baseline}}}\times100
\]

Negative means energy saving; positive means overhead.

Never describe energy overhead as an energy saving.

### D. EDP

Use:

\[
EDP=E_{\text{total}}\times T_{\text{execution}}
\]

Ensure units are consistent and label the resulting unit correctly.

If energy is in joules and execution time is in seconds, the direct unit is J·s. If the table reports kJ·s or MJ·s, convert explicitly.

Recalculate all EDP values from the authoritative energy and execution-time data.

### E. Distinguish K=2 and K=4 results

Do not confuse:
- The K=2 energy/EDP optimum in the energy campaign.
- The K=4 comparison configuration in the four-way architecture evaluation.

They are separate configurations and may have different execution times, traffic, and energy.

The report must state the scope of any "optimal" claim: K=2 is the best among the tested energy configurations only if the authoritative energy table supports that conclusion.

### F. Figures 9.1–9.3

Ensure plotted component breakdowns, Pareto points and EDP values match the corresponding stored energy rows.

No hand-entered chart values.

---

# 6. PHASE 4 — CROSS-REPORT MATHEMATICAL CONSISTENCY

After Point 1–7 corrections, perform a report-wide pass over all mathematical quantities.

Check the following.

## 6.1 Unit consistency

Search every chapter and appendix for:
- Bytes
- KB
- MB
- GB
- GiB
- MiB
- ns
- µs
- ms
- seconds
- Watts
- Joules
- mJ
- pJ
- nJ
- FLOPs
- TFLOPS
- bandwidth
- percentages

Never silently mix decimal and binary units.

Use:
- GB = \(10^9\) bytes
- GiB = \(2^{30}\) bytes
- MB = \(10^6\) bytes
- MiB = \(2^{20}\) bytes

Make the unit conventions explicit when needed.

## 6.2 Percentage calculations

For every reported percentage, identify:
- Numerator
- Denominator
- Baseline
- Direction of improvement
- Whether the result is a relative change or percentage-point difference

Do not treat a 10 percentage-point difference as a 10% relative improvement.

For instance, when comparing two percentages \(p_1\) and \(p_2\):

\[
\text{Percentage-point difference}=p_2-p_1
\]

\[
\text{Relative improvement}=
\frac{p_2-p_1}{p_1}\times100
\]

Use the appropriate one in the report.

## 6.3 Aggregation

For every aggregated number:
- Identify the population being aggregated.
- Identify its denominator.
- Identify whether it is a macro or micro average.
- State whether the value is a mean, sum, median, or weighted aggregate.
- Ensure the table and graph use the same method.

Do not average already-aggregated percentages unless that is the defined methodology.

## 6.4 Rounding

Internally preserve full precision.

Round only when writing report values.

Use a consistent precision policy:
- Percentages: normally 2 decimal places.
- Time values: choose precision appropriate to the scale.
- Memory traffic: use a consistent GB/GiB convention.
- Energy: normally 2 decimal places.
- EDP: use a consistent unit and precision.

Do not independently round intermediate components and then use those rounded values to compute totals.

## 6.5 Equation numbering

Check:
- All equations are numbered consistently.
- Equation references point to the correct equation.
- Variables are defined before or immediately after use.
- Subscripts and superscripts are consistent.
- Formula notation does not contradict the text or chart labels.

---

# 7. PHASE 5 — EXECUTIVE SUMMARY, CONCLUSION, AND CROSS-CHAPTER CLAIMS

After all tables and graphs are corrected, update the executive summary and conclusion.

Do not correct these sections first.

They must be regenerated from the final reconciled results.

## Executive summary

Check every number:
- Predictor accuracy
- Accuracy improvement
- W7 stall reduction
- Runtime reduction
- Coverage
- Late attachment
- Timely hit rate
- Cache benefits
- Energy saving
- EDP improvement

Each statement must identify the correct metric and experimental scope.

Do not report a result without its denominator or campaign where ambiguity is possible.

## Conclusion

The conclusion should state only what the final tables and figures support.

Preserve both positive and negative results.

Avoid claims such as:
- "Always improves performance"
- "Eliminates all memory stalls"
- "Consistently improves cache hit rate"
- "Zero energy overhead"

unless every relevant result supports the statement.

Do not exaggerate a small runtime improvement.

Keep the interpretation technically precise.

---

# 8. PHASE 6 — FIX THE REPORT-GENERATION PIPELINE

The goal is not to edit only the current DOCX.

Make the correction durable so that rebuilding the report does not reintroduce the same discrepancies.

## Required changes

1. Correct the active report builder.
2. Correct the figure interpretation source.
3. Correct stale hardcoded numerical claims.
4. Correct incorrect graph-source references.
5. Correct graph-generation scripts if they use incorrect data or labels.
6. Correct corresponding captions and narrative text.
7. Regenerate figures from the authoritative tables.
8. Regenerate tables and report sections.
9. Regenerate the final DOCX and PDF.

Avoid creating another competing builder or duplicate final-report pipeline unless necessary.

Preserve the repository's established structure and naming conventions.

## Single-source-of-truth principle

Where practical, define shared report data structures so captions, numerical summaries, and charts use the same underlying data.

For example, a metric used by both a chart and its explanatory text should be loaded from the same authoritative result table or generated figure-data export.

Do not maintain separate hardcoded numbers in:
- Figure scripts
- Captions
- Paragraphs
- Executive summary
- Conclusion

If one source value changes, the output should be regenerated consistently.

---

# 9. PHASE 7 — AUTOMATED CONSISTENCY TESTS

Implement automated tests for the final report's numerical consistency.

Create:

`mohman/final_report_v2/audit/validate_report_consistency.py`

This script must check the following.

## Test 1 — Figure-data provenance

Every report figure must have:
- A source dataset.
- A source column or calculation.
- A recorded filter.
- A machine-readable figure-data export.

Fail if a figure uses unexplained hardcoded numerical data.

## Test 2 — Graph versus table

For all figures whose values also appear in tables, compare the generated figure-data values with table values.

Use numerical tolerance only for plotting serialization/rounding, not to excuse mismatched values.

## Test 3 — Derived calculations

Recalculate:
- Transfer durations
- Completion times
- Stall reductions
- Runtime reductions
- Hit-rate differences
- Traffic totals
- Energy totals
- Energy shares
- EDP
- KV-cache sizes

Fail if a displayed result does not match its formula and inputs within the stated tolerance.

## Test 4 — Experiment campaign identity

Fail if a graph or table combines experiment rows from distinct campaigns without an explicitly justified and documented aggregate.

## Test 5 — Metric naming

Check for inconsistent use of:
- Recall
- Precision
- Hit rate
- Coverage
- Timely hits
- Late attachment
- Resident hits
- Unused prefetch rate
- Stall reduction
- Runtime reduction
- Traffic overhead
- Energy savings
- EDP improvement

Do not rely solely on keyword matching. Use the metric's actual source definition and denominator.

## Test 6 — Unit consistency

Flag cases where:
- GB and GiB are mixed.
- MB and MiB are mixed.
- ns, µs and ms are inconsistent.
- Energy units are inconsistent.
- Bandwidth conversion is incorrect.

## Test 7 — Report claims

Check that numbers in the executive summary and conclusion match the corresponding final table or chart data.

## Test 8 — Full graph inventory

The final report's figure inventory must be complete.

The validation script must report:
- Total expected figures.
- Figures found.
- Figures regenerated.
- Figures with source CSVs.
- Figures with figure-data exports.
- Figures passed.
- Figures failed.

## Output

The script must generate a human-readable report:

`mohman/final_report_v2/audit/REPORT_CONSISTENCY_VALIDATION.md`

Include:
- Total checks.
- Passed checks.
- Failed checks.
- Mismatched figures.
- Mismatched tables.
- Mathematical discrepancies.
- Unit discrepancies.
- Remaining unresolved issues.

The script must return a non-zero exit code if a critical numerical mismatch remains.

---

# 10. PHASE 8 — VISUAL DOCUMENT CHECK

After regenerating the DOCX and PDF, inspect the actual rendered pages.

Do not assume that a correct source file automatically produces a correct document.

Check:
- All figures are present.
- No figure has been replaced with an old version.
- Chart text is readable.
- Table rows are not clipped.
- Captions are attached to the correct charts.
- Equations render properly.
- Units and symbols display correctly.
- There are no blank pages caused by layout errors.
- Table continuation pages are understandable.
- Figure and table numbering is sequential.
- Table of contents and list of figures/tables are updated.
- Page references are correct.

Compare the rendered PDF against the regenerated DOCX, not just the source Markdown or Python files.

Inspect every page that contains a corrected chart, table, equation, or numerical claim.

Perform a final page-by-page pass through the PDF to identify accidental document-generation issues.

---

# 11. SPECIFIC DISCREPANCY TARGETS TO PRIORITIZE

The following are known high-priority areas from the current report. They are targets for reconciliation, not permission to blindly insert replacement numbers.

## A. Chapters 1–3

1. The 82.02% versus 68.12% predictor comparison must use one consistently defined metric everywhere.
2. The percentage gain must be distinguished as a percentage-point difference versus relative improvement.
3. Figures 3.1 and 3.2 need their actual X-axis and Y-axis values reconciled against captions and text.
4. Figure 3.3's cold/warm-state interpretation must correspond to the actual three plotted modes if the graph shows Frozen, Online Within-Request and Online Across-Request.
5. Figure 3.4's plotted categories and series must match the discussion of the three dataset groups and the actual broader workload chart.
6. Figure 3.5's explanation must describe its actual axes and plotted metrics, not a different threshold experiment.
7. The ST-MoE confidence thresholds and candidate sizes must not be confused with the actual chart variables.

## B. Chapter 4

1. Recalculate 84 MiB/88,080,384-byte transfer times at each specified bandwidth.
2. Recalculate Table 4.2 timestamps and transfer completions.
3. Reconcile the 3.720 ms and 3.901 ms examples by their exact input sizes and bandwidths.
4. Verify that the displayed late-attachment example's exposed stall follows its transfer completion timestamp.
5. Reconcile all time quantities in the worked example.
6. Ensure the narrative's "timely" classification is not assigned to ordinary resident hits.

## C. Chapter 5

1. Reconcile Table 5.1 and Figures 5.1–5.6 for all W1–W7 entries.
2. Check W7's 6.717-second execution-time reduction and its percentage denominator.
3. Reconcile Figure 5.7's batch sweep with the actual source rows.
4. Reconcile Figure 5.8's sequence-length sweep with the actual source rows.
5. Reconcile Figure 5.9 unused prefetch rates.
6. Reconcile Figure 5.10 byte/GB/GiB values.
7. Reconcile Figure 5.11 total, demand and speculative traffic.
8. Reconcile Figure 5.12 stall versus runtime reduction.
9. Reconcile Figure 5.13 latency units.
10. Reconcile Figure 5.14 workload scope and per-token denominator.

## D. Chapter 6

1. Reconcile LRU and FA-LRU hit-rate differences.
2. Preserve negative results.
3. Distinguish protected decisions from unique experts and avoided reloads.
4. Reconcile the scaling graph and text.

## E. Chapter 7

1. Reconcile all Config A/B/C/D execution-time values.
2. Reconcile Figure 7.1 incremental FA-LRU gains.
3. Reconcile Figure 7.2 exposed stall.
4. Reconcile Figure 7.3 demand plus speculative traffic against total traffic.
5. Check all displayed traffic percentages and units.

## F. Chapter 8

1. Check Point 6 lookahead values against the Point 6 consolidated results.
2. Keep Point 3 W7's 6.717 seconds out of Point 6 lookahead results.
3. Recalculate KV-cache values and conversions.
4. Reconcile Figure 8.4's stated scaling with the plotted curve.
5. Separate resident-hit rate from timely-prefetch coverage in Figure 8.5.
6. Ensure Figure 8.6's plotted quantity is accurately described.

## G. Chapter 9

1. Recalculate energy component sums.
2. Check energy percentage shares.
3. Check the treatment of CPU predictor energy.
4. Recalculate EDP and units.
5. Distinguish K=2 and K=4 experimental scopes.
6. Reconcile Figures 9.1–9.3 with the Point 7 tables.

---

# 12. FINAL DELIVERABLES

Produce the following corrected outputs.

### A. Corrected report

- `mohman/final_report_v2/MoE_Memory_Wall_Final_Report_CORRECTED.docx`
- `mohman/final_report_v2/MoE_Memory_Wall_Final_Report_CORRECTED.pdf`

### B. Reconciliation artifacts

- `mohman/final_report_v2/audit/REPORT_NUMERICAL_RECONCILIATION.csv`
- `mohman/final_report_v2/audit/REPORT_CONSISTENCY_VALIDATION.md`
- `mohman/final_report_v2/audit/figure_data/` — numerical data exports for all report figures

### C. Validation script

- `mohman/final_report_v2/audit/validate_report_consistency.py`

### D. Correction summary

Create:

`mohman/final_report_v2/audit/FINAL_CORRECTION_SUMMARY.md`

Include:
- Every corrected figure.
- Every corrected table.
- Every corrected equation or numerical derivation.
- Every corrected caption.
- Every corrected paragraph.
- The source CSV used for each correction.
- The type of discrepancy fixed.
- Any item that could not be reconciled.

Do not claim that all 32 figures passed unless all 32 were checked.

---

# 13. FINAL ACCEPTANCE CRITERIA — DO NOT STOP EARLY

The work is complete only when all of the following are true:

1. Every graph in the report has been checked against its stored numerical data.
2. Every graph's underlying plotted values are exported to a machine-readable file.
3. Every table agrees with its authoritative result dataset.
4. Every numerical statement in the main text agrees with the corresponding table or graph.
5. Every caption describes the actual graph.
6. Every X-axis and Y-axis label describes the actual plotted quantity.
7. All mathematical calculations reproduce their displayed results.
8. All percentage denominators are correct and documented.
9. All unit conversions are correct.
10. No experiment campaigns have been mixed accidentally.
11. No stale hardcoded values remain in the active report-generation pipeline.
12. The executive summary and conclusion have been synchronized with the final results.
13. The report has been regenerated as both DOCX and PDF.
14. The rendered PDF has been visually inspected.
15. The automated validation script passes all critical checks.
16. The final correction summary lists all changes and unresolved items.

## Final instruction

**Prioritize mathematical correctness and graph/table numerical consistency above stylistic rewriting.**

Do not make the report look more impressive by modifying verified experiment results. Do not force graphs to fit the prose. Use the authoritative stored result data to correct the graphs, and then correct the prose to match those graphs and tables.

Preserve valid results, including negative or small improvements. Where a number is correct but its interpretation is wrong, correct the interpretation rather than changing the number.

Where the stored data itself has a genuine unresolved conflict, do not guess. Mark the specific item unresolved, document the competing source rows, and do not present it as a verified final result.

The final output must be a technically consistent report in which a reader can trace every important plotted value back to a stored result table and reproduce every derived calculation.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-15T23:40:07Z.

The user's current state is as follows:
Other open documents:
- /home/mohman/ca_project/LLMSimulator/reproduction/configs/gpu_baseline.yaml (LANGUAGE_UNSPECIFIED)
- /home/mohman/ca_project/LLMSimulator/reproduction/configs/duplex.yaml (LANGUAGE_UNSPECIFIED)
- /home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/manifests/ds5c0_logical_physical_capacity_audit.md (LANGUAGE_UNSPECIFIED)
- /home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/manifests/ds5b1_ddr5_mapping_validation.md (LANGUAGE_UNSPECIFIED)
- /home/mohman/ca_project/LLMSimulator/src/dram/dram_interface.cpp (LANGUAGE_UNSPECIFIED)
</ADDITIONAL_METADATA>