#!/usr/bin/env python3
"""
generate_final_docx.py:
Builds the comprehensive, publication-quality Microsoft Word report:
mohman/final_report/MoE_Memory_Wall_Final_Report_CORRECTED.docx
Staging all 13 chapters, 15 verified native tables, authentic C++ source-code boxes,
exact mathematical derivations, and all 32 verified figures with complete 10-point explanations.
Enforces strict zero-ternary rule.
"""

import os
import sys
import shutil
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

# Import modular components
from report_sections_part1 import METADATA, TABLES_CH1_CH3, CODE_SNIPPETS_CH3
from report_sections_part2 import TABLES_CH4, CODE_SNIPPETS_CH4
from report_sections_part3 import TABLES_CH5_CH7, CODE_SNIPPETS_CH6
from report_sections_part4 import TABLES_CH8_CH13
from figure_interpretations import FIGURE_EXPLANATIONS

def set_cell_background(cell, fill_hex):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    margins = [('top', top), ('bottom', bottom), ('left', left), ('right', right)]
    for m, val in margins:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="none"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_styled_heading(doc, text, level):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.bold = True
    
    if level == 1:
        run.font.size = Pt(18)
        run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(8)
    elif level == 2:
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(0x00, 0x4D, 0x80)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
    else:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
    return p

def add_body_p(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(10.5)
    run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    return p

def add_bullet_p(doc, bold_prefix, text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(3)
    
    run_bold = p.add_run(bold_prefix + " ")
    run_bold.font.name = 'Calibri'
    run_bold.font.size = Pt(10.5)
    run_bold.bold = True
    run_bold.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
    
    run_text = p.add_run(text)
    run_text.font.name = 'Calibri'
    run_text.font.size = Pt(10.5)
    run_text.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    return p

def add_native_table(doc, table_def):
    caption = doc.add_paragraph()
    caption.paragraph_format.keep_with_next = True
    caption.paragraph_format.space_before = Pt(12)
    caption.paragraph_format.space_after = Pt(4)
    c_run1 = caption.add_run(f"{table_def['num']}: ")
    c_run1.font.name = 'Calibri'
    c_run1.font.size = Pt(10)
    c_run1.bold = True
    c_run1.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
    
    c_run2 = caption.add_run(table_def['title'])
    c_run2.font.name = 'Calibri'
    c_run2.font.size = Pt(10)
    c_run2.italic = True
    c_run2.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    headers = table_def['headers']
    rows = table_def['rows']
    num_cols = len(headers)
    num_rows = len(rows) + 1

    table = doc.add_table(rows=num_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, "D3D3D3")

    for col_idx, h_text in enumerate(headers):
        cell = table.cell(0, col_idx)
        set_cell_background(cell, "003366")
        set_cell_margins(cell, top=140, bottom=140, left=160, right=160)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h_text)
        run.font.name = 'Calibri'
        run.font.size = Pt(9.5)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for r_idx, row_data in enumerate(rows):
        fill_hex = "FFFFFF"
        if r_idx % 2 == 1:
            fill_hex = "F8FAFC"
        for col_idx, val in enumerate(row_data):
            cell = table.cell(r_idx + 1, col_idx)
            set_cell_background(cell, fill_hex)
            set_cell_margins(cell, top=100, bottom=100, left=160, right=160)
            p = cell.paragraphs[0]
            if col_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(val))
            run.font.name = 'Calibri'
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    space_p = doc.add_paragraph()
    space_p.paragraph_format.space_after = Pt(8)

def add_code_box(doc, code_def):
    header_p = doc.add_paragraph()
    header_p.paragraph_format.keep_with_next = True
    header_p.paragraph_format.space_before = Pt(10)
    header_p.paragraph_format.space_after = Pt(3)
    
    hrun1 = header_p.add_run(f"Listing: {code_def['func']} ")
    hrun1.font.name = 'Consolas'
    hrun1.font.size = Pt(9)
    hrun1.bold = True
    hrun1.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
    
    hrun2 = header_p.add_run(f"({code_def['file']})")
    hrun2.font.name = 'Consolas'
    hrun2.font.size = Pt(8.5)
    hrun2.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_background(cell, "F4F6F9")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="003366"/>'
        f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'  <w:right w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

    cp = cell.paragraphs[0]
    cp.paragraph_format.line_spacing = 1.05
    c_run = cp.add_run(code_def['code'])
    c_run.font.name = 'Consolas'
    c_run.font.size = Pt(8.5)
    c_run.font.color.rgb = RGBColor(0x1A, 0x20, 0x2C)

    desc_p = doc.add_paragraph()
    desc_p.paragraph_format.space_before = Pt(3)
    desc_p.paragraph_format.space_after = Pt(8)
    d_run = desc_p.add_run("Implementation Note: " + code_def['desc'])
    d_run.font.name = 'Calibri'
    d_run.font.size = Pt(8.5)
    d_run.italic = True
    d_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

def add_embedded_figure(doc, fig_key, image_path):
    if fig_key not in FIGURE_EXPLANATIONS:
        print(f"Warning: Missing explanation for figure {fig_key}")
        return

    info = FIGURE_EXPLANATIONS[fig_key]

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return

    # Add Figure Heading
    f_head = doc.add_paragraph()
    f_head.paragraph_format.keep_with_next = True
    f_head.paragraph_format.space_before = Pt(14)
    f_head.paragraph_format.space_after = Pt(4)
    f_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h_run = f_head.add_run(f"{info['num']} — {info['title']}")
    h_run.font.name = 'Calibri'
    h_run.font.size = Pt(11)
    h_run.bold = True
    h_run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    # Embed Picture
    p_img = doc.add_paragraph()
    p_img.paragraph_format.keep_with_next = True
    p_img.paragraph_format.space_after = Pt(4)
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_picture(image_path, width=Inches(5.6))
    
    # Caption
    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.keep_with_next = True
    p_cap.paragraph_format.space_before = Pt(4)
    p_cap.paragraph_format.space_after = Pt(6)
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c_run = p_cap.add_run(info['caption'])
    c_run.font.name = 'Calibri'
    c_run.font.size = Pt(9.5)
    c_run.italic = True
    c_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Detailed Academic Interpretation Paragraph
    p_int = doc.add_paragraph()
    p_int.paragraph_format.line_spacing = 1.15
    p_int.paragraph_format.space_after = Pt(10)
    int_prefix = p_int.add_run("Detailed Architectural Interpretation: ")
    int_prefix.font.name = 'Calibri'
    int_prefix.font.size = Pt(10)
    int_prefix.bold = True
    int_prefix.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    
    int_body = p_int.add_run(info['interp'])
    int_body.font.name = 'Calibri'
    int_body.font.size = Pt(10)
    int_body.font.color.rgb = RGBColor(0x26, 0x26, 0x26)

def build_cover_page(doc):
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(72)
    title_p.paragraph_format.space_after = Pt(10)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    trun = title_p.add_run(METADATA["title"])
    trun.font.name = 'Calibri'
    trun.font.size = Pt(24)
    trun.bold = True
    trun.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(28)
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    srun = sub_p.add_run(METADATA["extended_title"])
    srun.font.name = 'Calibri'
    srun.font.size = Pt(12.5)
    srun.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    course_p = doc.add_paragraph()
    course_p.paragraph_format.space_after = Pt(36)
    course_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    crun1 = course_p.add_run(METADATA["course"] + "\n")
    crun1.font.name = 'Calibri'
    crun1.font.size = Pt(13)
    crun1.bold = True
    crun1.font.color.rgb = RGBColor(0x00, 0x4D, 0x80)

    crun2 = course_p.add_run(METADATA["subtitle"] + " | " + METADATA["date"])
    crun2.font.name = 'Calibri'
    crun2.font.size = Pt(11)
    crun2.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    auth_table = doc.add_table(rows=1, cols=3)
    auth_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(auth_table, "E2E8F0")

    for idx, author in enumerate(METADATA["authors"]):
        cell = auth_table.cell(0, idx)
        set_cell_background(cell, "F8FAFC")
        set_cell_margins(cell, top=140, bottom=140, left=140, right=140)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        arun1 = p.add_run(author["name"] + "\n")
        arun1.font.name = 'Calibri'
        arun1.font.size = Pt(11)
        arun1.bold = True
        arun1.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
        
        arun2 = p.add_run(f"Roll No: {author['roll']}\nDept: {author['dept']}")
        arun2.font.name = 'Calibri'
        arun2.font.size = Pt(9.5)
        arun2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    inst_p = doc.add_paragraph()
    inst_p.paragraph_format.space_before = Pt(48)
    inst_p.paragraph_format.space_after = Pt(4)
    inst_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    irun = inst_p.add_run(METADATA["institution"] + "\n")
    irun.font.name = 'Calibri'
    irun.font.size = Pt(10)
    irun.bold = True
    irun.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    repo_run = inst_p.add_run("Project Repository: " + METADATA["repo"])
    repo_run.font.name = 'Calibri'
    repo_run.font.size = Pt(9.5)
    repo_run.font.color.rgb = RGBColor(0x00, 0x4D, 0x80)

    doc.add_page_break()

def generate_docx():
    output_dir = "mohman/final_report"
    docx_path = os.path.join(output_dir, "MoE_Memory_Wall_Final_Report_CORRECTED.docx")
    
    doc = Document()

    # Configure Margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
        header = s.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference | ")
        hrun.font.name = 'Calibri'
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        
        footer = s.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
        frun1 = fp.add_run("CSE 511/ECE 511 Computer Architecture (End Semester Report)")
        frun1.font.name = 'Calibri'
        frun1.font.size = Pt(8.5)
        frun1.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    # 1. Cover Page
    build_cover_page(doc)

    # 2. Executive Front Matter
    add_styled_heading(doc, "Executive Table of Contents", 1)
    toc_items = [
        ("Chapter 1: Executive Summary", "High-level overview, core findings across Points 1 to 7, verified system bounds"),
        ("Chapter 2: Project Motivation & Architecture", "The MoE memory wall, DDR5 offloading, system architecture & calibration"),
        ("Chapter 3: Point 1 — Training-Free Prediction", "Dirichlet CSTP formulation, ST-MoE saturating counters, worked 4-expert trace, 5 verified plots"),
        ("Chapter 4: Point 2 — Prefetching Implementation", "Deep architectural core: 20-step flow, LiveEngine, Queue, DDR5, GPU reconciler, worked timing trace"),
        ("Chapter 5: Point 3 — Prefetching Effectiveness", "Workloads W1–W7, latency hiding, coverage, traffic & 14 verified plots"),
        ("Chapter 6: Point 4 — HBM Cache & FA-LRU", "Speculative cache pollution, windowed bottom-W scan, lazy epoch halving & scaling plot"),
        ("Chapter 7: Point 5 — Comparative Evaluation", "Four-way architectural matrix (Configs A, B, C, D) & 3 comparative plots"),
        ("Chapter 8: Point 6 — Sensitivity Analysis", "Batch size, MLA KV cache scaling, lookahead dynamics, prefetch budget & 6 sensitivity plots"),
        ("Chapter 9: Point 7 — Energy & EDP Modeling", "Calibrated hardware energy model, 28-run matrix, K=2 sweet spot & 3 energy plots"),
        ("Chapter 10: Bonus — Authentic MoE Traces", "DeepSeek-V3, Mixtral, MoE-Anywhere parquet adapter & sequence replay"),
        ("Chapter 11: Reproducibility & Repository Guide", "Toolchain, cmake compilation, test suites & complete reproduction recipes"),
        ("Chapter 12: Limitations & Threats to Validity", "Bandwidth scaling limits, single-GPU scope, analytical energy bounds"),
        ("Chapter 13: Conclusions & Scientific Insights", "Architectural synthesis and final takeaways for future MoE hardware"),
        ("Appendices A–C", "Mathematical formulations, simulator YAML schemas, complete 191-run ledger")
    ]
    for ch_title, ch_desc in toc_items:
        add_bullet_p(doc, ch_title + " —", ch_desc)

    add_styled_heading(doc, "List of Figures (All 32 Verified Artifacts)", 2)
    fig_summary_list = [
        ("Figure 3.1", "Certified Frozen Recall vs. Profiling Percentage for CSTP Predictor"),
        ("Figure 3.2", "Offline Hit Rate vs. Profiling Percentage across Prediction Horizons (H=1, 2, 4)"),
        ("Figure 3.3", "ST-MoE Cold-Start vs. Warm-State Prediction Recall across Model Layers"),
        ("Figure 3.4", "End-to-End Prediction Hit Rate Comparison: CSTP vs. ST-MoE Baseline"),
        ("Figure 3.5", "ST-MoE Native Confidence Threshold vs. Candidate Recall Trade-Off"),
        ("Figure 5.1", "Total Execution Time vs. Exposed Memory Stall for Workload W1 (B=14, T=4)"),
        ("Figure 5.2", "Total Execution Time vs. Exposed Memory Stall for Workload W3 (B=14, T=128)"),
        ("Figure 5.3", "Total Execution Time vs. Exposed Memory Stall for Workload W4 (B=16, T=64)"),
        ("Figure 5.4", "Total Execution Time vs. Exposed Memory Stall for Workload W5 (B=32, T=32)"),
        ("Figure 5.5", "Total Execution Time vs. Exposed Memory Stall for Workload W7 (B=128, T=16)"),
        ("Figure 5.6", "Comprehensive Overview of Execution Time and Stall Reduction across All Workloads"),
        ("Figure 5.7", "Prefetch Coverage Scaling vs. Batch Size (B=1 to 128)"),
        ("Figure 5.8", "Prefetch Coverage vs. Sequence Generation Length (T=32 to 512 Tokens)"),
        ("Figure 5.9", "Unused Prefetch Rate (%) across Workloads W1–W7"),
        ("Figure 5.10", "Wasted Speculative DDR5 Memory Traffic Volume (GiB)"),
        ("Figure 5.11", "DDR5 Memory Traffic Composition: Demand Misses vs. Speculative Prefetches"),
        ("Figure 5.12", "Percentage Stall Time Reduction under Predictor-Driven Prefetching"),
        ("Figure 5.13", "Average Expert Load Latency Reduction across Workloads"),
        ("Figure 5.14", "End-to-End Decode Step Latency Reduction (%)"),
        ("Figure 6.1", "FA-LRU Scaling Effect: Cache Hit Rate and Pollution Resistance vs. Standard LRU"),
        ("Figure 7.1", "FA-LRU Incremental Contribution: Isolating Prefetching from Cache Management Gains"),
        ("Figure 7.2", "Four-Way Architectural Exposed Memory Stall Comparison (Configs A, B, C, D)"),
        ("Figure 7.3", "DDR5 Off-Chip Traffic Composition across Four Architectural Configurations"),
        ("Figure 8.1", "Exposed Memory Stall vs. Inference Batch Size Scaling (B=4 to 32)"),
        ("Figure 8.2", "Per-Token Generation Latency vs. Batch Size Scaling"),
        ("Figure 8.3", "KV Cache Memory Capacity Footprint vs. Context Sequence Length"),
        ("Figure 8.4", "Attention Computation Time Scaling vs. Context Length and MoE Weight Hiding"),
        ("Figure 8.5", "Timely Prefetch Coverage Dynamics vs. Lookahead Horizon (H=1 to 4)"),
        ("Figure 8.6", "Speculative DDR5 Bandwidth Inflation vs. Prefetch Budget (K=2 to 8)"),
        ("Figure 9.1", "Subsystem Energy Consumption Breakdown (Static DRAM, HBM, DDR5, MAC)"),
        ("Figure 9.2", "Energy-Performance Pareto Frontier across Configurations and Prefetch Budgets"),
        ("Figure 9.3", "Energy-Delay Product (EDP) Scaling across Prefetch Aggressiveness Levels")
    ]
    for f_id, f_name in fig_summary_list:
        add_bullet_p(doc, f_id + " :", f_name)

    add_styled_heading(doc, "List of Architecture Tables", 2)
    tbl_summary_list = [
        ("Table 2.1", "Calibrated Memory Hierarchy and Accelerator Physical Specifications"),
        ("Table 2.2", "Mapping Project Deliverables to Repository Implementation Modules"),
        ("Table 3.1", "Algorithmic Comparison: CSTP Proposed Predictor vs. ST-MoE Literature Baseline"),
        ("Table 3.2", "Offline Expert Prediction Accuracy across Benchmark Suites (H=1)"),
        ("Table 4.1", "Comprehensive Source Code Implementation Map of LLMSimulator Prefetching Architecture"),
        ("Table 4.2", "Step-by-Step Worked Numerical Execution Trace for DeepSeek-V3 Layer Transition"),
        ("Table 4.3", "Simulation Configuration Parameters and Hardware Architecture Constants"),
        ("Table 5.1", "Prefetching Effectiveness across Large Workloads W1 to W7 (DeepSeek-V3, 64 GB/s)"),
        ("Table 6.1", "HBM Cache Eviction Comparison: Baseline LRU vs. Frequency-Aware LRU (FA-LRU)"),
        ("Table 7.1", "Authoritative Four-Way Comparative Evaluation Matrix (Workload W4: B=32, T=128)"),
        ("Table 8.1", "Lookahead Horizon Sensitivity Dynamics (Anchor Workload: B=32, P=2048, T=128, K=4)"),
        ("Table 8.2", "Prefetch Candidate Budget (K) Sensitivity and Bus Saturation (H=1, B=32, T=128)"),
        ("Table 9.1", "Calibrated Physical Hardware Energy Model Coefficients"),
        ("Table 9.2", "Reconciled Hardware Energy and Delay Analysis Matrix (Anchor Workload: B=32, T=128)"),
        ("Table 10.1", "Authentic Production MoE Routing Traces Integrated into LLMSimulator")
    ]
    for t_id, t_name in tbl_summary_list:
        add_bullet_p(doc, t_id + " :", t_name)

    doc.add_page_break()

    # ==========================================
    # CHAPTER 1: EXECUTIVE SUMMARY
    # ==========================================
    add_styled_heading(doc, "Chapter 1: Executive Summary", 1)
    add_body_p(doc, "Mixture-of-Experts (MoE) architectures, exemplified by state-of-the-art models such as DeepSeek-V3 and Mixtral-8x7B, have fundamentally decoupled model parameter capacity from per-token computational cost. By selectively routing each token to a small subset of specialized feed-forward expert networks (Top-K routing), MoE models scale total parameters into the hundreds of billions while maintaining compute budgets comparable to much smaller dense models.")
    add_body_p(doc, "However, this parameter scaling introduces a catastrophic computer architecture bottleneck: the MoE Memory Wall. In offloaded inference systems where multi-hundred gigabyte expert weights cannot fit entirely within local accelerator High-Bandwidth Memory (HBM2e), expert tensors must reside in off-chip host DDR5 memory and be dynamically streamed across interconnect links (such as PCIe Gen4/Gen5) upon token demand. Because DDR5 transfer latency (3.901 ms per 84 MiB / 88,080,384-byte expert at 22.578 GB/s; 1.376 ms at 64 GB/s) dwarfs inter-layer GPU computation time (20 to 50 microseconds), unmitigated demand offloading causes the GPU to spend over 97% of its execution time idle, stalled waiting for off-chip memory fetches.")
    add_body_p(doc, "The objective of this research project is to design, implement, and rigorously evaluate an end-to-end predictive prefetching and intelligent caching system within the cycle-approximate LLMSimulator framework. Our proposed system introduces the Correlation-based Spatio-Temporal Predictor (CSTP), an asynchronous DDR5 prefetch engine, and a Frequency-Aware Least Recently Used (FA-LRU) cache replacement policy.")
    
    add_styled_heading(doc, "1.1 Key Scientific and Architectural Findings", 2)
    add_bullet_p(doc, "Training-Free Prediction Superiority:", "Our proposed CSTP predictor achieves an overall expert prediction recall of 82.02% across diverse benchmarks, decisively outperforming the literature-standard ST-MoE baseline (68.12%) by an absolute +13.90 percentage-point margin (+20.40% relative improvement) while requiring zero neural network training.")
    add_bullet_p(doc, "Resolution of Strict Timeliness vs. Late In-Flight Attachment:", "Empirical telemetry across 73 cycle-level simulations reveals that strict timely prefetching (transfers completing before demand) is physically impossible under standard interconnects (0.00% timely hits) because transfer latency exceeds compute windows by 39x. However, prefetching successfully achieves massive latency hiding via late in-flight attachment (up to 21.29% coverage), reducing exposed stall by up to 6.717 seconds in large workloads.")
    add_bullet_p(doc, "End-to-End Latency Improvements:", "In heavy production workloads (Workload W7: B=70, T=64, DeepSeek-V3), CSTP eliminates 6,744.86 ms of exposed stall time, delivering 6.717 seconds of wall-clock time saved (104.95 ms per token generation step across 64 steps).")
    add_bullet_p(doc, "Protection Against Speculative Cache Pollution:", "Our FA-LRU cache replacement policy successfully protects 215,314 hot expert instances in Workload W3 and 431,909 instances in W4 from speculative prefetch eviction, providing 36 critical hot-expert reuses that would have suffered costly DDR5 reloads under standard LRU.")
    add_bullet_p(doc, "Hardware Energy Optimization Sweet Spot:", "Exhaustive energy modeling across 28 controlled simulations establishes that prefetching is not inherently energy-expensive. Under a calibrated conservative budget (K = 2, k2_c4), the system achieves a net total energy reduction of -27.68 Joules (-0.295% savings) and minimizes the Energy-Delay Product (EDP) by -1.72% (7,079.32 kJ·s vs 7,202.85 kJ·s), because the static DRAM leakage energy saved by faster execution outweighs the minor active DDR5 transfer overhead.")

    # ==========================================
    # CHAPTER 2: MOTIVATION & ARCHITECTURE
    # ==========================================
    add_styled_heading(doc, "Chapter 2: Project Motivation and System Architecture", 1)
    add_body_p(doc, "Dense large language models execute identical feed-forward operations for every token, resulting in computational costs that scale strictly linearly with total parameter count. Mixture-of-Experts models circumvent this limit by replacing dense feed-forward blocks with an ensemble of N independent expert networks, coupled with a parametric router that computes a sparse routing vector per token. While this architectural paradigm enables trillion-parameter modeling capacity, it shifts the primary performance bottleneck from GPU compute arithmetic to memory capacity and interconnect transfer bandwidth.")
    
    add_styled_heading(doc, "2.1 The Memory Wall in Offloaded MoE Inference", 2)
    add_body_p(doc, "Consider the DeepSeek-V3 architecture: the model incorporates 256 routed experts per MoE layer across 40 transformer stages, with Top-8 experts activated per token. Each expert parameter tensor occupies approximately 84 MiB (88,080,384 bytes, FP16/BF16). Storing the complete model on-chip would require over 640 GiB of high-speed memory—far exceeding the 40 GiB capacity of a standard NVIDIA A100 accelerator. Consequently, expert weights must be offloaded to host system memory (DDR5) and dynamically fetched on demand.")
    add_body_p(doc, "Table 2.1 documents the calibrated physical memory hierarchy specifications governing our simulation infrastructure. The fundamental mismatch is glaring: transferring a single 84 MiB expert (88,080,384 bytes) across a 22.578 GB/s link requires exactly 3.901 milliseconds (1.376 ms under peak 64 GB/s). Meanwhile, the GPU Tensor Cores execute the corresponding layer's attention and routing computation in a mere 20 to 50 microseconds. Without prefetching, the accelerator spends over 97% of its execution timeline completely stalled waiting on off-chip transfers.")

    add_native_table(doc, TABLES_CH1_CH3["table_2_1"])

    add_styled_heading(doc, "2.2 Spatial-Temporal Correlation & Predictive Opportunity", 2)
    add_body_p(doc, "Although MoE routing decisions are dynamic and input-dependent, expert activations are not independently and identically distributed. Because natural language exhibits semantic coherence, consecutive tokens within a prompt activate highly correlated subsets of experts. Furthermore, routing choices at layer L exhibit strong predictive correlations with expert activations at layer L+1. By capturing these spatio-temporal routing correlations, a training-free predictor can anticipate future expert demands and dispatch asynchronous transfers ahead of time.")

    add_native_table(doc, TABLES_CH1_CH3["table_2_2"])

    # ==========================================
    # CHAPTER 3: POINT 1 — TRAINING-FREE PREDICTION
    # ==========================================
    add_styled_heading(doc, "Chapter 3: Point 1 — Training-Free Expert Prediction", 1)
    add_body_p(doc, "The first phase of the project focuses on training-free expert prediction: anticipating which expert tensors will be demanded by downstream transformer layers without requiring auxiliary neural networks or complex gradient updates during inference.")
    
    add_styled_heading(doc, "3.1 Formal Mathematical Formulation of CSTP and ST-MoE", 2)
    add_body_p(doc, "We mathematically formulate the proposed Correlation-based Spatio-Temporal Predictor (CSTP) and the literature baseline ST-MoE (Zhao et al., arXiv:2606.15453v1) directly from their C++ implementations in LLMSimulator:")
    
    add_body_p(doc, "1. CSTP Mathematical Formulation (LLMSimulator/src/prefetch/cstp_predictor.cpp):")
    add_bullet_p(doc, "Target Stage Marginal Prior:", "For any target stage L and candidate expert v in {0, ..., E-1}, the empirical marginal probability is smoothed with prior epsilon (1e-6): P_prior(v | L) = (M[L, v] + epsilon) / (sum_j M[L, j] + E * epsilon).")
    add_bullet_p(doc, "Spatial Conditional Probability:", "Given active source experts U at source stage L_src, the conditional probability of candidate v is smoothed with Dirichlet prior lambda (1.0): P_s(v | u, L_src, L_tgt) = (C_s[L_src, u, v] + lambda * P_prior(v | L_tgt)) / (sum_j C_s[L_src, u, j] + lambda). The aggregate spatial score is S_s(v) = (1 / |U|) * sum_{u in U} P_s(v | u, L_src, L_tgt).")
    add_bullet_p(doc, "Temporal Conditional Probability:", "Using the actual experts W routed at target stage L_tgt during the previous token (stored in history_table_[request_id][L_tgt]): P_t(v | w, L_tgt) = (C_t[L_tgt, w, v] + lambda * P_prior(v | L_tgt)) / (sum_j C_t[L_tgt, w, j] + lambda). The aggregate temporal score is S_t(v) = (1 / |W|) * sum_{w in W} P_t(v | w, L_tgt).")
    add_bullet_p(doc, "Composite Scoring & Top-K Admission:", "The final score is a convex combination: S(v) = alpha * S_s(v) + (1 - alpha) * S_t(v), where alpha = 0.5. The admitted prefetch candidates are P_K = TopK_{v} S(v).")
    add_bullet_p(doc, "Frozen Profiled Mode Dynamics:", "When online_adaptation_ == false, count matrices M, C_s, and C_t are frozen from profiling. However, history_table_ continues to update dynamically per token, maintaining active temporal tracking across sequence generation.")

    add_body_p(doc, "2. ST-MoE Baseline Formulation (LLMSimulator/src/prefetch/st_moe_predictor.cpp):")
    add_bullet_p(doc, "Cross-Layer Correlation Table (CCT):", "Maintains bounded capacity C = 8 per expert, with 2-bit saturating counters c in {0, 1, 2, 3} (initialized to 2).")
    add_bullet_p(doc, "History Table (HT):", "Stores active target experts from the previous token, contributing an additive +2.0 confidence bonus.")
    add_bullet_p(doc, "Scoring & Reinforcement Updates:", "S_ST-MoE(v) = sum_{u in U} c_{L_src, u, v} + 2 * 1[v in HT]. On observation: matching entries increment up to 3; mismatching entries decrement; candidates reaching 0 are replaced by new unmapped target experts initialized to 2.")

    add_styled_heading(doc, "3.2 Worked 4-Expert Numerical Demonstration", 2)
    add_body_p(doc, "To provide concrete numerical clarity, consider an MoE layer with E = 4 candidate experts {0, 1, 2, 3}. Suppose at source stage L = 10, active experts are U = {1, 3}. Target stage prior counts are M = [10, 30, 40, 20] (total = 100), yielding P_prior = [0.10, 0.30, 0.40, 0.20].")
    add_bullet_p(doc, "Spatial Computation:", "Source u=1 has co-counts C_s[1] = [5, 0, 15, 0] (sum=20, denom=21): P_s(0|1)=0.2429, P_s(1|1)=0.0143, P_s(2|1)=0.7333, P_s(3|1)=0.0095. Source u=3 has C_s[3] = [0, 10, 10, 0] (denom=21): P_s(0|3)=0.0048, P_s(1|3)=0.4905, P_s(2|3)=0.4952, P_s(3|3)=0.0095. Mean spatial scores: S_s = [0.1239, 0.2524, 0.6143, 0.0095].")
    add_bullet_p(doc, "Temporal Computation:", "Previous token target history is W = {2}. Temporal transition counts C_t[2] = [2, 4, 18, 6] (sum=30, denom=31): P_t(0|2)=0.0677, P_t(1|2)=0.1387, P_t(2|2)=0.5935, P_t(3|2)=0.2000. Hence S_t = [0.0677, 0.1387, 0.5935, 0.2000].")
    add_bullet_p(doc, "Composite Combination (alpha = 0.5):", "S(0) = 0.5(0.1239) + 0.5(0.0677) = 0.0958; S(1) = 0.5(0.2524) + 0.5(0.1387) = 0.1956; S(2) = 0.5(0.6143) + 0.5(0.5935) = 0.6039; S(3) = 0.5(0.0095) + 0.5(0.2000) = 0.1048.")
    add_bullet_p(doc, "Top-K Admission (K = 2):", "Ranking: Expert 2 (0.6039) > Expert 1 (0.1956) > Expert 3 (0.1048) > Expert 0 (0.0958). Admitted prefetch candidates: P_2 = {2, 1}.")

    add_styled_heading(doc, "3.3 Multi-Label Set-Theoretic Evaluation Metrics", 2)
    add_body_p(doc, "For predicted candidate set P_K (|P_K| = K) and ground-truth active router set A (|A| = K_act = 8):")
    add_bullet_p(doc, "Top-K Recall:", "Recall@K = |P_K cap A| / |A|. Measures fraction of demanded experts successfully anticipated.")
    add_bullet_p(doc, "Top-K Precision:", "Precision@K = |P_K cap A| / |P_K|. Measures fraction of prefetched candidates actually utilized.")
    add_bullet_p(doc, "Hit Rate:", "HitRate@K = 1[|P_K cap A| >= 1]. Evaluates probability that at least one demanded expert was prefetched.")
    add_bullet_p(doc, "Exact Set Accuracy:", "ExactSetAccuracy = 1[P_K == A]. Evaluates perfect set concordance.")

    add_native_table(doc, TABLES_CH1_CH3["table_3_1"])
    add_code_box(doc, CODE_SNIPPETS_CH3["cstp_core"])
    add_code_box(doc, CODE_SNIPPETS_CH3["st_moe_core"])

    add_styled_heading(doc, "3.4 Empirical Offline Accuracy Benchmark", 2)
    add_body_p(doc, "Table 3.2 documents the offline accuracy evaluation of CSTP versus ST-MoE across 188 prompts from the MoE-Anywhere and MoE-Suite benchmark suites. Under identical single-step lookahead (H = 1) and candidate budget (K = 4), CSTP achieves an aggregate recall of 82.02% compared to 68.12% for ST-MoE, delivering a decisive +13.90% absolute gain.")

    add_native_table(doc, TABLES_CH1_CH3["table_3_2"])

    add_styled_heading(doc, "3.5 Point 1 Figure Evaluations", 2)
    add_embedded_figure(doc, "fig1_certified_frozen_recall_vs_profiling", "mohman/point_1/cstp/results/plots/fig1_certified_frozen_recall_vs_profiling.png")
    add_embedded_figure(doc, "fig6_profiling_pct_vs_hit_rate", "mohman/point_1/cstp/results/plots/fig6_profiling_pct_vs_hit_rate.png")

    add_styled_heading(doc, "Impact of Different Factors on ST-MoE Prediction and Prefetching", 2)
    add_embedded_figure(doc, "cold_vs_warm_recall", "mohman/point_1/st_moe/plots/cold_vs_warm_recall.png")
    add_embedded_figure(doc, "hit_rate_comparison", "LLMSimulator/results/endsem_phase_c/figures/fig4_overall_hit_rate_comparison.png")
    add_embedded_figure(doc, "native_threshold_tradeoff", "mohman/point_1/st_moe/plots/native_threshold_tradeoff.png")

    # ==========================================
    # CHAPTER 4: POINT 2 — PREFETCHING IMPLEMENTATION
    # ==========================================
    add_styled_heading(doc, "Chapter 4: Point 2 — Complete Predictor-Driven Prefetching Implementation in LLMSimulator", 1)
    add_body_p(doc, "This chapter presents the definitive architectural and source-code walkthrough of the live predictor-driven prefetching engine implemented within LLMSimulator. Unlike offline trace analysis, the live engine operates asynchronously inside the cycle-approximate simulator, scheduling real DMA transfers across simulated DDR5 links, enforcing queue capacities, resolving memory contention, and reconciling GPU demand stalls.")

    add_styled_heading(doc, "4.1 The End-to-End 20-Step Prefetching Pipeline", 2)
    add_body_p(doc, "The live prefetching pipeline executes according to a strict 20-step causal sequence:")
    add_bullet_p(doc, "Step 1 (Token Ingestion):", "The sequence scheduler dispatches an active token batch to transformer Stage L.")
    add_bullet_p(doc, "Step 2 (Authoritative Routing):", "The router resolves Top-K routing decisions for Stage L, selecting active experts.")
    add_bullet_p(doc, "Step 3 (Live Engine Hook):", "BatchedSequence invokes LivePredictionEngine::on_stage_routing_complete with active expert IDs.")
    add_bullet_p(doc, "Step 4 (Past Prediction Verification):", "The engine reconciles past predictions for Stage L, logging ground-truth accuracy metrics.")
    add_bullet_p(doc, "Step 5 (Predictor Invocation):", "The active predictor (CSTP) queries correlation matrices for target Stage L + H.")
    add_bullet_p(doc, "Step 6 (Candidate Scoring):", "CSTP computes composite spatial-temporal scores for all candidate experts in Stage L + H.")
    add_bullet_p(doc, "Step 7 (Budget & Ranking):", "Top-K candidates are ranked in descending score order.")
    add_bullet_p(doc, "Step 8 (Candidate Enqueue):", "PrefetchCandidateQueue enqueues candidates, filtering against existing queue entries.")
    add_bullet_p(doc, "Step 9 (Capacity Check):", "Queue capacity (16 slots) is strictly enforced; lower-scoring candidates are pruned.")
    add_bullet_p(doc, "Step 10 (Dispatch Trigger):", "The scheduler invokes LivePredictionEngine::dispatch_candidates.")
    add_bullet_p(doc, "Step 11 (Residency Inspection):", "Device checks whether candidate experts already reside in on-chip HBM.")
    add_bullet_p(doc, "Step 12 (DDR5 Submission):", "For non-resident candidates, DDR5TransferEngine::submit_prefetch reserves bus cycles.")
    add_bullet_p(doc, "Step 13 (Bus Serialization):", "Transfers are serialized across the DDR5 bus at calibrated 22.578 GB/s bandwidth.")
    add_bullet_p(doc, "Step 14 (Computation Overlap):", "The GPU executes Stage L Tensor Core GEMM kernels while DMA transfers progress in background.")
    add_bullet_p(doc, "Step 15 (Transfer Arrival):", "Device::process_completed_prefetches marks completed transfers as resident in HBM.")
    add_bullet_p(doc, "Step 16 (Token Transition):", "Tokens advance to Stage L + H; GPU encounters expert weight demand.")
    add_bullet_p(doc, "Step 17 (Demand Resolution):", "LinearExecutionGPU evaluates residency, classifying request into one of four demand outcomes.")
    add_bullet_p(doc, "Step 18 (In-Flight Attachment):", "If transfer is in-flight, demand attaches; GPU stalls only for remaining transit duration.")
    add_bullet_p(doc, "Step 19 (Stall Accounting):", "PrefetchMetricsCollector records exposed stall nanoseconds and energy telemetry.")
    add_bullet_p(doc, "Step 20 (GEMM Execution):", "Expert GEMM executes; cache eviction logic updates residency.")

    add_native_table(doc, TABLES_CH4["table_4_1"])

    add_styled_heading(doc, "4.2 Deep Dive: Core Prefetching Engine Components", 2)
    add_body_p(doc, "The C++ implementation is partitioned across four primary classes:")
    add_bullet_p(doc, "LivePredictionEngine (live_prediction_engine.cpp):", "Acts as the central coordinator. It binds directly into BatchedSequence::update_expert, listening to causal routing events and triggering asynchronous candidate evaluation.")
    add_bullet_p(doc, "PrefetchCandidateQueue (prefetch_candidate_queue.cpp):", "A deterministic priority queue keyed by target layer and expert ID. It performs deduplication against in-flight transfers and applies candidate budget throttling.")
    add_bullet_p(doc, "DDR5TransferEngine (ddr5_transfer_engine.cpp):", "Cycle-accurate DMA transfer simulator modeling a 22.578 GB/s bus. It computes exact arrival timestamps and manages queueing delays.")
    add_bullet_p(doc, "LinearExecutionGPU (linear_impl.cpp):", "Handles device-side demand reconciliation, evaluating resident hits, timely prefetches, late in-flight attachments, and compulsory demand misses.")

    add_code_box(doc, CODE_SNIPPETS_CH4["live_engine_hook"])
    add_code_box(doc, CODE_SNIPPETS_CH4["candidate_queue_enqueue"])
    add_code_box(doc, CODE_SNIPPETS_CH4["ddr5_engine_submit"])
    add_code_box(doc, CODE_SNIPPETS_CH4["linear_demand_reconciliation"])

    add_styled_heading(doc, "4.3 Worked Numerical Execution Trace", 2)
    add_body_p(doc, "Table 4.2 presents a complete, concrete worked numerical execution trace for a single layer transition under DeepSeek-V3. It tracks timestamps, queue states, background DMA arrival, late in-flight attachment, and exact stall reduction.")

    add_native_table(doc, TABLES_CH4["table_4_2"])
    add_native_table(doc, TABLES_CH4["table_4_3"])

    # ==========================================
    # CHAPTER 5: POINT 3 — PREFETCHING EFFECTIVENESS
    # ==========================================
    add_styled_heading(doc, "Chapter 5: Point 3 — Prefetching Effectiveness Evaluation", 1)
    add_body_p(doc, "Point 3 evaluates how predictor accuracy translates into actual memory-system performance across 73 cycle-level simulations spanning Workloads W1 through W7. The investigation resolves the central architectural question: under physical interconnect constraints, does prefetching genuinely hide memory stall latency?")

    add_styled_heading(doc, "5.1 Large Workload Performance Matrix (W1–W7)", 2)
    add_body_p(doc, "Table 5.1 provides the comprehensive performance telemetry for Workloads W1 through W7 under DeepSeek-V3 on a calibrated 64 GB/s interconnect. The empirical data highlights a dramatic scaling effect: as batch size increases from B = 14 to B = 128, stall reduction expands from 2.75 ms to 6,717.00 ms (saving 6.717 seconds of wall-clock time in W7).")

    add_native_table(doc, TABLES_CH5_CH7["table_5_1"])

    add_styled_heading(doc, "5.2 Point 3 Figure Evaluations (All 14 Verified Plots)", 2)
    add_embedded_figure(doc, "fig_01a_w1_total_time_vs_stall", "mohman/point_3/final_results/plots/01_total_time_vs_stall_reduction/fig_01a_w1_total_time_vs_stall.png")
    add_embedded_figure(doc, "fig_01a_w3_total_time_vs_stall", "mohman/point_3/final_results/plots/01_total_time_vs_stall_reduction/fig_01a_w3_total_time_vs_stall.png")
    add_embedded_figure(doc, "fig_01a_w4_total_time_vs_stall", "mohman/point_3/final_results/plots/01_total_time_vs_stall_reduction/fig_01a_w4_total_time_vs_stall.png")
    add_embedded_figure(doc, "fig_01a_w5_total_time_vs_stall", "mohman/point_3/final_results/plots/01_total_time_vs_stall_reduction/fig_01a_w5_total_time_vs_stall.png")
    add_embedded_figure(doc, "fig_01a_w7_total_time_vs_stall", "mohman/point_3/final_results/plots/01_total_time_vs_stall_reduction/fig_01a_w7_total_time_vs_stall.png")
    add_embedded_figure(doc, "fig_01b_all_workloads_time_and_stall_overview", "mohman/point_3/final_results/plots/01_total_time_vs_stall_reduction/fig_01b_all_workloads_time_and_stall_overview.png")
    add_embedded_figure(doc, "fig_02b_coverage_vs_batch_size", "mohman/point_3/final_results/plots/02_prefetch_coverage/fig_02b_coverage_vs_batch_size.png")
    add_embedded_figure(doc, "fig_02c_coverage_vs_sequence_length", "mohman/point_3/final_results/plots/02_prefetch_coverage/fig_02c_coverage_vs_sequence_length.png")
    add_embedded_figure(doc, "fig_04a_unused_prefetch_rate", "mohman/point_3/final_results/plots/04_unused_prefetches/fig_04a_unused_prefetch_rate.png")
    add_embedded_figure(doc, "fig_04b_wasted_speculative_bytes", "mohman/point_3/final_results/plots/04_unused_prefetches/fig_04b_wasted_speculative_bytes.png")
    add_embedded_figure(doc, "fig_06a_ddr5_traffic_breakdown", "mohman/point_3/final_results/plots/06_memory_traffic/fig_06a_ddr5_traffic_breakdown.png")
    add_embedded_figure(doc, "fig_07c_stall_reduction_percentage", "mohman/point_3/final_results/plots/07_exposed_memory_stall/fig_07c_stall_reduction_percentage.png")
    add_embedded_figure(doc, "fig_08b_expert_load_latency_reduction", "mohman/point_3/final_results/plots/08_expert_load_latency/fig_08b_expert_load_latency_reduction.png")
    add_embedded_figure(doc, "fig_09b_decode_step_latency_reduction", "mohman/point_3/final_results/plots/09_end_to_end_latency/fig_09b_decode_step_latency_reduction.png")

    # ==========================================
    # CHAPTER 6: POINT 4 — HBM CACHE MANAGEMENT
    # ==========================================
    add_styled_heading(doc, "Chapter 6: Point 4 — HBM Expert Cache Management and FA-LRU", 1)
    add_body_p(doc, "When speculative prefetching is deployed, conventional Least Recently Used (LRU) cache replacement suffers from severe speculative cache pollution. Speculatively prefetched expert weights that turn out to be unneeded are placed at the Most Recently Used position, prematurely evicting frequently demanded base experts.")
    add_body_p(doc, "To protect the HBM expert cache from pollution, we implemented Frequency-Aware LRU (FA-LRU) in LLMSimulator/src/hardware/device.cpp. The active policy implements a Windowed Bottom-W Candidate Scan (W = 64) with Lazy Bit-Shift Epoch Halving:")
    add_bullet_p(doc, "Lazy Epoch Halving:", "Upon inspecting entry k with last access epoch e_last and current global epoch e_global, the access frequency is decayed: decayed = access_frequency >> (e_global - e_last). If the epoch delta exceeds 31, the frequency resets to 1; otherwise it is clamped to >= 1.")
    add_bullet_p(doc, "Victim Selection in Window W:", "FA-LRU evaluates the bottom W unpinned entries of lru_expert_list, selecting candidate k* with the minimum decayed access frequency. Ties are broken by oldest recency.")
    add_bullet_p(doc, "Fallback Beyond Window:", "If all candidates in the bottom-W window are currently pinned (executing GEMMs), the engine falls back to search beyond window W for the first unpinned candidate.")
    add_bullet_p(doc, "Divergence & Protection Diagnostics:", "When k* != k_LRU and the standard LRU candidate was demand-verified, the simulator increments protected_demand_experts_count. Telemetry confirms 431,909 protected decisions in W4 and 36 critical hot-expert reuses.")

    add_native_table(doc, TABLES_CH5_CH7["table_6_1"])
    add_code_box(doc, CODE_SNIPPETS_CH6["falru_victim_selection"])
    add_embedded_figure(doc, "fig9_falru_scaling_effect", "mohman/point_4/big_exp/figures/fig9_falru_scaling_effect.png")

    # ==========================================
    # CHAPTER 7: POINT 5 — COMPARATIVE EVALUATION
    # ==========================================
    add_styled_heading(doc, "Chapter 7: Point 5 — Four-Way Comparative Architecture Evaluation", 1)
    add_body_p(doc, "Point 5 synthesizes the architectural contributions across four controlled system configurations under identical workload parameters (Workload W4: B = 32, T = 128, C_HBM = 35 GiB):")
    add_bullet_p(doc, "Config A (Baseline Demand Offloading):", "Standard offloaded inference with no prediction and LRU eviction.")
    add_bullet_p(doc, "Config B (Shadow Prediction Only):", "CSTP predictor active, but speculative transfers suppressed.")
    add_bullet_p(doc, "Config C (Async Prefetch + LRU):", "Full asynchronous prefetching pipeline with standard LRU cache management.")
    add_bullet_p(doc, "Config D (Complete Proposed System):", "Full asynchronous prefetching combined with FA-LRU frequency-aware caching.")

    add_native_table(doc, TABLES_CH5_CH7["table_7_1"])
    add_embedded_figure(doc, "fig11_falru_incremental_contribution", "mohman/point_5/figures/fig11_falru_incremental_contribution.png")
    add_embedded_figure(doc, "fig6_exposed_stall_comparison", "mohman/point_5/figures/fig6_exposed_stall_comparison.png")
    add_embedded_figure(doc, "fig7_ddr5_traffic_composition", "mohman/point_5/figures/fig7_ddr5_traffic_composition.png")

    # ==========================================
    # CHAPTER 8: POINT 6 — SENSITIVITY ANALYSIS
    # ==========================================
    add_styled_heading(doc, "Chapter 8: Point 6 — Multi-Dimensional Sensitivity Analysis", 1)
    add_body_p(doc, "Point 6 conducts extensive sensitivity analysis across four fundamental architectural dimensions: inference batch size (B), context sequence length (P), prediction lookahead horizon (H), and prefetch candidate budget (K).")

    add_styled_heading(doc, "8.1 Multi-Head Latent Attention (MLA) KV Cache Sizing", 2)
    add_body_p(doc, "A critical architectural consideration in DeepSeek-V3 is the memory footprint of the Key-Value (KV) cache. Standard Multi-Head Attention (MHA) would consume 60.4 GiB at batch size B = 32 and context P = 2048, completely overwhelming a 40 GiB accelerator. DeepSeek-V3 implements Multi-Head Latent Attention (MLA), compressing Key and Value projections into a shared low-rank latent vector d_c = 512 plus decoupled RoPE key d_{c,rope} = 64:")
    add_bullet_p(doc, "Per-Token KV Storage:", "S_token = (d_c + d_{c,rope}) * b = (512 + 64) * 2 bytes = 1,152 bytes = 1.125 KiB per layer.")
    add_bullet_p(doc, "Total Model KV Footprint:", "Footprint_MLA = N_layers * B * P * S_token = 61 * 32 * 2048 * 1,152 bytes = 4.606 GB = 4.06 GiB.")
    add_bullet_p(doc, "Comparison vs. Standard MHA:", "Footprint_MHA = 2 * N_layers * B * P * H_KV * D_head * b = 2 * 61 * 32 * 2048 * 128 * 128 * 2 bytes = 64.9 GB = 60.4 GiB.")
    add_body_p(doc, "This 14.9x memory compression is what physically enables 35.0 GiB of HBM to be reserved for the MoE expert cache, leaving comfortable headroom for activations.")

    add_native_table(doc, TABLES_CH8_CH13["table_8_1"])
    add_native_table(doc, TABLES_CH8_CH13["table_8_2"])

    add_embedded_figure(doc, "fig2_batch_size_exposed_stall", "mohman/point_6/final_results/figures/batch_size/fig2_batch_size_exposed_stall.png")
    add_embedded_figure(doc, "fig3_batch_size_token_latency", "mohman/point_6/final_results/figures/batch_size/fig3_batch_size_token_latency.png")
    add_embedded_figure(doc, "fig4_context_length_kv_cache", "mohman/point_6/final_results/figures/context_length/fig4_context_length_kv_cache.png")
    add_embedded_figure(doc, "fig5_context_length_attention_time", "mohman/point_6/final_results/figures/context_length/fig5_context_length_attention_time.png")
    add_embedded_figure(doc, "fig7_lookahead_timely_coverage", "mohman/point_6/final_results/figures/lookahead/fig7_lookahead_timely_coverage.png")
    add_embedded_figure(doc, "fig10_budget_speculative_ddr5", "mohman/point_6/final_results/figures/prefetch_budget/fig10_budget_speculative_ddr5.png")

    # ==========================================
    # CHAPTER 9: POINT 7 — HARDWARE ENERGY & EDP
    # ==========================================
    add_styled_heading(doc, "Chapter 9: Point 7 — Hardware Energy Modeling and EDP Analysis", 1)
    add_body_p(doc, "Point 7 evaluates the hardware energy implications of predictive prefetching. The primary architectural question is whether the reduction in execution time (which cuts static DRAM leakage energy) outweighs the additional energy consumed by speculative DDR5 data transfers.")
    add_body_p(doc, "The total system energy E_total is modeled as the sum of four physical components:")
    add_body_p(doc, "E_total = E_static_dram + E_ddr5_active + E_hbm_io + E_gpu_mac")
    add_body_p(doc, "where E_static_dram = P_static * T_wall (P_static = 4.5 W), E_ddr5_active = 0.100 nJ/byte, E_hbm_io = 3.48 pJ/bit, and E_gpu_mac = 0.23 pJ/FLOP. Table 9.1 documents these physical coefficients.")
    add_body_p(doc, "In Table 9.2, the full subsystem energy breakdown is reconciled. For the baseline configuration, GPU MAC compute consumes exactly 71.25 Joules (0.76% of system total), HBM consumes 1,042.87 Joules (11.11%), Static DRAM consumes 3,452.72 Joules (36.78%), and DDR5 active transfers consume 4,820.77 Joules (51.35%), yielding a total system energy of 9,387.61 Joules.")
    add_body_p(doc, "Crucially, under the calibrated prefetch budget K = 2 (k2_c4), the system achieves a net energy saving of -27.68 Joules (-0.295%) and reduces the Energy-Delay Product by -1.72% (7,079.32 kJ·s vs 7,202.85 kJ·s). This proves that conservative prefetching is energy-positive.")

    add_native_table(doc, TABLES_CH8_CH13["table_9_1"])
    add_native_table(doc, TABLES_CH8_CH13["table_9_2"])

    add_embedded_figure(doc, "fig12_component_pie_breakdown", "mohman/point_7/figures/png/fig12_component_pie_breakdown.png")
    add_embedded_figure(doc, "fig11_energy_performance_pareto", "mohman/point_7/figures/png/fig11_energy_performance_pareto.png")
    add_embedded_figure(doc, "fig6_energy_delay_product", "mohman/point_7/figures/png/fig6_energy_delay_product.png")

    # ==========================================
    # CHAPTER 10: BONUS — AUTHENTIC MoE TRACES
    # ==========================================
    add_styled_heading(doc, "Chapter 10: Bonus — Realistic Routing Traffic from Authentic MoE Traces", 1)
    add_body_p(doc, "To guarantee that all evaluations reflect genuine production routing dynamics rather than synthetic distributions, our simulator integrates authentic routing traces from DeepSeek-V3, Mixtral-8x7B, and MoE-Anywhere parquet corpora.")
    add_native_table(doc, TABLES_CH8_CH13["table_10_1"])

    # ==========================================
    # CHAPTER 11: REPRODUCIBILITY & MANUAL
    # ==========================================
    add_styled_heading(doc, "Chapter 11: Reproducibility, Repository Layout, and Execution Manual", 1)
    add_body_p(doc, "All experiments and figures presented in this report are 100% reproducible directly from the project repository: https://github.com/Mohit14032007/ca_project_endsem.")
    add_body_p(doc, "Reproduction instructions, build commands, and execution scripts are fully documented in REPORT_REPRODUCIBILITY.md and REPORT_SOURCE_AUDIT.md. To recompile LLMSimulator:")
    add_body_p(doc, "cd LLMSimulator && mkdir -p build && cd build && cmake .. && make -j$(nproc)")

    # ==========================================
    # CHAPTER 12: LIMITATIONS & VALIDITY
    # ==========================================
    add_styled_heading(doc, "Chapter 12: Limitations, Threat to Validity, and Failure Modes", 1)
    add_body_p(doc, "We acknowledge several methodological and physical boundaries:")
    add_bullet_p(doc, "Interconnect Bandwidth Saturation:", "Under narrow interconnects (<32 GB/s), prefetch candidate budgets exceeding K = 4 induce severe bus congestion, degrading performance.")
    add_bullet_p(doc, "Single-Node Accelerator Boundary:", "Evaluations model a single host-to-GPU PCIe link; multi-node distributed MoE serving introduces additional network fabric latencies.")
    add_bullet_p(doc, "Analytical Energy Scope:", "The energy model accounts for DRAM, HBM, and compute MACs, omitting peripheral board fan and power supply conversion losses.")

    # ==========================================
    # CHAPTER 13: CONCLUSION
    # ==========================================
    add_styled_heading(doc, "Chapter 13: Conclusion and Research Insights", 1)
    add_body_p(doc, "The MoE Memory Wall poses one of the most critical challenges facing next-generation LLM serving infrastructure. This project has comprehensively demonstrated that training-free predictive prefetching (CSTP) coupled with frequency-aware caching (FA-LRU) provides a mathematically principled, hardware-viable solution.")
    add_body_p(doc, "By achieving 82.02% prediction recall, eliminating up to 6.717 seconds of exposed stall time, and lowering total system Energy-Delay Product by -1.72% under a calibrated K = 2 budget, this work proves that asynchronous prefetching can effectively bridge the memory wall for large-scale MoE inference.")

    # Save Document
    doc.save(docx_path)
    sz = os.path.getsize(docx_path)
    print(f"Successfully generated {docx_path} ({sz:,} bytes).")

    v2_docx_path = os.path.join("mohman/final_report_v2", "MoE_Memory_Wall_Final_Report_CORRECTED.docx")
    shutil.copy2(docx_path, v2_docx_path)
    print(f"Mirrored to {v2_docx_path} ({os.path.getsize(v2_docx_path):,} bytes).")

if __name__ == "__main__":
    generate_docx()
