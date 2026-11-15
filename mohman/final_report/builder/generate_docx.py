# generate_docx.py: Generates professional DOCX report with embedded graphs, tables, and callouts

import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_report_docx():
    md_path = "mohman/final_report/MoE_Memory_Wall_Detailed_Report.md"
    docx_path = "mohman/final_report/MoE_Memory_Wall_Detailed_Report.docx"
    
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    doc = Document()
    
    # Page setup: Standard Letter, 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Add page numbering footer
        footer = section.footer
        f_p = footer.paragraphs[0]
        f_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        f_run = f_p.add_run("MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference | ")
        f_run.font.name = 'Calibri'
        f_run.font.size = Pt(9)
        f_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        
    # Styles setup
    styles = doc.styles
    normal_style = styles['Normal']
    normal_font = normal_style.font
    normal_font.name = 'Calibri'
    normal_font.size = Pt(11)
    normal_font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    
    # Cover Page Elements
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference\n")
    title_run.font.name = 'Calibri'
    title_run.font.size = Pt(24)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D) # Navy
    
    sub_run = title_p.add_run("Comprehensive Technical Implementation and Experimental Evaluation Report\n\n")
    sub_run.font.name = 'Calibri'
    sub_run.font.size = Pt(14)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(0x4A, 0x60, 0x7A)
    
    # Metadata Box
    meta_table = doc.add_table(rows=6, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("GitHub Repository", "https://github.com/Mohit14032007/ca_project_endsem"),
        ("Course Title", "CS Advanced Computer Architecture (End-Semester Capstone Project)"),
        ("Student Author", "Mohit (Roll No: Mohit14032007, Username: mohman)"),
        ("Academic Term", "Autumn / End-Semester 2026"),
        ("Baseline Commit", "d5bf61ff0c066c06f72173189f9924329f613a88"),
        ("Submission Commit", "c76aa67")
    ]
    for row_idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[row_idx]
        cell_k, cell_v = row.cells[0], row.cells[1]
        cell_k.text = k
        cell_v.text = v
        set_cell_background(cell_k, "F0F4F8")
        set_cell_background(cell_v, "FFFFFF")
        set_cell_margins(cell_k)
        set_cell_margins(cell_v)
        cell_k.paragraphs[0].runs[0].font.bold = True
        cell_k.paragraphs[0].runs[0].font.size = Pt(10)
        cell_v.paragraphs[0].runs[0].font.size = Pt(10)
        
    doc.add_page_break()
    
    # Process Markdown lines
    in_code_block = False
    code_lines = []
    in_table = False
    table_lines = []
    
    idx = 0
    img_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        
        # Skip original markdown cover lines
        if idx < 25 and (stripped.startswith("# MoE") or stripped.startswith("## Comprehensive")):
            idx += 1
            continue
            
        # Image handling: ![Caption](path)
        img_match = img_pattern.match(stripped)
        if img_match:
            caption = img_match.group(1)
            img_rel_path = img_match.group(2)
            if os.path.exists(img_rel_path):
                # Add picture
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(12)
                p_img.paragraph_format.space_after = Pt(4)
                doc.add_picture(img_rel_path, width=Inches(6.0))
                
                # Add caption
                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_cap.paragraph_format.space_before = Pt(2)
                p_cap.paragraph_format.space_after = Pt(8)
                cap_run = p_cap.add_run(caption)
                cap_run.font.name = 'Calibri'
                cap_run.font.size = Pt(9.5)
                cap_run.font.italic = True
                cap_run.font.color.rgb = RGBColor(0x4A, 0x60, 0x7A)
                print(f"Embedded image in DOCX: {img_rel_path}")
            else:
                print(f"Warning: Image not found: {img_rel_path}")
            idx += 1
            continue
            
        # Code block handling
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "\n".join(code_lines)
                code_p = doc.add_paragraph()
                code_p.paragraph_format.left_indent = Inches(0.3)
                code_p.paragraph_format.right_indent = Inches(0.3)
                code_run = code_p.add_run(code_text)
                code_run.font.name = 'Consolas'
                code_run.font.size = Pt(9.0)
                code_run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            idx += 1
            continue
            
        if in_code_block:
            code_lines.append(line.rstrip())
            idx += 1
            continue
            
        # Table handling
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            table_lines.append(stripped)
            idx += 1
            continue
        elif in_table:
            in_table = False
            content_rows = [r for r in table_lines if not all(c.strip().replace('-', '').replace(':', '') == '' for c in r.strip('|').split('|'))]
            if len(content_rows) > 0:
                cols_count = len(content_rows[0].strip('|').split('|'))
                tbl = doc.add_table(rows=len(content_rows), cols=cols_count)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                for r_i, row_str in enumerate(content_rows):
                    cells = [c.strip() for c in row_str.strip('|').split('|')]
                    for c_i in range(min(len(cells), cols_count)):
                        cell = tbl.rows[r_i].cells[c_i]
                        cell.text = cells[c_i]
                        set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
                        if r_i == 0:
                            set_cell_background(cell, "1B365D")
                            if len(cell.paragraphs[0].runs) > 0:
                                cell.paragraphs[0].runs[0].font.bold = True
                                cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                                cell.paragraphs[0].runs[0].font.size = Pt(9.0)
                        else:
                            if r_i % 2 == 1:
                                set_cell_background(cell, "F9FBFC")
                            else:
                                set_cell_background(cell, "FFFFFF")
                            if len(cell.paragraphs[0].runs) > 0:
                                cell.paragraphs[0].runs[0].font.size = Pt(8.5)
                doc.add_paragraph()
            table_lines = []
            
        # Section Headers
        if stripped.startswith("# "):
            h = doc.add_heading(level=1)
            h.paragraph_format.space_before = Pt(18)
            h.paragraph_format.space_after = Pt(6)
            run = h.add_run(stripped[2:])
            run.font.name = 'Calibri'
            run.font.size = Pt(17)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        elif stripped.startswith("## "):
            h = doc.add_heading(level=2)
            h.paragraph_format.space_before = Pt(13)
            h.paragraph_format.space_after = Pt(4)
            run = h.add_run(stripped[3:])
            run.font.name = 'Calibri'
            run.font.size = Pt(13.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x2E, 0x5B, 0x88)
        elif stripped.startswith("### "):
            h = doc.add_heading(level=3)
            h.paragraph_format.space_before = Pt(10)
            h.paragraph_format.space_after = Pt(2)
            run = h.add_run(stripped[4:])
            run.font.name = 'Calibri'
            run.font.size = Pt(11.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        elif stripped.startswith("#### "):
            h = doc.add_heading(level=4)
            run = h.add_run(stripped[5:])
            run.font.name = 'Calibri'
            run.font.size = Pt(10.5)
            run.font.bold = True
        elif stripped.startswith("- ") or stripped.startswith("* "):
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            p.add_run(stripped[2:])
        elif stripped.startswith("1. ") or stripped.startswith("2. ") or stripped.startswith("3. ") or stripped.startswith("4. ") or stripped.startswith("5. ") or stripped.startswith("6. ") or stripped.startswith("7. ") or stripped.startswith("8. ") or stripped.startswith("9. ") or stripped.startswith("10. "):
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            dot_idx = stripped.find(". ")
            if dot_idx != -1:
                p.add_run(stripped[dot_idx+2:])
            else:
                p.add_run(stripped)
        elif stripped == "---":
            pass
        elif len(stripped) > 0:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2.5)
            p.paragraph_format.space_after = Pt(2.5)
            p.paragraph_format.line_spacing = 1.15
            p.add_run(stripped)
            
        idx += 1
        
    doc.save(docx_path)
    print(f"DOCX Report successfully written to {docx_path}")

if __name__ == "__main__":
    create_report_docx()
