# generate_pdf.py: Generates high-quality publication PDF using ReportLab with embedded figures

import os
import re
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Preformatted, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        if self._pageNumber > 1:
            self.saveState()
            self.setFont("Helvetica", 8.5)
            self.setFillColor(colors.HexColor("#555555"))
            
            # Header
            self.drawString(
                54, letter[1] - 36,
                "MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference"
            )
            self.setStrokeColor(colors.HexColor("#D0D7DE"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            
            # Footer
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(letter[0] - 54, 34, page_text)
            self.drawString(54, 34, "https://github.com/Mohit14032007/ca_project_endsem")
            self.line(54, 46, letter[0] - 54, 46)
            
            self.restoreState()

def create_report_pdf():
    md_path = "mohman/final_report/MoE_Memory_Wall_Detailed_Report.md"
    pdf_path = "mohman/final_report/MoE_Memory_Wall_Detailed_Report.pdf"
    
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom typography
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1B365D'),
        alignment=1,
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#4A607A'),
        alignment=1,
        spaceAfter=24
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1B365D'),
        spaceBefore=16,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2E5B88'),
        spaceBefore=12,
        spaceAfter=4,
        keepWithNext=True
    )
    
    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#222222'),
        spaceBefore=8,
        spaceAfter=2,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.0,
        leading=12.5,
        textColor=colors.HexColor('#222222'),
        spaceAfter=4
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=14,
        spaceAfter=2.5
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.2,
        leading=9.0,
        textColor=colors.HexColor('#111111')
    )
    
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11.0,
        textColor=colors.HexColor('#4A607A'),
        alignment=1,
        spaceBefore=3,
        spaceAfter=8
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=9.8,
        textColor=colors.HexColor('#222222')
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=9.8,
        textColor=colors.white
    )
    
    story = []
    
    # Cover Page
    story.append(Spacer(1, 40))
    story.append(Paragraph("MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference", title_style))
    story.append(Paragraph("Comprehensive Technical Implementation and Experimental Evaluation Report", subtitle_style))
    story.append(Spacer(1, 20))
    
    meta_data = [
        [Paragraph("<b>GitHub Repository:</b>", table_cell_style), Paragraph("<a href='https://github.com/Mohit14032007/ca_project_endsem'>https://github.com/Mohit14032007/ca_project_endsem</a>", table_cell_style)],
        [Paragraph("<b>Course Title:</b>", table_cell_style), Paragraph("CS Advanced Computer Architecture (End-Semester Capstone Project)", table_cell_style)],
        [Paragraph("<b>Student Author:</b>", table_cell_style), Paragraph("Mohit (Roll No: Mohit14032007, Username: mohman)", table_cell_style)],
        [Paragraph("<b>Academic Term:</b>", table_cell_style), Paragraph("Autumn / End-Semester 2026", table_cell_style)],
        [Paragraph("<b>Baseline Commit:</b>", table_cell_style), Paragraph("d5bf61ff0c066c06f72173189f9924329f613a88", table_cell_style)],
        [Paragraph("<b>Submission Commit:</b>", table_cell_style), Paragraph("c76aa67", table_cell_style)]
    ]
    t_meta = Table(meta_data, colWidths=[130, 374])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F0F4F8')),
        ('BACKGROUND', (1,0), (1,-1), colors.HexColor('#FFFFFF')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#D0D7DE')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E1E4E8')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 35))
    
    # Executive Abstract Callout on Cover
    abs_text = "<b>Executive Summary Abstract:</b> This report presents the authoritative implementation and experimental evaluation of a predictor-driven, asynchronous prefetching architecture designed to mitigate the Mixture-of-Experts (MoE) Memory Wall in large-scale LLM inference. By coupling a Context-Sensitive Temporal Predictor (CSTP) to an asynchronous, calibrated DDR5 transfer engine (22.578 GB/s) and protecting high-frequency working-set experts via Frequency-Aware LRU (FA-LRU) eviction, the system reduces exposed memory stalls by 78.20% and delivers a 23.50% end-to-end wall-clock speedup on 40-layer DeepSeek-V3 workloads while saving 2.026 GB of speculative memory traffic."
    p_abs = Paragraph(abs_text, body_style)
    t_abs = Table([[p_abs]], colWidths=[504])
    t_abs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F9FBFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#1B365D')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_abs)
    
    story.append(PageBreak())
    
    # Process Markdown
    in_code_block = False
    code_lines = []
    in_table = False
    table_lines = []
    
    idx = 0
    img_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        
        # Skip the original markdown cover lines
        if idx < 25 and (stripped.startswith("# MoE") or stripped.startswith("## Comprehensive")):
            idx += 1
            continue
            
        # Image handling: ![Caption](path)
        img_match = img_pattern.match(stripped)
        if img_match:
            caption = img_match.group(1)
            img_rel_path = img_match.group(2)
            if os.path.exists(img_rel_path):
                try:
                    pil_img = PILImage.open(img_rel_path)
                    im_w, im_h = pil_img.size
                    aspect = float(im_h) / float(im_w)
                    target_w = 480.0
                    target_h = target_w * aspect
                    if target_h > 260.0:
                        target_h = 240.0
                        target_w = target_h / aspect
                    rl_img = Image(img_rel_path, width=target_w, height=target_h)
                    rl_img.hAlign = 'CENTER'
                    story.append(Spacer(1, 6))
                    story.append(rl_img)
                    story.append(Paragraph(caption, caption_style))
                    print(f"Embedded image in PDF: {img_rel_path} ({target_w:.0f}x{target_h:.0f})")
                except Exception as e:
                    print(f"Error embedding {img_rel_path}: {e}")
            else:
                print(f"Warning: Image not found: {img_rel_path}")
            idx += 1
            continue
            
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "\n".join(code_lines)
                sub_lines = code_text.splitlines()[:30]
                t_code = Table([[Preformatted("\n".join(sub_lines), code_style)]], colWidths=[504])
                t_code.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F6F8FA')),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#D0D7DE')),
                    ('TOPPADDING', (0,0), (-1,-1), 5),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ('LEFTPADDING', (0,0), (-1,-1), 6),
                    ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(Spacer(1, 3))
                story.append(t_code)
                story.append(Spacer(1, 3))
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
                raw_headers = [c.strip() for c in content_rows[0].strip('|').split('|')]
                cols_count = len(raw_headers)
                col_w = 504.0 / float(cols_count)
                table_flowables = []
                for r_i, row_str in enumerate(content_rows):
                    cells = [c.strip() for c in row_str.strip('|').split('|')]
                    row_cells = []
                    for c_i in range(min(len(cells), cols_count)):
                        txt = cells[c_i]
                        txt = txt.replace('$', '').replace('\\text{', '').replace('}', '').replace('\\times', 'x').replace('\\approx', '~')
                        if r_i == 0:
                            p = Paragraph(f"<b>{txt}</b>", table_header_style)
                        else:
                            p = Paragraph(txt, table_cell_style)
                        row_cells.append(p)
                    table_flowables.append(row_cells)
                t_render = Table(table_flowables, colWidths=[col_w]*cols_count)
                t_render.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1B365D')),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#D0D7DE')),
                    ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E1E4E8')),
                    ('TOPPADDING', (0,0), (-1,-1), 3),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                    ('LEFTPADDING', (0,0), (-1,-1), 3),
                    ('RIGHTPADDING', (0,0), (-1,-1), 3),
                ]))
                story.append(Spacer(1, 4))
                story.append(t_render)
                story.append(Spacer(1, 5))
            table_lines = []
            
        # Headings & Body
        if stripped.startswith("# "):
            story.append(Spacer(1, 10))
            story.append(Paragraph(stripped[2:], h1_style))
        elif stripped.startswith("## "):
            story.append(Spacer(1, 8))
            story.append(Paragraph(stripped[3:], h2_style))
        elif stripped.startswith("### "):
            story.append(Spacer(1, 5))
            story.append(Paragraph(stripped[4:], h3_style))
        elif stripped.startswith("#### "):
            story.append(Paragraph(f"<b>{stripped[5:]}</b>", body_style))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            txt = stripped[2:].replace('$', '')
            story.append(Paragraph(f"&bull; {txt}", bullet_style))
        elif stripped.startswith("1. ") or stripped.startswith("2. ") or stripped.startswith("3. ") or stripped.startswith("4. ") or stripped.startswith("5. ") or stripped.startswith("6. ") or stripped.startswith("7. ") or stripped.startswith("8. ") or stripped.startswith("9. ") or stripped.startswith("10. "):
            dot_idx = stripped.find(". ")
            num_prefix = stripped[:dot_idx+1]
            txt = stripped[dot_idx+2:].replace('$', '')
            story.append(Paragraph(f"<b>{num_prefix}</b> {txt}", bullet_style))
        elif stripped == "---":
            story.append(Spacer(1, 4))
        elif len(stripped) > 0:
            txt = stripped.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            txt = txt.replace('$', '')
            story.append(Paragraph(txt, body_style))
            
        idx += 1
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF Report successfully written to {pdf_path}")

if __name__ == "__main__":
    create_report_pdf()
