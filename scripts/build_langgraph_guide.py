"""Build the editable Word guide from its Markdown source."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "LangGraph_RAG_Pipeline_Deep_Dive.md"
OUTPUT = ROOT / "docs" / "LangGraph_RAG_Pipeline_Deep_Dive.docx"
NAVY, BLUE, MUTED, DARK = "203754", "356DA8", "66778A", "243447"


def shade(cell, color: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), color)
    props.append(element)


def inline(paragraph, text: str) -> None:
    """Render small Markdown inline subset without leaving raw markers."""
    pattern = re.compile(r"(\*\*.+?\*\*|`.+?`)")
    pos = 0
    for match in pattern.finditer(text):
        if match.start() > pos:
            paragraph.add_run(text[pos:match.start()])
        token = match.group(0)
        run = paragraph.add_run(token[2:-2])
        if token.startswith("**"):
            run.bold = True
        else:
            run.font.name = "Cascadia Mono"
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor.from_string("34495E")
        pos = match.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def add_table(doc: Document, lines: list[str]) -> None:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", re.sub(r"\s", "", item)) for item in cells):
            continue
        rows.append(cells)
    if not rows:
        return
    columns = max(len(row) for row in rows)
    table = doc.add_table(rows=1, cols=columns)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row_index, values in enumerate(rows):
        cells = table.rows[0].cells if row_index == 0 else table.add_row().cells
        for col in range(columns):
            cell = cells[col]
            cell.text = ""
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(1)
            para.paragraph_format.space_before = Pt(1)
            inline(para, values[col] if col < len(values) else "")
            for run in para.runs:
                run.font.name = "Aptos"
                run.font.size = Pt(8)
                if row_index == 0:
                    run.bold = True
                    run.font.color.rgb = RGBColor.from_string("FFFFFF")
            if row_index == 0:
                shade(cell, NAVY)
            elif row_index % 2 == 0:
                shade(cell, "F3F7FB")
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def build() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(.68)
    section.bottom_margin = Inches(.65)
    section.left_margin = Inches(.72)
    section.right_margin = Inches(.72)
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9.2)
    normal.font.color.rgb = RGBColor.from_string(DARK)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.06
    for style_name, size, color in [("Title", 29, NAVY), ("Heading 1", 18, NAVY),
                                    ("Heading 2", 13.5, BLUE), ("Heading 3", 11, NAVY)]:
        style = doc.styles[style_name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(5)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = header.add_run("DOCUMIND  /  ENGINEERING REFERENCE")
    r.font.size = Pt(8)
    r.bold = True
    r.font.color.rgb = RGBColor.from_string(MUTED)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer.add_run("LangGraph RAG Pipeline Deep Dive  •  ")
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor.from_string(MUTED)
    page = OxmlElement("w:fldSimple")
    page.set(qn("w:instr"), "PAGE")
    footer._p.append(page)

    i = 0
    seen_title = False
    cover_break_done = False
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Inches(.15)
            para.paragraph_format.right_indent = Inches(.08)
            para.paragraph_format.space_before = Pt(3)
            para.paragraph_format.space_after = Pt(8)
            para.paragraph_format.line_spacing = 1.0
            props = para._p.get_or_add_pPr()
            bg = OxmlElement("w:shd")
            bg.set(qn("w:fill"), "F1F4F7")
            props.append(bg)
            for line_no, code_line in enumerate(block):
                if line_no:
                    para.add_run().add_break()
                run = para.add_run(code_line)
                run.font.name = "Cascadia Mono"
                run.font.size = Pt(7.8)
            i += 1
            continue
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            add_table(doc, table_lines)
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if level == 1 and not seen_title:
                para = doc.add_paragraph(style="Title")
                inline(para, text)
                seen_title = True
            else:
                if not cover_break_done and text == "Contents":
                    doc.add_page_break()
                    cover_break_done = True
                doc.add_heading(text, level=min(level, 3))
            i += 1
            continue
        if stripped.startswith("> "):
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Inches(.15)
            para.paragraph_format.right_indent = Inches(.1)
            para.paragraph_format.space_before = Pt(4)
            para.paragraph_format.space_after = Pt(8)
            props = para._p.get_or_add_pPr()
            bg = OxmlElement("w:shd")
            bg.set(qn("w:fill"), "EAF1F8")
            props.append(bg)
            inline(para, stripped[2:])
            i += 1
            continue
        if stripped.startswith("- "):
            para = doc.add_paragraph(style="List Bullet")
            para.paragraph_format.space_after = Pt(2)
            inline(para, stripped[2:])
            i += 1
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if numbered:
            para = doc.add_paragraph(style="List Number")
            para.paragraph_format.space_after = Pt(2)
            inline(para, numbered.group(2))
            i += 1
            continue
        para = doc.add_paragraph()
        inline(para, stripped.replace("  ", " "))
        i += 1

    doc.core_properties.title = "LangGraph RAG Pipeline Deep Dive"
    doc.core_properties.subject = "DocuMind graph stages, routing, edge cases, and code review"
    doc.core_properties.author = "OpenAI Codex"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
