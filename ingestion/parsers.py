from __future__ import annotations

import re
import csv
import json
from pathlib import Path
from urllib.parse import urlparse

from .models import ContentBlock, ParsedDocument, Section

SUPPORTED_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx",
    ".pptx", ".xlsx", ".csv", ".json",
}


def _markdown_table(rows: list[list[object]]) -> str:
    cleaned = [[re.sub(r"\s+", " ", str(value if value is not None else "")).strip().replace("|", "\\|")
                for value in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    width = max(map(len, cleaned))
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    return "\n".join(["| " + " | ".join(cleaned[0]) + " |",
                      "| " + " | ".join(["---"] * width) + " |",
                      *["| " + " | ".join(row) + " |" for row in cleaned[1:]]])


def _parse_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf; install requirements.txt") from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("This PDF is password-protected and cannot be ingested")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"## Page {number}\n\n{text}")
        if not pages:
            raise ValueError("No selectable text was found in this PDF. Scanned PDFs need OCR before upload.")
        return "\n\n".join(pages)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {type(exc).__name__}") from exc


def _parse_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("Word document ingestion requires python-docx; install requirements.txt") from exc
    try:
        document = Document(str(path))
        blocks: list[str] = []
        for item in document.iter_inner_content():
            if hasattr(item, "text"):
                text = item.text.strip()
                if not text:
                    continue
                style = item.style.name if item.style else ""
                heading = re.match(r"Heading\s+(\d+)", style, re.IGNORECASE)
                if heading:
                    blocks.append(f"{'#' * min(int(heading.group(1)), 6)} {text}")
                elif "list bullet" in style.casefold():
                    blocks.append(f"- {text}")
                elif "list number" in style.casefold():
                    blocks.append(f"1. {text}")
                elif "code" in style.casefold() or any(run.font.name in {"Consolas", "Courier New"} for run in item.runs):
                    blocks.append(f"```text\n{text}\n```")
                else:
                    blocks.append(text)
            else:
                table = _markdown_table([[cell.text for cell in row.cells] for row in item.rows])
                if table:
                    blocks.append(table)
        return "\n\n".join(blocks)
    except Exception as exc:
        raise ValueError(f"Could not read Word document: {type(exc).__name__}") from exc


def _parse_pptx(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError("PowerPoint ingestion requires python-pptx; install requirements.txt") from exc
    try:
        presentation = Presentation(str(path))
        slides = []
        for number, slide in enumerate(presentation.slides, start=1):
            blocks = [f"## Slide {number}"]
            for shape in slide.shapes:
                if getattr(shape, "has_table", False):
                    table = _markdown_table([[cell.text for cell in row.cells] for row in shape.table.rows])
                    if table:
                        blocks.append(table)
                elif getattr(shape, "has_text_frame", False):
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if not text:
                            continue
                        if shape == slide.shapes.title:
                            blocks.append(f"### {text}")
                        elif paragraph.level:
                            blocks.append(f"{'  ' * paragraph.level}- {text}")
                        else:
                            blocks.append(text)
            slides.append("\n\n".join(blocks))
        return "\n\n".join(slides)
    except Exception as exc:
        raise ValueError(f"Could not read PowerPoint document: {type(exc).__name__}") from exc


def _parse_xlsx(path: Path) -> str:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("Excel ingestion requires openpyxl; install requirements.txt") from exc
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
        sheets = []
        for sheet in workbook.worksheets:
            table = _markdown_table([["" if value is None else value for value in row]
                                     for row in sheet.iter_rows(values_only=True)])
            if table:
                sheets.append(f"## {sheet.title}\n\n{table}")
        workbook.close()
        return "\n\n".join(sheets)
    except Exception as exc:
        raise ValueError(f"Could not read Excel workbook: {type(exc).__name__}") from exc


def _parse_csv(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            sample = source.read(8192)
            source.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            return _markdown_table(list(csv.reader(source, dialect)))
    except UnicodeDecodeError as exc:
        raise ValueError("CSV document must be UTF-8 encoded") from exc


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must be an absolute HTTP or HTTPS URL")


def _markdown_blocks(text: str) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    headings: list[str] = []
    paragraph: list[str] = []
    code: list[str] | None = None
    fence = ""

    def current_section() -> Section:
        return Section(headings[-1] if headings else "Document", len(headings), headings.copy())

    def flush_paragraph() -> None:
        if paragraph:
            joined = "\n".join(paragraph).strip()
            if joined:
                kind = "list" if re.match(r"^\s*(?:[-*+]\s|\d+[.)]\s)", joined) else ("table" if "|" in joined and "\n" in joined else "paragraph")
                blocks.append(ContentBlock(kind, joined, current_section()))
            paragraph.clear()

    for line in text.splitlines():
        fence_match = re.match(r"^\s*(```+|~~~+)", line)
        if code is not None:
            code.append(line)
            if fence_match and fence_match.group(1)[0] == fence[0] and len(fence_match.group(1)) >= len(fence):
                blocks.append(ContentBlock("code", "\n".join(code), current_section()))
                code, fence = None, ""
            continue
        if fence_match:
            flush_paragraph()
            fence = fence_match.group(1)
            code = [line]
            continue
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush_paragraph()
            level, title = len(heading.group(1)), heading.group(2).strip()
            headings = headings[: level - 1] + [title]
            blocks.append(ContentBlock("heading", title, current_section()))
        elif not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line.rstrip())
    if code is not None:
        blocks.append(ContentBlock("code", "\n".join(code), current_section()))
    flush_paragraph()
    return blocks


def _html_to_markdown(source: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("HTML parsing requires beautifulsoup4; install requirements.txt") from exc
    soup = BeautifulSoup(source, "html.parser")
    for node in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer"]):
        node.decompose()
    for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(node.name[1])
        node.replace_with(f"\n{'#' * level} {node.get_text(' ', strip=True)}\n")
    for node in soup.find_all("pre"):
        language = ""
        code = node.get_text("\n", strip=True)
        node.replace_with(f"\n```{language}\n{code}\n```\n")
    for node in soup.find_all("table"):
        rows = []
        for row in node.find_all("tr"):
            cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in row.find_all(["th", "td"], recursive=False)]
            if cells:
                rows.append(cells)
        if rows:
            width = max(map(len, rows))
            rows = [row + [""] * (width - len(row)) for row in rows]
            header = rows[0]
            rendered = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
            rendered.extend("| " + " | ".join(row) + " |" for row in rows[1:])
            node.replace_with("\n" + "\n".join(rendered) + "\n")
    for node in soup.find_all("li"):
        node.insert_before("\n- ")
    for node in soup.find_all(["p", "div", "section", "article", "br"]):
        node.insert_before("\n")
        node.insert_after("\n")
    return soup.get_text(" ", strip=True).replace("\xa0", " ")


def parse_text(text: str, filename: str, source_url: str | None = None) -> ParsedDocument:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        raise ValueError("Document is empty after parsing")
    return ParsedDocument(cleaned, _markdown_blocks(cleaned), filename, source_url)


def parse_file(path: str | Path) -> ParsedDocument:
    file_path = Path(path)
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext or '(none)'}'; supported: PDF, DOCX, PPTX, XLSX, CSV, JSON, Markdown, TXT, and HTML")
    if ext == ".pdf":
        return parse_text(_parse_pdf(file_path), file_path.name)
    if ext == ".docx":
        return parse_text(_parse_docx(file_path), file_path.name)
    if ext == ".pptx":
        return parse_text(_parse_pptx(file_path), file_path.name)
    if ext == ".xlsx":
        return parse_text(_parse_xlsx(file_path), file_path.name)
    if ext == ".csv":
        return parse_text(_parse_csv(file_path), file_path.name)
    try:
        content = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Document must be UTF-8 encoded") from exc
    if ext in {".html", ".htm"}:
        content = _html_to_markdown(content)
    elif ext == ".json":
        try:
            content = json.dumps(json.loads(content), ensure_ascii=False, indent=2)
        except json.JSONDecodeError as exc:
            raise ValueError("JSON document is malformed") from exc
    return parse_text(content, file_path.name)


def parse_url(url: str, timeout: int = 20) -> ParsedDocument:
    validate_url(url)
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("URL ingestion requires requests; install requirements.txt") from exc
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "TechnicalDocumentationAssistant/1.0"})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"Could not fetch URL: {exc}") from exc
    content_type = response.headers.get("Content-Type", "").lower()
    if "html" in content_type or not content_type:
        content = _html_to_markdown(response.text)
    else:
        content = response.text
    filename = Path(urlparse(url).path).name or urlparse(url).netloc
    return parse_text(content, filename, url)
