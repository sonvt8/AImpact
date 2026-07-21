from pathlib import Path

import docx
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

import ingest


FONT_PATH = Path(__file__).resolve().parents[1] / "fonts" / "DejaVuSans.ttf"
PDF_FONT = "IngestTestDejaVu"
pdfmetrics.registerFont(TTFont(PDF_FONT, FONT_PATH))


def write_pdf(path, text):
    document = canvas.Canvas(str(path))
    document.setFont(PDF_FONT, 12)
    document.drawString(72, 720, text)
    document.save()


def write_docx(path, text):
    document = docx.Document()
    document.add_paragraph(text)
    document.save(path)


def test_parse_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    write_pdf(path, "Xin chào RAG kỹ thuật")

    records = ingest.parse_file(path)

    assert len(records) == 1
    assert records[0].source_type == "pdf"
    assert records[0].locator == "p.1"
    assert "Xin chào RAG kỹ thuật" in records[0].text_verbatim
    assert records[0].is_section is False


def test_parse_docx(tmp_path):
    path = tmp_path / "sample.docx"
    write_docx(path, "Nội dung DOCX có dấu")

    records = ingest.parse_file(path)

    assert len(records) == 1
    assert records[0].source_type == "docx"
    assert "Nội dung DOCX có dấu" in records[0].text_verbatim


def test_dispatch_no_longer_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "parse_pdf", lambda path: [])
    monkeypatch.setattr(ingest, "parse_docx", lambda path: [])

    assert ingest.parse_file(tmp_path / "sample.pdf") == []
    assert ingest.parse_file(tmp_path / "sample.docx") == []


def test_verbatim_not_normalized(tmp_path):
    text = "GiỮ Nguyên DẤU và Hoa/Thường"
    pdf_path = tmp_path / "verbatim.pdf"
    docx_path = tmp_path / "verbatim.docx"
    write_pdf(pdf_path, text)
    write_docx(docx_path, text)

    assert text in ingest.parse_file(pdf_path)[0].text_verbatim
    assert text in ingest.parse_file(docx_path)[0].text_verbatim
