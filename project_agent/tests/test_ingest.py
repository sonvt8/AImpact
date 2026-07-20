import re
from collections import Counter
from pathlib import Path

import openpyxl
import pytest

import ingest


SAMPLE_XLSX = Path(__file__).resolve().parents[1] / "Phu luc 1.xlsx"


@pytest.fixture(scope="module")
def sample_records():
    return ingest.parse_file(SAMPLE_XLSX)


def test_xlsx_counts(sample_records):
    counts = Counter(
        record.sheet_name for record in sample_records if not record.is_section
    )

    assert counts["1.Ds UCTT"] == 61
    assert counts["Form XLSC"] == 50
    assert counts["3. Ds VHKT"] == 6


def test_verbatim_no_mapping(sample_records):
    record = next(
        record
        for record in sample_records
        if record.sheet_name == "1.Ds UCTT"
        and not record.is_section
        and "Sự cố 1 lộ điện lưới" in record.text_verbatim
    )

    assert "Sự cố 1 lộ điện lưới" in record.text_verbatim
    assert "lỗi 1 lộ điện lưới" not in record.text_verbatim.lower()
    assert record.section_path
    assert record.text_verbatim.startswith("\n".join(record.section_path))


def test_locator_cellrange(sample_records):
    record = next(
        record
        for record in sample_records
        if record.sheet_name == "1.Ds UCTT" and not record.is_section
    )

    assert re.fullmatch(r"1\.Ds UCTT!A\d+:[A-Z]+\d+", record.locator)


def test_generic_and_no_ffill(tmp_path):
    path = tmp_path / "generic.xlsx"
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(["id", "name", "note"])
    worksheet.append([1, "Alpha", "n1"])
    worksheet.append([2, "Beta", ""])
    worksheet.append([3, "Gamma", "n3"])
    workbook.save(path)

    records = ingest.parse_file(path)
    data_records = [record for record in records if not record.is_section]

    assert len(data_records) == 3
    assert not any(record.is_section for record in records)
    assert data_records[0].headers == ["id", "name", "note"]
    beta = next(record for record in data_records if record.fields["id"] == 2)
    assert beta.fields["note"] == ""


def test_txt_and_csv(tmp_path):
    txt_path = tmp_path / "sample.txt"
    txt_path.write_bytes("A\r\nB".encode("utf-8"))
    txt_record = ingest.parse_file(txt_path)[0]
    assert txt_record.text_verbatim == "A\r\nB"
    assert txt_record.locator == "chars:0-4"

    csv_path = tmp_path / "sample.csv"
    csv_path.write_bytes(b"id,name,note\r\n1,Alpha,\r\n")
    csv_record = ingest.parse_file(csv_path)[0]
    assert csv_record.locator == "row:2"
    assert csv_record.fields == {"id": "1", "name": "Alpha", "note": ""}
