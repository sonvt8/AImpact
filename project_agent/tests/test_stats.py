from pathlib import Path

import openpyxl

from stats import column_totals, read_table


SAMPLE_XLSX = Path(__file__).resolve().parents[1] / "Phu luc 1.xlsx"


def test_read_tong_hop():
    rows = read_table(SAMPLE_XLSX, "Tong hop")

    assert len(rows) == 5
    assert next(row for row in rows if row["SỰ CỐ"] == "AC") == {
        "STT": 1,
        "SỰ CỐ": "AC",
        "UCTT": 28,
        "XLSC": 27,
        "VHKT": 6,
        "TỔNG": 61,
    }


def test_formula_values_resolved():
    rows = read_table(SAMPLE_XLSX, "Tong hop")

    assert [row["TỔNG"] for row in rows] == [61, 17, 18, 16, 5]


def test_column_totals():
    totals = column_totals(read_table(SAMPLE_XLSX, "Tong hop"))

    assert totals["UCTT"] == 61
    assert totals["XLSC"] == 50
    assert totals["VHKT"] == 6
    assert totals["TỔNG"] == 117


def test_blank_cells_preserved():
    rows = read_table(SAMPLE_XLSX, "Tong hop")

    assert next(row for row in rows if row["SỰ CỐ"] == "DC")["VHKT"] is None


def test_read_generic_synthetic(tmp_path):
    path = tmp_path / "generic.xlsx"
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(["a", "b"])
    worksheet.append([1, 2])
    worksheet.append([3, 4])
    sheet_name = worksheet.title
    workbook.save(path)
    workbook.close()

    rows = read_table(path, sheet_name)

    assert rows == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert column_totals(rows) == {"a": 4, "b": 6}
