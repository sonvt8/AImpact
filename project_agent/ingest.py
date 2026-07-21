import csv
import re
from dataclasses import dataclass

import openpyxl
from openpyxl.utils import get_column_letter


@dataclass
class Record:
    source_type: str
    locator: str
    text_verbatim: str
    sheet_name: str
    row_index: int | None
    headers: list[str]
    fields: dict[str, object]
    section_path: list[str]
    stt: object | None
    is_section: bool


def parse_file(path) -> list[Record]:
    extension = str(path).lower().rsplit(".", 1)[-1]
    if extension == "pdf":
        return parse_pdf(path)
    if extension == "docx":
        return parse_docx(path)
    if extension == "xlsx":
        return parse_xlsx(path)
    if extension == "txt":
        return parse_txt(path)
    if extension == "csv":
        return parse_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def parse_pdf(path) -> list[Record]:
    import pdfplumber

    records = []
    with pdfplumber.open(path) as document:
        for page_number, page in enumerate(document.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                try:
                    import pytesseract

                    text = pytesseract.image_to_string(page.to_image().original) or ""
                except Exception:
                    text = ""
            if text.strip():
                records.append(
                    Record(
                        source_type="pdf",
                        locator=f"p.{page_number}",
                        text_verbatim=text,
                        sheet_name="",
                        row_index=page_number,
                        headers=[],
                        fields={},
                        section_path=[],
                        stt=None,
                        is_section=False,
                    )
                )
    return records


def parse_docx(path) -> list[Record]:
    import docx

    document = docx.Document(path)
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    parts.extend(
        cell.text
        for table in document.tables
        for row in table.rows
        for cell in row.cells
        if cell.text.strip()
    )
    text = "\n".join(parts)
    if not text:
        return []
    return [
        Record(
            source_type="docx",
            locator=f"chars:0-{len(text)}",
            text_verbatim=text,
            sheet_name="",
            row_index=None,
            headers=[],
            fields={},
            section_path=[],
            stt=None,
            is_section=False,
        )
    ]


def parse_xlsx(path) -> list[Record]:
    workbook = openpyxl.load_workbook(path, data_only=False)
    records = []
    try:
        for worksheet in workbook.worksheets:
            records.extend(_parse_worksheet(worksheet))
    finally:
        workbook.close()
    return records


def parse_txt(path) -> list[Record]:
    with open(path, "r", encoding="utf-8", newline="") as source:
        text = source.read()
    return [
        Record(
            source_type="txt",
            locator=f"chars:0-{len(text)}",
            text_verbatim=text,
            sheet_name="",
            row_index=None,
            headers=[],
            fields={},
            section_path=[],
            stt=None,
            is_section=False,
        )
    ]


def parse_csv(path) -> list[Record]:
    with open(path, "r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))
        if not rows:
            return []
        header_row = rows[0]
        width = max(len(row) for row in rows)
        header_row += [None] * (width - len(header_row))
        headers = _unique_headers(header_row, 1)
        records = []
        for row_index, row in enumerate(rows[1:], start=2):
            values = row[: len(headers)] + [""] * max(0, len(headers) - len(row))
            fields = dict(zip(headers, values))
            records.append(
                Record(
                    source_type="csv",
                    locator=f"row:{row_index}",
                    text_verbatim=_record_text(headers, values, []),
                    sheet_name="",
                    row_index=row_index,
                    headers=list(headers),
                    fields=fields,
                    section_path=[],
                    stt=values[0] if values else None,
                    is_section=False,
                )
            )
    return records


def _parse_worksheet(worksheet) -> list[Record]:
    header_row = _find_header_row(worksheet)
    merged_values = _merged_values(worksheet)
    used_columns = [
        column
        for column in range(1, worksheet.max_column + 1)
        if any(
            value is not None and str(value).strip()
            for value in (
                _cell_value(worksheet, row, column, merged_values)
                for row in range(header_row, worksheet.max_row + 1)
            )
        )
    ]
    if not used_columns:
        return []

    start_column, end_column = min(used_columns), max(used_columns)
    header_values = [
        _cell_value(worksheet, header_row, column, merged_values)
        for column in range(start_column, end_column + 1)
    ]
    headers = _unique_headers(header_values, start_column)
    section_path = []
    records = []

    for row_index in range(header_row + 1, worksheet.max_row + 1):
        values = [
            _cell_value(worksheet, row_index, column, merged_values)
            for column in range(start_column, end_column + 1)
        ]
        if not any(value is not None and str(value).strip() for value in values):
            continue

        nonnull = [(index, value) for index, value in enumerate(values) if value is not None]
        distinct_nonfirst = {
            str(value).strip() for index, value in nonnull if index != 0
        }
        is_section = len(distinct_nonfirst) <= 1
        fields = {
            header: "" if value is None else value
            for header, value in zip(headers, values)
        }
        locator = (
            f"{worksheet.title}!{get_column_letter(start_column)}{row_index}:"
            f"{get_column_letter(end_column)}{row_index}"
        )

        if is_section:
            label = _section_label(values)
            level = _section_level(values[0])
            section_path = section_path[:level] + [label]
            text_verbatim = label
        else:
            text_verbatim = _record_text(headers, values, section_path)

        records.append(
            Record(
                source_type="xlsx",
                locator=locator,
                text_verbatim=text_verbatim,
                sheet_name=worksheet.title,
                row_index=row_index,
                headers=list(headers),
                fields=fields,
                section_path=list(section_path),
                stt=values[0] if values else None,
                is_section=is_section,
            )
        )
    return records


def _find_header_row(worksheet) -> int:
    rows = range(1, min(8, worksheet.max_row) + 1)
    return max(
        rows,
        key=lambda row: sum(
            isinstance(worksheet.cell(row, column).value, str)
            and bool(worksheet.cell(row, column).value.strip())
            for column in range(1, worksheet.max_column + 1)
        ),
    )


def _merged_values(worksheet) -> dict[tuple[int, int], object]:
    values = {}
    for merged_range in worksheet.merged_cells.ranges:
        anchor = worksheet.cell(merged_range.min_row, merged_range.min_col).value
        for row in range(merged_range.min_row, merged_range.max_row + 1):
            for column in range(merged_range.min_col, merged_range.max_col + 1):
                values[(row, column)] = anchor
    return values


def _cell_value(worksheet, row: int, column: int, merged_values):
    cell = worksheet.cell(row, column)
    if cell.value is None and (row, column) in merged_values:
        return merged_values[(row, column)]
    if cell.value is None and cell.data_type == "inlineStr":
        return ""
    return cell.value


def _unique_headers(values, start_column: int) -> list[str]:
    counts = {}
    headers = []
    for offset, value in enumerate(values):
        base = str(value) if value is not None and str(value).strip() else get_column_letter(start_column + offset)
        counts[base] = counts.get(base, 0) + 1
        headers.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return headers


def _section_label(values) -> str:
    parts = []
    seen = set()
    for value in values:
        if value is None or not str(value).strip():
            continue
        key = str(value).strip()
        if key not in seen:
            seen.add(key)
            parts.append(str(value))
    return " | ".join(parts)


def _section_level(value) -> int:
    outline = "" if value is None else str(value).strip()
    if re.fullmatch(r"[IVXLC]+", outline, re.IGNORECASE):
        return 0
    if re.fullmatch(r"[IVXLC0-9]+(?:\.\d+)+", outline, re.IGNORECASE):
        return outline.count(".")
    return 0


def _record_text(headers, values, section_path) -> str:
    body = "\n".join(
        f"{header}: {'' if value is None else value}"
        for header, value in zip(headers, values)
    )
    return "\n".join([*section_path, body]) if section_path else body
