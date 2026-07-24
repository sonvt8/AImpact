import gc
from dataclasses import replace

import openpyxl

import index
import ingest


def create_index(tmp_path, **kwargs):
    return index.VectorIndex(
        persist_directory=tmp_path / "chroma",
        key_path=tmp_path / "encryption.key",
        embedding_model_id="test-embedding",
        **kwargs,
    )


def constant_embed(texts):
    return [[1.0, 0.0] for _ in texts]


def record(sheet_name="Data", locator="Data!A2:B2", text="Code: AC\nValue: 6"):
    return ingest.Record(
        source_type="xlsx",
        locator=locator,
        text_verbatim=text,
        sheet_name=sheet_name,
        row_index=2,
        headers=["Code", "Value"],
        fields={"Code": "AC", "Value": 6},
        section_path=[],
        stt="AC",
        is_section=False,
    )


def test_form_sheets_still_parse_but_are_excluded_by_default(tmp_path):
    workbook = openpyxl.Workbook()
    workbook.active.title = "Data"
    workbook.active.append(["Code", "Value"])
    workbook.active.append(["AC", 6])
    form = workbook.create_sheet("Form incident")
    form.append(["Code", "Value"])
    form.append(["AC", 6])
    workbook_path = tmp_path / "sample.xlsx"
    workbook.save(workbook_path)
    workbook.close()

    records = ingest.parse_file(workbook_path)
    assert {item.sheet_name for item in records} == {"Data", "Form incident"}

    vector_index = create_index(tmp_path)
    try:
        result = vector_index.add_records(
            records,
            filename=workbook_path.name,
            doc_checksum="form-filter",
            embed_fn=constant_embed,
            parser_version="test-parser-v1",
        )
        metadata = vector_index.collection.get(include=["metadatas"])["metadatas"]

        assert result["added"] == 1
        assert [item["sheet_name"] for item in metadata] == ["Data"]
    finally:
        del vector_index
        gc.collect()


def test_exact_structured_duplicates_are_removed_within_a_sheet(tmp_path):
    original = record()
    duplicate = replace(original, locator="Data!A3:B3", row_index=3)
    other_sheet = replace(original, sheet_name="Archive", locator="Archive!A2:B2")
    vector_index = create_index(tmp_path, excluded_sheet_prefixes=())
    try:
        result = vector_index.add_records(
            [original, duplicate, other_sheet],
            filename="sample.xlsx",
            doc_checksum="dedupe",
            embed_fn=constant_embed,
            parser_version="test-parser-v1",
        )
        metadata = vector_index.collection.get(include=["metadatas"])["metadatas"]

        assert result["added"] == 2
        assert {item["locator"] for item in metadata} == {
            original.locator,
            other_sheet.locator,
        }
    finally:
        del vector_index
        gc.collect()


def test_embedding_uses_projection_and_query_returns_encrypted_verbatim(tmp_path):
    original = replace(
        record(text="Nhóm sự cố\nCode: AC\nValue: 6\nEmpty: "),
        sheet_name="Tong hop",
        locator="Tong hop!A2:C2",
        headers=["Code", "Value", "Empty"],
        fields={"Code": " AC ", "Value": 6, "Empty": ""},
        section_path=["Nhóm sự cố"],
    )
    embedded = []

    def capture_embed(texts):
        embedded.append(list(texts))
        return constant_embed(texts)

    vector_index = create_index(tmp_path)
    try:
        vector_index.add_records(
            [original],
            filename="sample.xlsx",
            doc_checksum="projection",
            embed_fn=capture_embed,
            parser_version="test-parser-v1",
        )
        stored = vector_index.collection.get(include=["documents", "metadatas"])
        result = vector_index.query("AC", capture_embed, top_k=1)[0]

        assert embedded[0] == ["Tong hop\nNhóm sự cố\nCode: AC\nValue: 6"]
        assert original.text_verbatim not in stored["documents"][0]
        assert result["text_verbatim"] == original.text_verbatim
        assert stored["metadatas"][0]["retrieval_projection_version"] == "fields-v1"
    finally:
        del vector_index
        gc.collect()
