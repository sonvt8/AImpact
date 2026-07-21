import gc
import hashlib
import re
from pathlib import Path

import pytest

import index
import ingest


WORKBOOK_PATH = Path(__file__).resolve().parents[1] / "Phu luc 1.xlsx"


def stub_embed(texts):
    return [
        [value / 255.0 for value in hashlib.sha256(text.encode("utf-8")).digest()[:8]]
        for text in texts
    ]


@pytest.fixture(scope="module")
def workbook_records():
    return ingest.parse_file(WORKBOOK_PATH)


def create_index(tmp_path, embedding_model_id="test-embedding"):
    return index.VectorIndex(
        persist_directory=tmp_path / "chroma",
        key_path=tmp_path / "encryption.key",
        embedding_model_id=embedding_model_id,
    )


def test_add_counts_and_metadata(tmp_path, workbook_records):
    vector_index = create_index(tmp_path)
    try:
        checksum = index.file_checksum(WORKBOOK_PATH)
        result = vector_index.add_records(
            workbook_records,
            filename=WORKBOOK_PATH.name,
            doc_checksum=checksum,
            embed_fn=stub_embed,
            parser_version="test-parser-v1",
        )

        assert result == {"status": "added", "added": 357, "version": 1}
        assert vector_index.count() == 357
        metadata = vector_index.collection.get(
            limit=1,
            include=["metadatas"],
        )["metadatas"][0]
        assert metadata["doc_checksum"] == checksum
        assert metadata["doc_version"] == 1
        assert metadata["parser_version"] == "test-parser-v1"
        assert metadata["embedding_model_id"] == "test-embedding"
        assert re.match(r"^.+!.+\d+:.+\d+$", metadata["locator"])
    finally:
        del vector_index
        gc.collect()


def test_verbatim_roundtrip(tmp_path, workbook_records):
    original = next(record for record in workbook_records if not record.is_section)
    vector_index = create_index(tmp_path)
    try:
        vector_index.add_records(
            [original],
            filename=WORKBOOK_PATH.name,
            doc_checksum="verbatim-checksum",
            embed_fn=stub_embed,
            parser_version="test-parser-v1",
        )

        results = vector_index.query(original.text_verbatim, stub_embed, top_k=1)

        assert results[0]["text_verbatim"] == original.text_verbatim
    finally:
        del vector_index
        gc.collect()


def test_cosine_metric(tmp_path):
    vector_index = create_index(tmp_path)
    try:
        assert vector_index.collection.metadata["hnsw:space"] == "cosine"
    finally:
        del vector_index
        gc.collect()


def test_no_allowlist_and_where(tmp_path, workbook_records):
    vector_index = create_index(tmp_path)
    try:
        vector_index.add_records(
            workbook_records,
            filename=WORKBOOK_PATH.name,
            doc_checksum="where-checksum",
            embed_fn=stub_embed,
            parser_version="test-parser-v1",
        )
        query_record = next(
            record
            for record in workbook_records
            if not record.is_section and record.sheet_name == "3. Ds VHKT"
        )

        unfiltered = vector_index.query(
            query_record.text_verbatim,
            stub_embed,
            top_k=50,
            where=None,
        )
        filtered = vector_index.query(
            query_record.text_verbatim,
            stub_embed,
            top_k=50,
            where={"sheet_name": "3. Ds VHKT"},
        )

        assert len({item["metadata"]["sheet_name"] for item in unfiltered}) > 1
        similarities = [item["similarity"] for item in unfiltered]
        assert all(0.0 <= similarity <= 1.0 for similarity in similarities)
        assert similarities == sorted(similarities, reverse=True)
        assert len(filtered) == 6
        assert all(
            item["metadata"]["sheet_name"] == "3. Ds VHKT"
            for item in filtered
        )
    finally:
        del vector_index
        gc.collect()


def test_reindex_on_checksum(tmp_path, workbook_records):
    records = [record for record in workbook_records if not record.is_section][:3]
    vector_index = create_index(tmp_path)
    try:
        first = vector_index.add_records(
            records,
            filename="sample.xlsx",
            doc_checksum="A",
            embed_fn=stub_embed,
            parser_version="test-parser-v1",
        )
        initial_count = vector_index.count()
        second = vector_index.add_records(
            records,
            filename="sample.xlsx",
            doc_checksum="B",
            embed_fn=stub_embed,
            parser_version="test-parser-v1",
        )
        count_after_reindex = vector_index.count()
        third = vector_index.add_records(
            records,
            filename="sample.xlsx",
            doc_checksum="B",
            embed_fn=stub_embed,
            parser_version="test-parser-v1",
        )
        metadata = vector_index.collection.get(
            where={"filename": "sample.xlsx"},
            include=["metadatas"],
        )["metadatas"]

        assert first == {"status": "added", "added": 3, "version": 1}
        assert initial_count == 3
        assert second == {"status": "reindexed", "added": 3, "version": 2}
        assert count_after_reindex == 3
        assert third == {"status": "skipped_same_checksum", "added": 0}
        assert vector_index.count() == 3
        assert all(item["doc_checksum"] == "B" for item in metadata)
        assert all(item["doc_version"] == 2 for item in metadata)
    finally:
        del vector_index
        gc.collect()


def test_embedding_model_guard(tmp_path, workbook_records):
    record = next(record for record in workbook_records if not record.is_section)
    vector_index = create_index(tmp_path, embedding_model_id="m1")
    vector_index.add_records(
        [record],
        filename="sample.xlsx",
        doc_checksum="guard-checksum",
        embed_fn=stub_embed,
        parser_version="test-parser-v1",
    )
    del vector_index
    gc.collect()

    with pytest.raises(ValueError, match="Embedding model mismatch"):
        create_index(tmp_path, embedding_model_id="m2")
    gc.collect()

def test_list_filenames_reset_delete(tmp_path, workbook_records):
    record = next(record for record in workbook_records if not record.is_section)
    vector_index = create_index(tmp_path)
    try:
        vector_index.add_records(
            [record],
            filename="A.xlsx",
            doc_checksum="A",
            embed_fn=stub_embed,
            parser_version="test-parser-v1",
        )
        vector_index.add_records(
            [record],
            filename="B.xlsx",
            doc_checksum="B",
            embed_fn=stub_embed,
            parser_version="test-parser-v1",
        )

        assert vector_index.list_filenames() == ["A.xlsx", "B.xlsx"]

        vector_index.delete_file("A.xlsx")

        assert vector_index.count() == 1
        assert vector_index.list_filenames() == ["B.xlsx"]

        vector_index.reset()

        assert vector_index.count() == 0
        assert vector_index.list_filenames() == []
        assert vector_index.collection.metadata["hnsw:space"] == "cosine"
    finally:
        del vector_index
        gc.collect()
