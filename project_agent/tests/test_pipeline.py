import pipeline
import rag


def make_hit(similarity=0.9):
    return {
        "text_verbatim": "Nội dung nguyên văn",
        "metadata": {
            "filename": "sample.xlsx",
            "sheet_name": "Sheet A",
            "locator": "Sheet A!A2:C2",
            "row_index": 2,
            "section_path": "Section",
            "stt": "1",
        },
        "similarity": similarity,
    }


def test_no_evidence_does_not_call_llm():
    generator_calls = []

    def generator_fn(prompt):
        generator_calls.append(prompt)
        return "unexpected"

    for hits in ([make_hit(0.49)], []):
        result = pipeline.run_query(
            "Câu hỏi",
            retrieve_fn=lambda query, top_k, where, hits=hits: hits,
            generator_fn=generator_fn,
            history="",
            threshold=0.5,
            role="user",
        )

        assert result == {
            "answer": rag.NO_EVIDENCE_MESSAGE,
            "citations": [],
            "llm_called": False,
        }
    assert generator_calls == []


def test_evidence_calls_llm_with_history_and_citations():
    captured_prompts = []
    hit = make_hit()

    def generator_fn(prompt):
        captured_prompts.append(prompt)
        return "OK"

    result = pipeline.run_query(
        "Câu hỏi",
        retrieve_fn=lambda query, top_k, where: [hit],
        generator_fn=generator_fn,
        history="Lịch sử cần giữ",
        threshold=0.5,
        role="user",
    )

    assert result["answer"] == "OK"
    assert result["llm_called"] is True
    assert len(captured_prompts) == 1
    assert "Lịch sử cần giữ" in captured_prompts[0]
    assert hit["metadata"]["locator"] in captured_prompts[0]
    assert result["citations"][0]["locator"] == hit["metadata"]["locator"]


def test_where_and_topk_forwarded():
    captured = {}

    def retrieve_fn(query, top_k, where):
        captured.update(query=query, top_k=top_k, where=where)
        return []

    pipeline.run_query(
        "Câu hỏi lọc",
        retrieve_fn=retrieve_fn,
        generator_fn=lambda prompt: "unexpected",
        history="",
        threshold=0.5,
        role="user",
        top_k=7,
        where={"sheet_name": "Sheet A"},
    )

    assert captured == {
        "query": "Câu hỏi lọc",
        "top_k": 7,
        "where": {"sheet_name": "Sheet A"},
    }


def test_ingest_file_calls_add_with_parsed_records():
    records = [object(), object()]
    captured = {}
    embed_fn = object()

    def parse_fn(path):
        captured["path"] = path
        return records

    def add_fn(*args):
        captured["add_args"] = args
        return {"status": "added", "added": 2, "version": 1}

    result = pipeline.ingest_file(
        parse_fn=parse_fn,
        add_fn=add_fn,
        path="document.xlsx",
        filename="document.xlsx",
        doc_checksum="checksum",
        embed_fn=embed_fn,
        parser_version="parser-v1",
    )

    assert result == {"status": "added", "added": 2, "version": 1}
    assert captured["path"] == "document.xlsx"
    assert captured["add_args"] == (
        records,
        "document.xlsx",
        "checksum",
        embed_fn,
        "parser-v1",
    )
