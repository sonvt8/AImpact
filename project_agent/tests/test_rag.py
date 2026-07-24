import rag


def make_hit(number, similarity, text=None):
    return {
        "text_verbatim": text if text is not None else f"Nội dung {number}",
        "metadata": {
            "filename": "sample.xlsx",
            "sheet_name": "Sheet A",
            "locator": f"Sheet A!A{number}:C{number}",
            "row_index": number,
            "section_path": "Section",
            "stt": str(number),
        },
        "similarity": similarity,
    }


def test_no_answer_gate():
    below_threshold = [make_hit(1, 0.49), make_hit(2, 0.10)]

    assert rag.answer_or_refuse(
        "Câu hỏi",
        below_threshold,
        threshold=0.5,
        history="",
        role="user",
    ) == (None, [])
    assert rag.answer_or_refuse(
        "Câu hỏi",
        [],
        threshold=0.5,
        history="",
        role="user",
    ) == (None, [])


def test_lexical_gate_is_required_but_defaults_true():
    blocked = make_hit(1, 0.99)
    blocked["lexical_gate"] = False
    compatible = make_hit(2, 0.90)

    kept, has_evidence = rag.select_evidence(
        [blocked, compatible],
        threshold=0.5,
    )

    assert kept == [compatible]
    assert has_evidence is True


def test_evidence_selected_and_capped():
    hits = [make_hit(index, 0.99 - index * 0.01) for index in range(8)]

    prompt, citations = rag.answer_or_refuse(
        "Câu hỏi",
        hits,
        threshold=0.5,
        history="Lịch sử",
        role="user",
    )

    assert prompt is not None
    assert len(citations) == 5
    assert [item["similarity"] for item in citations] == [
        hit["similarity"] for hit in hits[:5]
    ]


def test_prompt_contains_history_and_citations():
    hit = make_hit(3, 0.9)
    context = rag.build_context([hit])
    citations = rag.build_citations([hit])

    prompt = rag.build_prompt(
        "Câu hỏi kiểm tra",
        context,
        citations,
        history="Lịch sử hội thoại cần giữ",
        role="user",
    )

    assert "Lịch sử hội thoại cần giữ" in prompt
    assert citations[0]["locator"] in prompt
    assert context in prompt
    assert rag.NO_EVIDENCE_MESSAGE in prompt


def test_context_has_provenance():
    hit = make_hit(4, 0.87654)

    context = rag.build_context([hit])

    assert hit["metadata"]["filename"] in context
    assert hit["metadata"]["sheet_name"] in context
    assert hit["metadata"]["locator"] in context


def test_citations_are_verbatim():
    verbatim = "  Nguyên văn có dấu\nDòng thứ hai  "
    hit = make_hit(5, 0.8, text=verbatim)

    citations = rag.build_citations([hit])

    assert citations[0]["content"] == verbatim
    assert citations[0]["locator"] == hit["metadata"]["locator"]


def test_answer_format_injected():
    prompt, citations = rag.answer_or_refuse(
        "Câu hỏi",
        [make_hit(6, 0.9)],
        threshold=0.5,
        history="",
        role="user",
        answer_format="X-Y-Z",
    )

    assert citations
    assert "X-Y-Z" in prompt
