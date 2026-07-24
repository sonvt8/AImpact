NO_EVIDENCE_MESSAGE = "Không tìm thấy thông tin phù hợp trong tài liệu."


def select_evidence(hits, threshold, max_ctx=5) -> tuple[list[dict], bool]:
    kept = [
        hit
        for hit in hits
        if hit["similarity"] >= threshold and hit.get("lexical_gate", True)
    ][: max(0, max_ctx)]
    return kept, bool(kept)


def build_context(kept) -> str:
    parts = []
    for hit in kept:
        metadata = hit["metadata"]
        parts.append(
            "[Nguồn: {filename} | Sheet: {sheet_name} | Vị trí: {locator} | "
            "Tương đồng: {similarity:.4f}]\n{text}".format(
                filename=metadata.get("filename", ""),
                sheet_name=metadata.get("sheet_name", ""),
                locator=metadata.get("locator", ""),
                similarity=hit["similarity"],
                text=hit["text_verbatim"],
            )
        )
    return "\n\n".join(parts)


def build_citations(kept) -> list[dict]:
    return [
        {
            "filename": hit["metadata"].get("filename", ""),
            "sheet_name": hit["metadata"].get("sheet_name", ""),
            "locator": hit["metadata"].get("locator", ""),
            "row_index": hit["metadata"].get("row_index", 0),
            "section_path": hit["metadata"].get("section_path", ""),
            "stt": hit["metadata"].get("stt", ""),
            "similarity": hit["similarity"],
            "content": hit["text_verbatim"],
        }
        for hit in kept
    ]


def build_prompt(
    question,
    context,
    citations,
    history,
    role,
    is_summary=False,
    answer_format=None,
) -> str:
    citation_text = "\n".join(
        "- {filename} | Sheet: {sheet_name} | Vị trí: {locator}".format(**citation)
        for citation in citations
    ) or "- Không có trích dẫn"
    task = "Tóm tắt thông tin để trả lời câu hỏi." if is_summary else "Trả lời câu hỏi."
    format_instruction = (
        f"Định dạng trả lời:\n{answer_format}"
        if answer_format is not None
        else "Trả lời mạch lạc, ngắn gọn và trực tiếp vào câu hỏi."
    )
    return f"""Vai trò: {role}
Nhiệm vụ: {task}

Lịch sử hội thoại:
{history}

Ngữ cảnh tài liệu:
{context}

Trích dẫn:
{citation_text}

Câu hỏi:
{question}

Hướng dẫn:
- Chỉ sử dụng thông tin có trong ngữ cảnh tài liệu.
- Không suy diễn hoặc bổ sung thông tin ngoài tài liệu.
- Dùng trích dẫn để xác định nguồn và vị trí của bằng chứng.
- Nếu ngữ cảnh không đủ bằng chứng, chỉ trả đúng: {NO_EVIDENCE_MESSAGE}
- {format_instruction}

Trả lời:"""


def answer_or_refuse(
    question,
    hits,
    threshold,
    history,
    role,
    is_summary=False,
    answer_format=None,
) -> tuple[str | None, list[dict]]:
    kept, has_evidence = select_evidence(hits, threshold)
    if not has_evidence:
        return None, []
    context = build_context(kept)
    citations = build_citations(kept)
    prompt = build_prompt(
        question,
        context,
        citations,
        history,
        role,
        is_summary=is_summary,
        answer_format=answer_format,
    )
    return prompt, citations
