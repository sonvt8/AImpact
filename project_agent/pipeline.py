import rag


def run_query(
    query,
    *,
    retrieve_fn,
    generator_fn,
    history,
    threshold,
    role,
    top_k=10,
    where=None,
    is_summary=False,
    answer_format=None,
) -> dict:
    hits = retrieve_fn(query, top_k=top_k, where=where)
    prompt, citations = rag.answer_or_refuse(
        query,
        hits,
        threshold,
        history,
        role,
        is_summary=is_summary,
        answer_format=answer_format,
    )
    if prompt is None:
        return {
            "answer": rag.NO_EVIDENCE_MESSAGE,
            "citations": [],
            "llm_called": False,
        }
    return {
        "answer": generator_fn(prompt),
        "citations": citations,
        "llm_called": True,
    }


def ingest_file(
    *,
    parse_fn,
    add_fn,
    path,
    filename,
    doc_checksum,
    embed_fn,
    parser_version,
) -> dict:
    records = parse_fn(path)
    return add_fn(records, filename, doc_checksum, embed_fn, parser_version)
