"""Offline retrieval/gate baseline evaluation. No LLM is called."""

import argparse
import gc
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # noqa: E402
import embedding  # noqa: E402
import index as index_mod  # noqa: E402
import ingest  # noqa: E402
import rag  # noqa: E402
import structured  # noqa: E402
import textnorm  # noqa: E402


ROUTES = ("structured", "vector", "refuse")
LOCATOR_RE = re.compile(
    r"^(?P<sheet>.+)!(?P<c1>[A-Z]+)(?P<r1>\d+)(?::(?P<c2>[A-Z]+)(?P<r2>\d+))?$"
)

def frange(start, stop, step):
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def _column_number(column):
    number = 0
    for character in column:
        number = number * 26 + ord(character) - ord("A") + 1
    return number


def _locator_range(locator):
    match = LOCATOR_RE.fullmatch(locator)
    if not match:
        return None
    start_column = _column_number(match.group("c1"))
    start_row = int(match.group("r1"))
    end_column = _column_number(match.group("c2") or match.group("c1"))
    end_row = int(match.group("r2") or match.group("r1"))
    return (
        match.group("sheet").strip().casefold(),
        min(start_column, end_column),
        min(start_row, end_row),
        max(start_column, end_column),
        max(start_row, end_row),
    )


def locator_matches(actual, expected):
    actual_range = _locator_range(actual)
    expected_range = _locator_range(expected)
    if not actual_range or not expected_range or actual_range[0] != expected_range[0]:
        return actual.strip().casefold() == expected.strip().casefold()
    _, actual_c1, actual_r1, actual_c2, actual_r2 = actual_range
    _, expected_c1, expected_r1, expected_c2, expected_r2 = expected_range
    return not (
        actual_c2 < expected_c1
        or expected_c2 < actual_c1
        or actual_r2 < expected_r1
        or expected_r2 < actual_r1
    )


def validate_golden(golden):
    errors = []
    queries = golden.get("queries") if isinstance(golden, dict) else None
    if not isinstance(queries, list):
        raise ValueError("golden.queries must be a list")

    seen_ids = set()
    route_counts = Counter()
    for position, item in enumerate(queries, 1):
        prefix = f"queries[{position}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        query_id = item.get("id")
        if not isinstance(query_id, str) or not query_id.strip():
            errors.append(f"{prefix}.id must be a non-empty string")
        elif query_id in seen_ids:
            errors.append(f"duplicate id: {query_id}")
        else:
            seen_ids.add(query_id)

        route = item.get("route")
        if route not in ROUTES:
            errors.append(f"{prefix}.route must be one of {ROUTES}")
        else:
            route_counts[route] += 1
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            errors.append(f"{prefix}.query must be a non-empty string")

        locators = item.get("acceptable_locators")
        if not isinstance(locators, list) or any(
            not isinstance(locator, str) or not locator.strip() for locator in locators or []
        ):
            errors.append(f"{prefix}.acceptable_locators must be a string list")
            locators = []
        for locator in locators:
            if not _locator_range(locator):
                errors.append(f"{prefix} has invalid Excel locator: {locator!r}")
        if route in {"structured", "vector"} and not locators:
            errors.append(f"{prefix} requires at least one acceptable locator")
        if route == "refuse" and locators:
            errors.append(f"{prefix} refuse route must not have acceptable locators")

        if route == "structured":
            for field in (
                "expected_answer",
                "expected_filter",
                "expected_column",
                "expected_operation",
            ):
                if field not in item:
                    errors.append(f"{prefix}.{field} is required for structured route")
            if not isinstance(item.get("expected_filter"), dict) or not item.get(
                "expected_filter"
            ):
                errors.append(f"{prefix}.expected_filter must be a non-empty object")
            if not isinstance(item.get("expected_column"), str) or not item.get(
                "expected_column", ""
            ).strip():
                errors.append(f"{prefix}.expected_column must be a non-empty string")
            if item.get("expected_operation") not in {"count", "sum", "value"}:
                errors.append(
                    f"{prefix}.expected_operation must be count, sum, or value"
                )

    for route in ROUTES:
        if route_counts[route] < 12:
            errors.append(f"route {route!r} has {route_counts[route]} queries; need >= 12")
    if errors:
        raise ValueError("Invalid golden schema:\n- " + "\n- ".join(errors))
    return queries


def load_json(path):
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def select_embedder(choice, model_path):
    path = Path(model_path)
    selected = "onnx-e5" if choice == "auto" and path.is_dir() else choice
    if selected == "auto":
        selected = "fasttext"
    if selected == "fasttext":
        if not path.is_file():
            raise ValueError(f"FastText model must be a file: {path}")
        return (
            embedding.FastTextEmbedder(
                model_path=str(path),
                preprocess=textnorm.normalize,
            ),
            selected,
        )
    if selected == "onnx-e5":
        if not path.is_dir():
            raise ValueError(f"ONNX E5 model must be a directory: {path}")
        return (
            embedding.LocalOnnxEmbedder(
                path,
                max_length=96,
                batch_size=16,
            ),
            selected,
        )
    raise ValueError(f"Unknown embedder: {choice}")


def build_index(doc_path, tmp_dir, embedder_choice="auto", model_path=None):
    timings = {}
    embedder, selected = select_embedder(
        embedder_choice,
        model_path or config.MODEL_PATH,
    )
    started = time.perf_counter()
    embedder.load()
    timings["model_load_ms"] = (time.perf_counter() - started) * 1000
    embed_passages = getattr(embedder, "embed_passages", embedder.embed)
    embed_queries = getattr(embedder, "embed_queries", embedder.embed)
    vindex = index_mod.VectorIndex(
        persist_directory=os.path.join(tmp_dir, "chroma_db"),
        key_path=os.path.join(tmp_dir, "eval.key"),
        collection_name="eval_docs",
        embedding_model_id=(
            config.EMBEDDING_MODEL_ID
            if selected == "onnx-e5"
            else "fasttext-cc.vi.300-v1"
        ),
    )
    started = time.perf_counter()
    records = ingest.parse_file(doc_path)
    timings["parse_ms"] = (time.perf_counter() - started) * 1000
    checksum = index_mod.file_checksum(doc_path)
    started = time.perf_counter()
    result = vindex.add_records(
        records,
        filename=os.path.basename(doc_path),
        doc_checksum=checksum,
        embed_fn=embed_passages,
        parser_version="eval-s0",
    )
    timings["index_ms"] = (time.perf_counter() - started) * 1000
    timings["embedder"] = selected
    timings["model_path"] = str(model_path or config.MODEL_PATH)
    return vindex, embed_queries, result, timings


def _query_timed(vindex, embed_fn, query, top_k, threshold):
    embed_ms = 0.0

    def timed_embed(texts):
        nonlocal embed_ms
        started = time.perf_counter()
        result = embed_fn(texts)
        embed_ms += (time.perf_counter() - started) * 1000
        return result

    started = time.perf_counter()
    hits = vindex.query(query, timed_embed, top_k=top_k)
    chroma_ms = max(0.0, (time.perf_counter() - started) * 1000 - embed_ms)

    started = time.perf_counter()
    kept, _ = rag.select_evidence(hits, threshold)
    gate_ms = (time.perf_counter() - started) * 1000
    return hits, kept, {
        "embed_ms": embed_ms,
        "chroma_ms": chroma_ms,
        "gate_ms": gate_ms,
    }


def _is_relevant(hit, acceptable_locators):
    actual = hit["metadata"].get("locator", "")
    return any(locator_matches(actual, expected) for expected in acceptable_locators)


def _citation_is_relevant(citation, acceptable_locators):
    actual = citation.get("locator", "")
    return any(locator_matches(actual, expected) for expected in acceptable_locators)


def _same_answer(actual, expected):
    return str(actual).strip().casefold() == str(expected).strip().casefold()


def evaluate(vindex, embed_fn, queries, top_k, threshold, doc_path):
    rows = []
    for item in queries:
        expected_route = item["route"]
        router_result = None
        router_ms = None
        if expected_route in {"structured", "refuse"}:
            started = time.perf_counter()
            router_result = structured.execute(item["query"], doc_path)
            router_ms = (time.perf_counter() - started) * 1000

        use_structured = expected_route == "structured" or (
            expected_route == "refuse"
            and router_result is not None
            and not router_result.get("citations")
        )
        hits = []
        kept = []
        timings = {"embed_ms": None, "chroma_ms": None, "gate_ms": None}
        if use_structured:
            actual_answer = (
                router_result.get("answer") if router_result is not None else None
            )
            citations = (
                list(router_result.get("citations") or [])
                if router_result is not None
                else []
            )
            answered = bool(citations)
            actual_route = "structured" if answered else "refuse"
            routed_refusal = actual_route == "refuse"
        else:
            hits, kept, timings = _query_timed(
                vindex, embed_fn, item["query"], top_k, threshold
            )
            citations = rag.build_citations(kept)
            answered = bool(kept)
            actual_route = "vector" if answered else "refuse"
            actual_answer = None if answered else rag.NO_EVIDENCE_MESSAGE
            routed_refusal = False

        acceptable = item["acceptable_locators"]
        ranked_locators = (
            [hit["metadata"].get("locator", "") for hit in hits]
            if hits
            else [citation.get("locator", "") for citation in citations]
        )
        relevant_ranks = [
            rank
            for rank, locator in enumerate(ranked_locators, 1)
            if any(locator_matches(locator, expected) for expected in acceptable)
        ]
        relevant_hits = [hit for hit in hits if _is_relevant(hit, acceptable)]
        matched_expected = sum(
            any(
                locator_matches(citation.get("locator", ""), expected)
                for citation in citations
            )
            for expected in acceptable
        )
        rows.append(
            {
                "id": item["id"],
                "route": expected_route,
                "actual_route": actual_route,
                "route_correct": actual_route == expected_route,
                "query": item["query"],
                "top1_sim": hits[0]["similarity"] if hits else 0.0,
                "top1_locator": ranked_locators[0] if ranked_locators else "",
                "best_relevant_sim": max(
                    (hit["similarity"] for hit in relevant_hits), default=0.0
                ),
                "best_gated_relevant_sim": max(
                    (
                        hit["similarity"]
                        for hit in relevant_hits
                        if hit.get("lexical_gate", True)
                    ),
                    default=0.0,
                ),
                "best_gate_sim": max(
                    (
                        hit["similarity"]
                        for hit in hits
                        if hit.get("lexical_gate", True)
                    ),
                    default=0.0,
                ),
                "routed_refusal": routed_refusal,
                "first_relevant_rank": relevant_ranks[0] if relevant_ranks else None,
                "locator_recall_1": bool(relevant_ranks and relevant_ranks[0] <= 1),
                "locator_recall_5": bool(relevant_ranks and relevant_ranks[0] <= 5),
                "mrr": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
                "citation_precision": (
                    sum(
                        _citation_is_relevant(citation, acceptable)
                        for citation in citations
                    )
                    / len(citations)
                    if citations and acceptable
                    else 0.0 if acceptable else None
                ),
                "citation_recall": (
                    matched_expected / len(acceptable) if acceptable else None
                ),
                "answered": answered,
                "relevant_survives_gate": any(
                    _citation_is_relevant(citation, acceptable)
                    for citation in citations
                ),
                "expected_answer": item.get("expected_answer"),
                "actual_answer": actual_answer,
                "answer_correct": (
                    _same_answer(actual_answer, item["expected_answer"])
                    if "expected_answer" in item
                    else None
                ),
                "router_ms": router_ms,
                **timings,
            }
        )
    return rows


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _p95(values):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * 0.95) - 1)] if values else None


def _format_rate(value):
    return "   n/a" if value is None else f"{value:>6.3f}"


def _format_ms(value):
    return "-" if value is None else f"{value:.2f}"


def print_route_metrics(rows):
    print("\n=== METRICS BY EXPECTED ROUTE ===")
    print(
        f"{'route':<11} {'n':>3} {'Route':>7} {'LocR@1':>7} {'LocR@5':>7} {'MRR':>7} "
        f"{'CitP':>7} {'CitR':>7} {'gateAns':>8} {'relGate':>8} {'exact answer'}"
    )
    for route in ROUTES:
        subset = [row for row in rows if row["route"] == route]
        labelled = [row for row in subset if row["citation_recall"] is not None]
        evaluated_answers = [
            row for row in subset if row["answer_correct"] is not None
        ]
        answer_slots = sum(row["expected_answer"] is not None for row in subset)
        exact = (
            f"{sum(row['answer_correct'] for row in evaluated_answers)}/{len(evaluated_answers)}"
            if evaluated_answers
            else f"n/a (0/{answer_slots} supplied)" if answer_slots else "-"
        )
        print(
            f"{route:<11} {len(subset):>3} "
            f"{_format_rate(_mean(row['route_correct'] for row in subset))} "
            f"{_format_rate(_mean(row['locator_recall_1'] for row in labelled))} "
            f"{_format_rate(_mean(row['locator_recall_5'] for row in labelled))} "
            f"{_format_rate(_mean(row['mrr'] for row in labelled))} "
            f"{_format_rate(_mean(row['citation_precision'] for row in labelled))} "
            f"{_format_rate(_mean(row['citation_recall'] for row in labelled))} "
            f"{_format_rate(_mean(row['answered'] for row in subset))} "
            f"{_format_rate(_mean(row['relevant_survives_gate'] for row in labelled))} "
            f"{exact}"
        )


def refusal_metrics(rows):
    rag_rows = [row for row in rows if row["route"] in {"vector", "refuse"}]
    true_refuse = sum(row["route"] == "refuse" and not row["answered"] for row in rag_rows)
    false_refuse = sum(row["route"] == "vector" and not row["answered"] for row in rag_rows)
    false_answer = sum(row["route"] == "refuse" and row["answered"] for row in rag_rows)
    refuse_total = sum(row["route"] == "refuse" for row in rag_rows)
    return {
        "precision": true_refuse / (true_refuse + false_refuse)
        if true_refuse + false_refuse
        else 0.0,
        "recall": true_refuse / refuse_total if refuse_total else 0.0,
        "false_answer_rate": false_answer / refuse_total if refuse_total else 0.0,
        "true_refuse": true_refuse,
        "false_refuse": false_refuse,
        "false_answer": false_answer,
    }


def sweep(rows, thresholds):
    vector_rows = [row for row in rows if row["route"] == "vector"]
    refuse_rows = [row for row in rows if row["route"] == "refuse"]
    table = []
    for threshold in thresholds:
        vector_pass = sum(
            row["best_gated_relevant_sim"] >= threshold for row in vector_rows
        )
        true_refuse = sum(
            row["routed_refusal"] or row["best_gate_sim"] < threshold
            for row in refuse_rows
        )
        false_refuse = sum(
            row["best_gate_sim"] < threshold for row in vector_rows
        )
        vector_rate = vector_pass / len(vector_rows) if vector_rows else 0.0
        refusal_recall = true_refuse / len(refuse_rows) if refuse_rows else 0.0
        refusal_precision = (
            true_refuse / (true_refuse + false_refuse)
            if true_refuse + false_refuse
            else 0.0
        )
        table.append(
            {
                "threshold": threshold,
                "vector_rate": vector_rate,
                "refusal_precision": refusal_precision,
                "refusal_recall": refusal_recall,
                "false_answer_rate": 1 - refusal_recall,
                "balanced": (vector_rate + refusal_recall) / 2,
            }
        )
    return table


def print_timings(rows):
    print("\n=== QUERY TIMINGS MS (avg / p95) ===")
    print(
        f"{'route':<11} {'router':>17} {'embed':>17} "
        f"{'Chroma+rerank':>17} {'gate':>17} {'LLM':>7}"
    )
    for route in (*ROUTES, "all"):
        subset = rows if route == "all" else [row for row in rows if row["route"] == route]
        values = []
        for key in ("router_ms", "embed_ms", "chroma_ms", "gate_ms"):
            phase = [row[key] for row in subset if row[key] is not None]
            values.append(
                f"{_mean(phase):.3f} / {_p95(phase):.3f}" if phase else "n/a"
            )
        print(
            f"{route:<11} {values[0]:>17} {values[1]:>17} "
            f"{values[2]:>17} {values[3]:>17} {'n/a':>7}"
        )


def print_report(rows, build_timings, ingest_result, index_count, args):
    print(f"Ingested {args.doc}: {ingest_result}; index_count={index_count}")
    print(
        f"Embedder: {build_timings['embedder']} ({build_timings['model_path']})\n"
        "Build timings ms: "
        f"model_load={build_timings['model_load_ms']:.1f}, "
        f"parse={build_timings['parse_ms']:.1f}, index={build_timings['index_ms']:.1f}"
    )
    print(f"Telemetry: ANONYMIZED_TELEMETRY={os.environ['ANONYMIZED_TELEMETRY']}")
    print(f"Operating threshold: {args.threshold:.3f}; no LLM")

    print("\n=== PER QUERY ===")
    print(
        f"{'id':<25} {'expected':<10} {'actual':<10} {'top1':>6} {'rank':>4} "
        f"{'emb':>7} {'chr':>7} {'locator'}"
    )
    for row in rows:
        rank = row["first_relevant_rank"] or "-"
        print(
            f"{row['id']:<25} {row['route']:<10} {row['actual_route']:<10} "
            f"{row['top1_sim']:>6.3f} {str(rank):>4} "
            f"{_format_ms(row['embed_ms']):>7} {_format_ms(row['chroma_ms']):>7} "
            f"{row['top1_locator'][:52]}"
        )

    print_route_metrics(rows)
    refusal = refusal_metrics(rows)
    print("\n=== RAG GATE (vector + refuse only) ===")
    print(
        f"refusal precision={refusal['precision']:.3f} "
        f"recall={refusal['recall']:.3f} "
        f"false-answer-rate={refusal['false_answer_rate']:.3f} "
        f"(true_refuse={refusal['true_refuse']}, false_refuse={refusal['false_refuse']}, "
        f"false_answer={refusal['false_answer']})"
    )
    print_timings(rows)

    table = sweep(rows, frange(args.tmin, args.tmax, args.tstep))
    print("\n=== THRESHOLD SWEEP (vector relevance vs refuse gate) ===")
    print(f"{'T':>6} {'vec-pass':>9} {'ref-P':>8} {'ref-R':>8} {'false-A':>8} {'balanced':>9}")
    for row in table:
        print(
            f"{row['threshold']:>6.2f} {row['vector_rate']:>9.3f} "
            f"{row['refusal_precision']:>8.3f} {row['refusal_recall']:>8.3f} "
            f"{row['false_answer_rate']:>8.3f} {row['balanced']:>9.3f}"
        )
    feasible = [row for row in table if row["vector_rate"] >= args.recall_target]
    pool = feasible or table
    best = max(
        pool,
        key=lambda row: (
            row["balanced"],
            row["refusal_precision"],
            -row["threshold"],
        ),
    )
    print("\n=== RECOMMENDATION ===")
    if not feasible:
        print(
            f"[WARN] No threshold keeps vector relevant-pass >= {args.recall_target:.2f}."
        )
    print(
        f"SIMILARITY_THRESHOLD={best['threshold']:.2f} "
        f"(vector-pass={best['vector_rate']:.3f}, refusal-P={best['refusal_precision']:.3f}, "
        f"refusal-R={best['refusal_recall']:.3f}, balanced={best['balanced']:.3f})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="Phu luc 1.xlsx")
    parser.add_argument(
        "--golden",
        default=os.path.join(os.path.dirname(__file__), "golden_queries.json"),
    )
    parser.add_argument(
        "--embedder",
        choices=("auto", "fasttext", "onnx-e5"),
        default="auto",
    )
    parser.add_argument("--model-path", default=config.MODEL_PATH)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=config.SIMILARITY_THRESHOLD)
    parser.add_argument("--tmin", type=float, default=0.40)
    parser.add_argument("--tmax", type=float, default=0.90)
    parser.add_argument("--tstep", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--recall-target", type=float, default=0.90)
    args = parser.parse_args()

    golden = load_json(args.golden)
    queries = validate_golden(golden)
    print(
        "Golden schema valid: "
        + ", ".join(
            f"{route}={sum(item['route'] == route for item in queries)}"
            for route in ROUTES
        )
    )
    if args.validate_only:
        return
    if args.top_k <= 0 or not 0 <= args.threshold <= 1:
        raise ValueError("top-k must be positive and threshold must be between 0 and 1")
    if not os.path.exists(args.doc):
        print(f"[BLOCKER] Document not found: {args.doc}")
        return

    try:
        with tempfile.TemporaryDirectory(
            prefix="eval_retrieval_", ignore_cleanup_errors=True
        ) as tmp_dir:
            vindex, embed_fn, ingest_result, build_timings = build_index(
                args.doc,
                tmp_dir,
                embedder_choice=args.embedder,
                model_path=args.model_path,
            )
            rows = evaluate(
                vindex, embed_fn, queries, args.top_k, args.threshold, args.doc
            )
            print_report(rows, build_timings, ingest_result, vindex.count(), args)
            del rows, embed_fn, vindex
            gc.collect()
    except ValueError as error:
        print(f"[BLOCKER] {error}")


if __name__ == "__main__":
    main()
