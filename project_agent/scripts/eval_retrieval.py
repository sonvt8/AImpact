"""C-EVAL-1: retrieval-only golden evaluation and threshold calibration.

Runs REAL FastText + Chroma retrieval (NO LLM) against a golden query set and
sweeps the similarity threshold to find the value that best separates in-scope
from out-of-scope queries while keeping in-scope Recall@5 high.

Design notes (generic-first, reuses the shipped core; no behaviour change):
- Reuses embedding.FastTextEmbedder + index.VectorIndex exactly as production does.
- Ingests the document into an ISOLATED temp Chroma dir so production data/ is
  never touched. Uses config.MODEL_PATH for the real model.
- Relevance is judged at SHEET level (expected_sheets), which is robust and does
  not hard-code STT/locator into the eval.
- "in-scope pass @T"  := best relevant hit similarity >= T (a relevant doc
  survives the gate, so the system would answer).
- "out-of-scope reject @T" := top-1 similarity of ANY hit < T (nothing survives,
  so the system refuses with NO_EVIDENCE).
- Recall@5 (in-scope) := fraction of in-scope queries with a relevant hit in top-5.

Usage (from project_agent/ root, venv active, model present at config.MODEL_PATH):
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --doc "Phu luc 1.xlsx" --golden scripts/golden_queries.json
    python scripts/eval_retrieval.py --tmin 0.40 --tmax 0.85 --tstep 0.02 --top-k 10
Exit code 0 always (report tool). Prints a table and a recommended threshold.
"""

import argparse
import json
import os
import sys
import tempfile

# make project_agent importable when run as scripts/eval_retrieval.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # noqa: E402
import embedding  # noqa: E402
import index as index_mod  # noqa: E402
import ingest  # noqa: E402
import textnorm  # noqa: E402


def frange(start, stop, step):
    values = []
    current = start
    # guard against float drift; round to 4 decimals
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def build_index(doc_path, tmp_dir):
    embedder = embedding.FastTextEmbedder(
        model_path=config.MODEL_PATH,
        preprocess=textnorm.normalize,
    )
    embed_fn = embedding.make_embed_fn(embedder)
    vindex = index_mod.VectorIndex(
        persist_directory=os.path.join(tmp_dir, "chroma_db"),
        key_path=os.path.join(tmp_dir, "eval.key"),
        collection_name="eval_docs",
        embedding_model_id=config.EMBEDDING_MODEL_ID,
    )
    records = ingest.parse_file(doc_path)
    checksum = index_mod.file_checksum(doc_path)
    result = vindex.add_records(
        records,
        filename=os.path.basename(doc_path),
        doc_checksum=checksum,
        embed_fn=embed_fn,
        parser_version="eval",
    )
    return vindex, embed_fn, result


def evaluate(vindex, embed_fn, queries, top_k):
    """Return per-query dict: best_relevant_sim, top1_sim, relevant_in_top5."""
    rows = []
    for item in queries:
        hits = vindex.query(item["query"], embed_fn, top_k=top_k)
        top1 = hits[0]["similarity"] if hits else 0.0
        expected = set(item.get("expected_sheets") or [])
        relevant_sims = [
            h["similarity"]
            for h in hits
            if h["metadata"].get("sheet_name") in expected
        ]
        top5_sheets = [h["metadata"].get("sheet_name") for h in hits[:5]]
        rows.append(
            {
                "id": item["id"],
                "in_scope": bool(item["in_scope"]),
                "top1_sim": top1,
                "best_relevant_sim": max(relevant_sims) if relevant_sims else 0.0,
                "relevant_in_top5": any(s in expected for s in top5_sheets),
                "query": item["query"],
            }
        )
    return rows


def sweep(rows, thresholds):
    in_scope = [r for r in rows if r["in_scope"]]
    out_scope = [r for r in rows if not r["in_scope"]]
    table = []
    for t in thresholds:
        in_pass = sum(1 for r in in_scope if r["best_relevant_sim"] >= t)
        out_reject = sum(1 for r in out_scope if r["top1_sim"] < t)
        in_rate = in_pass / len(in_scope) if in_scope else 0.0
        out_rate = out_reject / len(out_scope) if out_scope else 0.0
        table.append(
            {
                "threshold": t,
                "in_pass": in_pass,
                "in_total": len(in_scope),
                "in_rate": in_rate,
                "out_reject": out_reject,
                "out_total": len(out_scope),
                "out_rate": out_rate,
                "balanced": (in_rate + out_rate) / 2,
            }
        )
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="Phu luc 1.xlsx")
    parser.add_argument("--golden", default=os.path.join(os.path.dirname(__file__), "golden_queries.json"))
    parser.add_argument("--tmin", type=float, default=0.40)
    parser.add_argument("--tmax", type=float, default=0.85)
    parser.add_argument("--tstep", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--recall-target", type=float, default=0.90)
    args = parser.parse_args()

    if not os.path.exists(config.MODEL_PATH):
        print(f"[BLOCKER] FastText model not found at {config.MODEL_PATH}. "
              f"Set MODEL_PATH or run load_model.py first.")
        sys.exit(0)
    if not os.path.exists(args.doc):
        print(f"[BLOCKER] Document not found: {args.doc}")
        sys.exit(0)

    with open(args.golden, "r", encoding="utf-8") as fh:
        golden = json.load(fh)
    queries = golden["queries"]

    with tempfile.TemporaryDirectory(prefix="eval_retrieval_") as tmp_dir:
        vindex, embed_fn, ingest_result = build_index(args.doc, tmp_dir)
        print(f"Ingested {args.doc}: {ingest_result}")
        print(f"Index count: {vindex.count()}")
        rows = evaluate(vindex, embed_fn, queries, args.top_k)

    # per-query detail
    print("\n=== PER-QUERY (retrieval only, no LLM) ===")
    print(f"{'id':<12} {'scope':<6} {'top1':>6} {'bestRel':>8} {'rel@5':>6}  query")
    for r in rows:
        scope = "IN" if r["in_scope"] else "OUT"
        rel5 = "yes" if r["relevant_in_top5"] else "-"
        print(f"{r['id']:<12} {scope:<6} {r['top1_sim']:>6.3f} "
              f"{r['best_relevant_sim']:>8.3f} {rel5:>6}  {r['query'][:48]}")

    in_scope = [r for r in rows if r["in_scope"]]
    recall5 = sum(1 for r in in_scope if r["relevant_in_top5"]) / len(in_scope) if in_scope else 0.0
    print(f"\nRecall@5 (in-scope): {recall5:.3f}  (target >= {args.recall_target})")

    # threshold sweep
    thresholds = frange(args.tmin, args.tmax, args.tstep)
    table = sweep(rows, thresholds)
    print("\n=== THRESHOLD SWEEP ===")
    print(f"{'T':>6} {'in_pass':>8} {'in_rate':>8} {'out_rej':>8} {'out_rate':>9} {'balanced':>9}")
    for row in table:
        print(f"{row['threshold']:>6.2f} {row['in_pass']:>3}/{row['in_total']:<4} "
              f"{row['in_rate']:>8.2f} {row['out_reject']:>4}/{row['out_total']:<3} "
              f"{row['out_rate']:>9.2f} {row['balanced']:>9.3f}")

    # recommendation: among thresholds meeting recall target on the gate for in-scope,
    # pick the one with the highest balanced separation; tie-break to lower T.
    feasible = [r for r in table if r["in_rate"] >= args.recall_target]
    pool = feasible if feasible else table
    best = max(pool, key=lambda r: (r["balanced"], -r["threshold"]))
    print("\n=== RECOMMENDATION ===")
    if not feasible:
        print(f"[WARN] No threshold keeps in-scope pass-rate >= {args.recall_target}. "
              f"FastText alone may not separate classes; prompt-refusal remains the safety net.")
    print(f"Recommended SIMILARITY_THRESHOLD = {best['threshold']:.2f} "
          f"(in {best['in_rate']:.2f}, out-reject {best['out_rate']:.2f}, "
          f"balanced {best['balanced']:.3f})")
    print("Set it via env: SIMILARITY_THRESHOLD=<value>  (no code change).")


if __name__ == "__main__":
    main()
