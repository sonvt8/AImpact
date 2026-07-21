import gc
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
import embedding
import index
import rag
import service
import textnorm
from langchain_openai import ChatOpenAI


QUERIES = {
    "q_vhkt": "quy trình vận hành kỹ thuật",
    "q_uctt": "sự cố mất điện lưới tòa nhà N6",
    "q_out": "công thức nấu phở bò",
}


def main():
    try:
        config.validate()
    except Exception as error:
        print(f"CONFIG ERROR: {error}")
        return 1
    print("CONFIG OK")

    try:
        embedder = embedding.FastTextEmbedder(
            model_path=config.MODEL_PATH,
            preprocess=textnorm.normalize,
        ).load()
    except Exception as error:
        print(f"FASTTEXT ERROR: {error}")
        return 1
    print("FASTTEXT LOADED")

    tempdir = Path(tempfile.mkdtemp(prefix="project-agent-smoke-"))
    vindex = None
    svc = None
    try:
        vindex = index.VectorIndex(
            persist_directory=tempdir / "chroma",
            key_path=tempdir / "k.key",
            embedding_model_id=config.EMBEDDING_MODEL_ID,
        )
        llm = ChatOpenAI(
            **config.build_llm_client_kwargs("openai_compatible")
        )

        def generator_fn(prompt):
            try:
                return llm.invoke(prompt).content
            except Exception as error:
                return f"[LLM ERROR] {error}"

        svc = service.build_service(
            generator_fn,
            embedder=embedder,
            index_obj=vindex,
        )
        ingest_result = svc.ingest_path(
            PROJECT_ROOT / "Phu luc 1.xlsx",
            "Phu luc 1.xlsx",
        )
        print(f"INGEST RESULT: {ingest_result}")
        assert ingest_result["added"] == 357

        hits_by_query = {}
        answers = {}
        for name, query in QUERIES.items():
            print(f"\n{name}: {query}")
            hits = vindex.query(query, svc.embed_fn, top_k=5)
            hits_by_query[name] = hits
            print("TOP-5:")
            for position, hit in enumerate(hits, start=1):
                metadata = hit["metadata"]
                print(
                    f"  {position}. "
                    f"({metadata.get('sheet_name', '')!r}, "
                    f"{metadata.get('locator', '')!r}, "
                    f"{round(hit['similarity'], 4)})"
                )

            result = svc.answer(
                query,
                threshold=config.SIMILARITY_THRESHOLD,
                top_k=5,
            )
            answers[name] = result
            print(f"llm_called={result['llm_called']}")
            print(f"answer[:200]={result['answer'][:200]}")
            print(
                "citations="
                f"{[citation['locator'] for citation in result['citations']]}"
            )

        ingest_357 = ingest_result["added"] == 357
        vhkt_reachable = any(
            hit["metadata"].get("sheet_name") == "3. Ds VHKT"
            for hits in hits_by_query.values()
            for hit in hits
        )
        out_of_scope_llm_called = answers["q_out"]["llm_called"]
        llm_reachable = any(
            answers[name]["llm_called"]
            and bool(answers[name]["answer"])
            and not answers[name]["answer"].startswith("[LLM ERROR]")
            for name in ("q_vhkt", "q_uctt")
        )

        print("\nSMOKE SUMMARY")
        print(f"ingest_357={ingest_357}")
        print(f"vhkt_reachable={vhkt_reachable}")
        print(f"out_of_scope_llm_called={out_of_scope_llm_called}")
        print(f"llm_reachable={llm_reachable}")
        return 0
    finally:
        if vindex is not None:
            vindex.client.clear_system_cache()
        svc = None
        vindex = None
        gc.collect()
        shutil.rmtree(tempdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
