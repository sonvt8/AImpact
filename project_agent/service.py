import os

import config
import embedding
import index
import ingest
import pipeline
import textnorm


PARSER_VERSION = "ingest-v1"


class RagService:
    def __init__(self, embed_fn, index_obj, generator_fn, role="Trợ lý kỹ thuật"):
        self.embed_fn = embed_fn
        self.index_obj = index_obj
        self.generator_fn = generator_fn
        self.role = role

    def ingest_path(self, path, filename) -> dict:
        checksum = index.file_checksum(path)
        return pipeline.ingest_file(
            parse_fn=ingest.parse_file,
            add_fn=self.index_obj.add_records,
            path=path,
            filename=filename,
            doc_checksum=checksum,
            embed_fn=self.embed_fn,
            parser_version=PARSER_VERSION,
        )

    def answer(
        self,
        query,
        history="",
        *,
        threshold=None,
        top_k=10,
        where=None,
        is_summary=False,
        answer_format=None,
    ) -> dict:
        threshold = (
            config.SIMILARITY_THRESHOLD if threshold is None else threshold
        )
        retrieve_fn = lambda q, top_k, where: self.index_obj.query(
            q,
            self.embed_fn,
            top_k=top_k,
            where=where,
        )
        return pipeline.run_query(
            query,
            retrieve_fn=retrieve_fn,
            generator_fn=self.generator_fn,
            history=history,
            threshold=threshold,
            role=self.role,
            top_k=top_k,
            where=where,
            is_summary=is_summary,
            answer_format=answer_format,
        )


def build_service(generator_fn, *, embedder=None, index_obj=None) -> RagService:
    embedder = embedder or embedding.FastTextEmbedder(
        model_path=config.MODEL_PATH,
        preprocess=textnorm.normalize,
    )
    index_obj = index_obj or index.VectorIndex(
        persist_directory=os.path.join(config.DATA_DIR, "chroma_db"),
        key_path=os.path.join(config.DATA_DIR, "encryption_key.key"),
        embedding_model_id=config.EMBEDDING_MODEL_ID,
    )
    return RagService(
        embedding.make_embed_fn(embedder),
        index_obj,
        generator_fn,
    )
