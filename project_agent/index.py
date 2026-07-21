import hashlib
import os

import chromadb
from cryptography.fernet import Fernet


def file_checksum(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VectorIndex:
    def __init__(
        self,
        persist_directory,
        key_path,
        collection_name="viettel_docs",
        embedding_model_id="fasttext-cc.vi.300-v1",
    ):
        self.embedding_model_id = embedding_model_id
        key_path = str(key_path)
        key_directory = os.path.dirname(key_path)
        if key_directory:
            os.makedirs(key_directory, exist_ok=True)
        if os.path.exists(key_path):
            with open(key_path, "rb") as source:
                key = source.read()
        else:
            key = Fernet.generate_key()
            with open(key_path, "wb") as destination:
                destination.write(key)
        self.fernet = Fernet(key)

        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._guard_embedding_model()

    def add_records(
        self,
        records,
        filename,
        doc_checksum,
        embed_fn,
        parser_version,
    ) -> dict:
        filename = str(filename)
        doc_checksum = str(doc_checksum)
        data = [record for record in records if not record.is_section]
        existing = self.collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        existing_metadata = existing.get("metadatas") or []

        if any(
            metadata.get("doc_checksum") == doc_checksum
            for metadata in existing_metadata
        ):
            return {"status": "skipped_same_checksum", "added": 0}

        version = (
            max(int(metadata.get("doc_version", 0)) for metadata in existing_metadata)
            + 1
            if existing_metadata
            else 1
        )
        texts = [record.text_verbatim for record in data]
        embeddings = embed_fn(texts) if texts else []
        _validate_embeddings(texts, embeddings)

        ids = [
            hashlib.sha256(
                f"{doc_checksum}|{record.locator}".encode("utf-8")
            ).hexdigest()
            for record in data
        ]
        documents = [
            self.fernet.encrypt(record.text_verbatim.encode("utf-8")).decode("utf-8")
            for record in data
        ]
        metadatas = [
            {
                "filename": filename,
                "sheet_name": str(record.sheet_name or ""),
                "locator": str(record.locator),
                "row_index": int(record.row_index or 0),
                "section_path": " > ".join(
                    str(section) for section in record.section_path
                ),
                "stt": "" if record.stt is None else str(record.stt),
                "source_type": str(record.source_type),
                "doc_checksum": doc_checksum,
                "doc_version": version,
                "parser_version": str(parser_version),
                "embedding_model_id": str(self.embedding_model_id),
            }
            for record in data
        ]

        status = "added"
        if existing_metadata:
            self.collection.delete(where={"filename": filename})
            status = "reindexed"
        if data:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        return {
            "status": status,
            "added": len(data),
            "version": version,
        }

    def query(self, query_text, embed_fn, top_k=10, where=None) -> list[dict]:
        if self.count() == 0 or top_k <= 0:
            return []
        query_embeddings = embed_fn([query_text])
        _validate_embeddings([query_text], query_embeddings)
        query_arguments = {
            "query_embeddings": [query_embeddings[0]],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_arguments["where"] = where
        result = self.collection.query(**query_arguments)
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            {
                "text_verbatim": self.fernet.decrypt(document.encode("utf-8")).decode(
                    "utf-8"
                ),
                "metadata": metadata,
                "similarity": max(0.0, min(1.0, 1.0 - float(distance))),
            }
            for document, metadata, distance in zip(
                documents,
                metadatas,
                distances,
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    def list_filenames(self) -> list[str]:
        result = self.collection.get(include=["metadatas"])
        return sorted(
            {
                metadata["filename"]
                for metadata in result.get("metadatas") or []
                if metadata.get("filename")
            }
        )

    def delete_file(self, filename) -> None:
        self.collection.delete(where={"filename": str(filename)})

    def reset(self) -> None:
        collection_name = self.collection.name
        self.client.delete_collection(collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _guard_embedding_model(self) -> None:
        if self.collection.count() == 0:
            return
        sample = self.collection.get(limit=1, include=["metadatas"])
        metadata = (sample.get("metadatas") or [{}])[0]
        stored_model = metadata.get("embedding_model_id")
        if stored_model != self.embedding_model_id:
            raise ValueError(
                "Embedding model mismatch: "
                f"collection={stored_model!r}, requested={self.embedding_model_id!r}"
            )


def _validate_embeddings(texts, embeddings) -> None:
    if len(embeddings) != len(texts):
        raise ValueError("embed_fn must return one embedding per text")
    if not embeddings:
        return
    dimension = len(embeddings[0])
    if dimension == 0 or any(len(embedding) != dimension for embedding in embeddings):
        raise ValueError("embed_fn embeddings must have one non-zero shared dimension")
