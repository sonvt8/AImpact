import hashlib
import os
import re

import chromadb
from chromadb.config import Settings
from chromadb.telemetry.product import ProductTelemetryClient
from cryptography.fernet import Fernet
from overrides import override


RETRIEVAL_PROJECTION_VERSION = "fields-v1"
DEDUPLICATION_VERSION = "structured-exact-v1"
QUERY_ALIASES = (
    ("nguồn lưu điện", "UPS"),
    ("pin", "ắc quy"),
    ("đầu cấp bị ngắt", "không có nguồn vào"),
)
EXACT_IDENTIFIERS = frozenset(
    {
        "AC",
        "DC",
        "UPS",
        "N4",
        "N6",
        "ACB8",
        "ACB9",
        "MCCB",
        "HĐB1",
        "HĐB2",
        "UDB",
        "PDU",
        "VHKT",
        "UCTT",
        "XLSC",
        "FM200",
    }
)
IDENTIFIER_PATTERN = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        re.escape(value)
        for value in sorted(EXACT_IDENTIFIERS, key=len, reverse=True)
    )
    + r"|ACB\d+|HĐB\d+|N\d+|FM\d+)(?!\w)",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
LEXICAL_STOP_WORDS = frozenset(
    "bị có của đang do gì hãy là như nào ra tại thì trong từ và về với xử lý thế".split()
)


class NoopProductTelemetry(ProductTelemetryClient):
    @override
    def capture(self, *args, **kwargs):
        return None


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
        excluded_sheet_prefixes=("Form ",),
    ):
        self.embedding_model_id = embedding_model_id
        self.excluded_sheet_prefixes = tuple(
            str(prefix).casefold() for prefix in excluded_sheet_prefixes
        )
        self.excluded_sheet_prefixes_metadata = "\n".join(
            self.excluded_sheet_prefixes
        )
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
        settings = Settings(
            anonymized_telemetry=False,
            chroma_product_telemetry_impl=f"{__name__}.NoopProductTelemetry",
        )
        try:
            self.client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=settings,
            )
        except TypeError as error:
            if "settings" not in str(error):
                raise
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
        data = []
        seen = set()
        for record in records:
            if record.is_section or str(record.sheet_name).casefold().startswith(
                self.excluded_sheet_prefixes
            ):
                continue
            dedupe_key = None
            if record.source_type in {"xlsx", "csv"}:
                dedupe_key = (
                    record.source_type,
                    str(record.sheet_name),
                    record.text_verbatim,
                )
            if dedupe_key is not None and dedupe_key in seen:
                continue
            if dedupe_key is not None:
                seen.add(dedupe_key)
            data.append(record)
        existing = self.collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        existing_metadata = existing.get("metadatas") or []

        if existing_metadata and all(
            metadata.get("doc_checksum") == doc_checksum
            and metadata.get("parser_version") == str(parser_version)
            and metadata.get("retrieval_projection_version")
            == RETRIEVAL_PROJECTION_VERSION
            and metadata.get("deduplication_version") == DEDUPLICATION_VERSION
            and metadata.get("excluded_sheet_prefixes")
            == self.excluded_sheet_prefixes_metadata
            for metadata in existing_metadata
        ):
            return {"status": "skipped_same_checksum", "added": 0}

        version = (
            max(int(metadata.get("doc_version", 0)) for metadata in existing_metadata)
            + 1
            if existing_metadata
            else 1
        )
        texts = [record.text_retrieval for record in data]
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
                "retrieval_projection_version": RETRIEVAL_PROJECTION_VERSION,
                "deduplication_version": DEDUPLICATION_VERSION,
                "excluded_sheet_prefixes": self.excluded_sheet_prefixes_metadata,
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
        collection_count = self.count()
        if collection_count == 0 or top_k <= 0:
            return []
        expanded_query = _expand_query(query_text)
        query_embeddings = embed_fn([expanded_query])
        _validate_embeddings([expanded_query], query_embeddings)
        query_arguments = {
            "query_embeddings": [query_embeddings[0]],
            "n_results": collection_count,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_arguments["where"] = where
        result = self.collection.query(**query_arguments)
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        query_identifiers = _extract_identifiers(expanded_query)
        query_tokens = _lexical_tokens(expanded_query)
        ranked = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            text_verbatim = self.fernet.decrypt(document.encode("utf-8")).decode(
                "utf-8"
            )
            lexical_text = "\n".join(
                (
                    text_verbatim,
                    str(metadata.get("sheet_name", "")),
                    str(metadata.get("section_path", "")),
                    str(metadata.get("stt", "")),
                )
            )
            hit_identifiers = _extract_identifiers(lexical_text)
            identifier_matches = query_identifiers & hit_identifiers
            token_overlap = query_tokens & _lexical_tokens(lexical_text)
            lexical_gate = query_identifiers <= hit_identifiers and (
                bool(query_identifiers) or len(token_overlap) >= 2
            )
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            hit = {
                "text_verbatim": text_verbatim,
                "metadata": metadata,
                "similarity": similarity,
                "lexical_gate": lexical_gate,
            }
            ranked.append(
                (
                    (
                        lexical_gate,
                        len(identifier_matches),
                        len(token_overlap),
                        similarity,
                    ),
                    hit,
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in ranked[:top_k]]

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


def _expand_query(query_text) -> str:
    query_text = str(query_text)
    folded = query_text.casefold()
    aliases = [
        alias
        for phrase, alias in QUERY_ALIASES
        if phrase in folded and alias.casefold() not in folded
    ]
    return " ".join((query_text, *aliases))


def _extract_identifiers(text) -> set[str]:
    return {match.group(0).upper() for match in IDENTIFIER_PATTERN.finditer(str(text))}


def _lexical_tokens(text) -> set[str]:
    return {
        token
        for token in (value.casefold() for value in TOKEN_PATTERN.findall(str(text)))
        if len(token) > 1 and token not in LEXICAL_STOP_WORDS
    }


def _validate_embeddings(texts, embeddings) -> None:
    if len(embeddings) != len(texts):
        raise ValueError("embed_fn must return one embedding per text")
    if not embeddings:
        return
    dimension = len(embeddings[0])
    if dimension == 0 or any(len(embedding) != dimension for embedding in embeddings):
        raise ValueError("embed_fn embeddings must have one non-zero shared dimension")
