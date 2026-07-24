import hashlib
import json
import os
from pathlib import Path

import numpy as np


class FastTextEmbedder:
    def __init__(self, model=None, model_path=None, preprocess=None):
        self._model = model
        self.model_path = model_path
        self.preprocess = preprocess or (lambda text: text)

    def load(self):
        if self._model is None:
            import fasttext

            self._model = fasttext.load_model(self.model_path)
        return self

    def embed(self, texts) -> list[list[float]]:
        if self._model is None:
            self.load()
        embeddings = []
        for text in texts:
            processed = self.preprocess(text)
            vector = np.asarray(
                self._model.get_sentence_vector(processed),
                dtype=float,
            )
            norm = np.linalg.norm(vector)
            if norm != 0:
                vector = vector / norm
            embeddings.append(vector.tolist())
        return embeddings


class LocalOnnxEmbedder:
    def __init__(
        self,
        bundle_path,
        batch_size=32,
        max_length=512,
        tokenizer=None,
        session=None,
    ):
        self.bundle_path = Path(bundle_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self._tokenizer = tokenizer
        self._session = session
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        if self.batch_size < 1 or self.max_length < 1:
            raise ValueError("batch_size and max_length must be positive")

        manifest = json.loads(
            (self.bundle_path / "manifest.json").read_text(encoding="utf-8")
        )
        for relative_path in ("tokenizer.json", "onnx/model_O4.onnx"):
            path = self.bundle_path / relative_path
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            expected = manifest.get("files", {}).get(relative_path, {}).get("sha256")
            if not expected or digest.hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch for {relative_path}")

        if self._tokenizer is None:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(
                str(self.bundle_path / "tokenizer.json")
            )
        self._tokenizer.enable_truncation(max_length=self.max_length)
        if self._session is None:
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            options.intra_op_num_threads = min(16, max(1, (os.cpu_count() or 2) // 2))
            self._session = onnxruntime.InferenceSession(
                str(self.bundle_path / "onnx" / "model_O4.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        self._loaded = True
        return self

    def _embed(self, texts, prefix):
        texts = list(texts)
        if not texts:
            return []
        self.load()

        encodings = self._tokenizer.encode_batch(
            [f"{prefix}{text}" for text in texts]
        )
        sorted_encodings = sorted(
            enumerate(encodings), key=lambda item: len(item[1].ids)
        )
        pad_id = self._tokenizer.token_to_id("<pad>")
        pad_id = 1 if pad_id is None else pad_id
        embeddings = [None] * len(texts)
        for start in range(0, len(sorted_encodings), self.batch_size):
            batch = sorted_encodings[start : start + self.batch_size]
            width = max(len(encoding.ids) for _, encoding in batch)
            input_ids = np.full((len(batch), width), pad_id, dtype=np.int64)
            attention_mask = np.zeros_like(input_ids)
            token_type_ids = np.zeros_like(input_ids)
            for row, (_, encoding) in enumerate(batch):
                length = len(encoding.ids)
                input_ids[row, :length] = encoding.ids
                attention_mask[row, :length] = encoding.attention_mask
                token_type_ids[row, :length] = encoding.type_ids

            hidden = np.asarray(
                self._session.run(
                    None,
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                    },
                )[0],
                dtype=np.float32,
            )
            mask = attention_mask[..., None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.maximum(
                mask.sum(axis=1), 1.0
            )
            pooled /= np.maximum(
                np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12
            )
            for (original_index, _), vector in zip(batch, pooled):
                embeddings[original_index] = vector.tolist()
        return embeddings

    def embed_passages(self, texts) -> list[list[float]]:
        return self._embed(texts, "passage: ")

    def embed_queries(self, texts) -> list[list[float]]:
        return self._embed(texts, "query: ")

    def embed(self, texts) -> list[list[float]]:
        return self.embed_passages(texts)


def make_embed_fn(embedder):
    return lambda texts: embedder.embed(texts)
