import hashlib
import json

import numpy as np
import pytest

import embedding


class FakeModel:
    def __init__(self):
        self.inputs = []

    def get_sentence_vector(self, text):
        self.inputs.append(text)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return np.array([digest[0] + 1, digest[1] + 1, digest[2] + 1], dtype=float)


class FakeEncoding:
    def __init__(self, length):
        self.ids = list(range(1, length + 1))
        self.attention_mask = [1] * length
        self.type_ids = [0] * length


class FakeTokenizer:
    def __init__(self):
        self.batches = []

    def enable_truncation(self, max_length):
        self.max_length = max_length

    def token_to_id(self, token):
        return 9 if token == "<pad>" else None

    def encode_batch(self, texts):
        self.batches.append(list(texts))
        return [FakeEncoding(len(text.split(": ", 1)[1])) for text in texts]


class FakeSession:
    def __init__(self):
        self.shapes = []
        self.input_ids = []

    def run(self, _, inputs):
        input_ids = inputs["input_ids"]
        self.shapes.append(input_ids.shape)
        self.input_ids.append(input_ids.copy())
        return [np.stack((input_ids, np.ones_like(input_ids)), axis=2)]


def make_bundle(tmp_path):
    files = {
        "tokenizer.json": b"tokenizer",
        "onnx/model_O4.onnx": b"model",
    }
    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    path: {"sha256": hashlib.sha256(content).hexdigest()}
                    for path, content in files.items()
                }
            }
        ),
        encoding="utf-8",
    )


def test_embed_alignment():
    vectors = embedding.FastTextEmbedder(model=FakeModel()).embed(["a", "b", "c"])

    assert len(vectors) == 3
    assert {len(vector) for vector in vectors} == {3}
    assert all(np.linalg.norm(vector) == pytest.approx(1.0) for vector in vectors)


def test_no_drop_on_empty():
    model = FakeModel()

    vectors = embedding.FastTextEmbedder(model=model).embed(["x", ""])

    assert len(vectors) == 2
    assert model.inputs == ["x", ""]


def test_preprocess_applied():
    model = FakeModel()

    embedding.FastTextEmbedder(model=model, preprocess=str.upper).embed(["hello"])

    assert model.inputs == ["HELLO"]


def test_make_embed_fn():
    embed_fn = embedding.make_embed_fn(embedding.FastTextEmbedder(model=FakeModel()))

    vectors = embed_fn(["x"])

    assert len(vectors) == 1
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)


def test_local_onnx_prefixes_pooling_normalization_and_batches(tmp_path):
    make_bundle(tmp_path)
    tokenizer = FakeTokenizer()
    session = FakeSession()
    embedder = embedding.LocalOnnxEmbedder(
        tmp_path, batch_size=2, tokenizer=tokenizer, session=session
    )

    passages = embedder.embed(["a", "bb", "ccc"])
    queries = embedder.embed_queries(["q"])

    assert tokenizer.batches == [
        ["passage: a", "passage: bb", "passage: ccc"],
        ["query: q"],
    ]
    assert session.shapes == [(2, 2), (1, 3), (1, 1)]
    assert session.input_ids[0][0, 1] == 9
    assert passages[0] == pytest.approx([2**-0.5, 2**-0.5])
    assert all(
        np.linalg.norm(vector) == pytest.approx(1.0)
        for vector in passages + queries
    )


def test_local_onnx_refuses_manifest_mismatch_before_load(tmp_path):
    make_bundle(tmp_path)
    tokenizer = FakeTokenizer()
    session = FakeSession()
    (tmp_path / "onnx" / "model_O4.onnx").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        embedding.LocalOnnxEmbedder(
            tmp_path, tokenizer=tokenizer, session=session
        ).embed_queries(["x"])

    assert tokenizer.batches == []
    assert session.shapes == []


def test_local_onnx_restores_original_order_after_length_sort(tmp_path):
    make_bundle(tmp_path)
    embedder = embedding.LocalOnnxEmbedder(
        tmp_path,
        batch_size=2,
        tokenizer=FakeTokenizer(),
        session=FakeSession(),
    )

    vectors = embedder.embed_passages(["ccc", "a", "bb"])

    expected = []
    for length in (3, 1, 2):
        vector = np.array([(length + 1) / 2, 1.0])
        expected.append((vector / np.linalg.norm(vector)).tolist())
    np.testing.assert_allclose(vectors, expected)
