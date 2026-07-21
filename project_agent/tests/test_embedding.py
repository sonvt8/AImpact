import hashlib

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
