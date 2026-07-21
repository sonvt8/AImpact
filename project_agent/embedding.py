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


def make_embed_fn(embedder):
    return lambda texts: embedder.embed(texts)
