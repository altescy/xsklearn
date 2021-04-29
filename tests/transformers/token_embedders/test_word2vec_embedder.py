import gensim

from xsklearn.transformers.token_embedders.word2vec_embedder import Word2VecEmbedder


class TestWord2VecEmbedder:
    def setup(self) -> None:
        self.inputs = [
            ["this", "is", "a", "first", "sentence", "."],
            ["this", "is", "a", "second", "sentence", "."],
        ]
        self.model = gensim.models.Word2Vec(
            sentences=self.inputs,
            vector_size=50,
            window=5,
            min_count=1,
        )

    def test_get_token_embedding(self) -> None:
        embedder = Word2VecEmbedder(model=self.model)
        embedding = embedder.get("this")

        assert embedding is not None
        assert embedding.shape == (50,)

    def test_transform(self) -> None:
        embedder = Word2VecEmbedder(model=self.model)
        embeddings = embedder.transform(self.inputs)

        assert len(embeddings) == 2

    def test_fit(self) -> None:
        embedder = Word2VecEmbedder()
        embeddings = embedder.fit_transform(self.inputs)

        assert len(embeddings) == 2
