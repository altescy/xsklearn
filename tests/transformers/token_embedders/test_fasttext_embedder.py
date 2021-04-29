from xsklearn.transformers.token_embedders.fasttext_embedder import FastTextEmbedder


class TestFastTextEmbedder:
    def setup(self) -> None:
        self.inputs = [
            ["this", "is", "a", "first", "sentence", "."],
            ["this", "is", "a", "second", "sentence", "."],
        ]

    def test_fit(self) -> None:
        embedder = FastTextEmbedder()
        embeddings = embedder.fit_transform(self.inputs)

        assert len(embeddings) == 2
