import gensim

from xsklearn.transformers.text_vectorizers.bag_of_embeddings import BagOfEmbeddings
from xsklearn.transformers.token_embedders.word2vec_embedder import Word2VecEmbedder


class TestBagOfEmbeddings:
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
        self.token_embedder = Word2VecEmbedder(model=self.model)

    def test_transform(self) -> None:
        boe = BagOfEmbeddings(token_embedder=self.token_embedder)
        outputs = boe.transform(self.inputs)
        assert outputs.shape == (2, 50)
