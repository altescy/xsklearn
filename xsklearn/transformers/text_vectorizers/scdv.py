from __future__ import annotations

from typing import Any, Callable, List, Optional, cast

import numpy
from sklearn.mixture import GaussianMixture

from xsklearn.transformers.text_vectorizers import TextVectorizer, TfidfVectorizer
from xsklearn.transformers.token_embedders import FastTextEmbedder, TokenEmbedder


class SCDV(TextVectorizer):
    def __init__(
        self,
        token_embedder: Optional[TokenEmbedder] = None,
        gaussian_mixture: Optional[GaussianMixture] = None,
        tfidf_vectorizer: Optional[TfidfVectorizer] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        sparcity: float = 0.04,
    ) -> None:
        super().__init__(tokenizer)
        self.token_embedder = token_embedder or FastTextEmbedder()
        self.gaussian_mixture = gaussian_mixture or GaussianMixture()
        self.tfidf_vectorizer = tfidf_vectorizer or TfidfVectorizer()
        self.sparcity = sparcity
        self._sparcity_threshold: Optional[float] = None
        self._composite_token_vectors: Optional[numpy.ndarray] = None

    def get_output_dim(self) -> int:
        num_clustrs = int(self.gaussian_mixture.n_components)
        embedding_size = self.token_embedder.get_output_dim()
        return num_clustrs * embedding_size

    def _fit(self, texts: List[List[str]], y: Any = None) -> TextVectorizer:
        self.token_embedder.fit(texts)
        self.tfidf_vectorizer.fit(texts)

        # Shape: (num_unique_tokens, )
        unique_tokens = self.tfidf_vectorizer.get_vocabulary()
        # Shape: (num_unique_tokens, )
        document_frequency = numpy.array(
            [
                self.tfidf_vectorizer.get_document_frequency(token)
                for token in unique_tokens
            ]
        )
        # Shape: (num_unique_tokens, )
        idfs = 1.0 / (document_frequency + 1e-10)

        num_unique_tokens = len(unique_tokens)
        embedding_size = self.token_embedder.get_output_dim()

        token_vectors = numpy.zeros((num_unique_tokens, embedding_size))
        for token_id, token in enumerate(unique_tokens):
            if token in self.token_embedder:
                token_vectors[token_id] = self.token_embedder[token]

        self.gaussian_mixture.fit(token_vectors)

        # Shape: (num_unique_tokens, num_clusters)
        cluster_probs = self.gaussian_mixture.predict_proba(token_vectors)
        # Shape: (num_unique_tokens, num_clusters, embedding_size)
        token_cluster_vector = numpy.einsum("ij,ik->ijk", cluster_probs, token_vectors)
        # Shape: (num_unique_tokens, num_cluster * embedding_size)
        composite_token_vectors = token_cluster_vector.reshape(num_unique_tokens, -1)
        composite_token_vectors = numpy.expand_dims(idfs, 1) * composite_token_vectors

        self._composite_token_vectors = composite_token_vectors
        self._sparcity_threshold = self._compute_sparcity_threshold(
            composite_token_vectors, self.sparcity
        )

        return self

    def _transoform(self, texts: List[List[str]]) -> numpy.ndarray:
        assert self._composite_token_vectors is not None

        num_examples = len(texts)
        embedding_size = self.get_output_dim()

        vocabulary = self.tfidf_vectorizer.get_vocabulary()

        outputs = numpy.zeros((num_examples, embedding_size))
        for text_index, tokens in enumerate(texts):
            for token in tokens:
                if token not in vocabulary:
                    continue
                token_id = self.tfidf_vectorizer.token_to_idex(token)
                token_vector = self._composite_token_vectors[token_id]
                outputs[text_index] += token_vector

            outputs[text_index] /= len(tokens)

        return self._make_sparse(outputs)

    @staticmethod
    def _compute_sparcity_threshold(vectors: numpy.ndarray, sparcity: float) -> float:
        a_min = vectors.min(1).mean()
        a_max = vectors.max(1).mean()
        t = 0.5 * (abs(a_min) + abs(a_max))
        return float(sparcity * t)

    def _make_sparse(self, vectors: numpy.ndarray) -> numpy.ndarray:
        outputs = numpy.where(vectors, numpy.abs(vectors) < self._sparcity_threshold, 0)
        return cast(numpy.ndarray, outputs)
