from __future__ import annotations

from typing import Any, List, Optional, Union, cast

import gensim
import numpy

from xsklearn.transformers.token_embedders.token_embedder import TokenEmbedder


class Word2VecEmbedder(TokenEmbedder):
    def __init__(
        self,
        model: Optional[Union[str, gensim.models.Word2Vec]] = None,
        min_count: int = 1,
        epochs: int = 10,
        **optional_params: Any,
    ) -> None:
        super().__init__()

        if isinstance(model, str):
            model = gensim.models.Word2Vec.load(model)

        self.model = model or gensim.models.Word2Vec(
            min_count=min_count,
            **optional_params,
        )
        self.epochs = epochs
        self.min_count = min_count
        self.optional_params = optional_params

    def fit(self, X: List[List[str]], y: Any = None) -> Word2VecEmbedder:
        self.model.build_vocab(X)
        self.model.train(
            corpus_iterable=X,
            total_examples=self.model.corpus_count,
            epochs=self.epochs,
        )
        return self

    def get(self, token: str) -> Optional[numpy.ndarray]:
        if token in self.model.wv:
            return cast(numpy.ndarray, self.model.wv[token])
        return None

    def get_output_dim(self) -> int:
        return int(self.model.vector_size)
