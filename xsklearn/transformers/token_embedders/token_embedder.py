from __future__ import annotations

from typing import Any, List, Optional

import numpy
from sklearn.base import TransformerMixin


class TokenEmbedder(TransformerMixin):  # type: ignore
    def __init__(
        self,
        unknow_vector: Optional[numpy.ndarray] = None,
    ) -> None:
        """
        Parameter
        =========
            unknow_vector: `numpy.ndarray`, optional (default = None)
                This is used for word vector of out-of-vocabulary
                tokens. If this is `None`, ignore the unknown token.
        """
        super().__init__()
        self.unknow_vector = unknow_vector

    def fit(self, X: List[List[str]], y: Any = None) -> TokenEmbedder:
        return self

    def transform(self, X: List[List[str]]) -> List[numpy.ndarray]:
        embeddings_list: List[numpy.ndarray] = []
        for tokens in X:
            embedding_list = []
            for token in tokens:
                embedding = self.get(token)

                if embedding is None:
                    if self.unknow_vector is None:
                        continue
                    embedding = self.unknow_vector.copy()

                embedding_list.append(embedding)

            # Shape: (num_tokens, embedding_size)
            embeddings = numpy.vstack(embedding_list)
            embeddings_list.append(embeddings)

        return embeddings_list

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def get(self, token: str) -> Optional[numpy.ndarray]:
        raise NotImplementedError

    def __getitem__(self, token: str) -> numpy.ndarray:
        embedding = self.get(token)
        if embedding is None:
            raise KeyError(f"Unknow token: {token}")
        return embedding

    def __contains__(self, token: str) -> bool:
        return self.get(token) is not None
