from typing import Optional, Union, cast

import gensim
import numpy

from xsklearn.transformers.token_embedders.token_embedder import TokenEmbedder


class Word2VecEmbedder(TokenEmbedder):
    def __init__(
        self,
        model: Union[str, gensim.models.Word2Vec],
    ) -> None:
        super().__init__()

        if isinstance(model, str):
            model = gensim.models.Word2Vec.load(model)

        self.model = model

    def get(self, token: str) -> Optional[numpy.ndarray]:
        if token in self.model.wv:
            return cast(numpy.ndarray, self.model.wv[token])
        return None

    def get_output_dim(self) -> int:
        return int(self.model.vector_size)
