from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy

from xsklearn.exceptions import ConfigurationError
from xsklearn.transformers.text_vectorizers.text_vectorizer import TextVectorizer
from xsklearn.transformers.token_embedders.token_embedder import TokenEmbedder

REDUCERS: Dict[str, Callable[[numpy.ndarray], numpy.ndarray]] = {
    "max": lambda x: x.max(0),  # type: ignore
    "mean": lambda x: x.mean(0),  # type: ignore
    "sum": lambda x: x.sum(0),  # type: ignore
}


class BagOfEmbeddings(TextVectorizer):
    def __init__(
        self,
        token_embedder: TokenEmbedder,
        reducer: Union[str, Callable[[numpy.ndarray], numpy.ndarray]] = "mean",
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        super().__init__(tokenizer)
        if isinstance(reducer, str):
            if reducer not in REDUCERS:
                raise ConfigurationError(f"Invalid reducer: {reducer}")
            reducer = REDUCERS[reducer]

        self.token_embedder = token_embedder
        self.reducer = reducer

    def _fit(self, texts: List[List[str]], y: Any = None) -> BagOfEmbeddings:
        self.token_embedder.fit(texts, y)
        return self

    def _transoform(self, texts: List[List[str]]) -> numpy.ndarray:
        num_examples = len(texts)
        embedding_size = self.get_output_dim()

        embeddings_list = self.token_embedder.transform(texts)

        outputs = numpy.zeros((num_examples, embedding_size))
        for text_index, embeddings in enumerate(embeddings_list):
            vector = self.reducer(embeddings)
            outputs[text_index] = vector

        return outputs

    def get_output_dim(self) -> int:
        return self.token_embedder.get_output_dim()
