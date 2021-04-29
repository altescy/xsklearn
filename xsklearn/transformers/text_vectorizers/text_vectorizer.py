from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

import numpy
from sklearn.base import TransformerMixin

from xsklearn.transformers.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


class TextVectorizer(TransformerMixin):  # type: ignore
    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer or WhitespaceTokenizer()

    def _tokenize(
        self,
        texts: Union[List[str], List[List[str]]],
    ) -> List[List[str]]:
        if isinstance(texts[0], str):
            return [self.tokenizer(text) for text in texts]  # type: ignore
        return texts  # type: ignore

    def fit(
        self,
        X: Union[List[str], List[List[str]]],
        y: Any = None,
    ) -> TextVectorizer:
        X = self._tokenize(X)
        return self._fit(X, y)

    def _fit(self, texts: List[List[str]], y: Any = None) -> TextVectorizer:
        return self

    def transform(
        self,
        X: Union[List[str], List[List[str]]],
    ) -> numpy.ndarray:
        X = self._tokenize(X)
        return self._transoform(X)

    def _transoform(self, texts: List[List[str]]) -> numpy.ndarray:
        raise NotImplementedError
