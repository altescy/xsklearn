from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

import numpy
from sklearn.base import TransformerMixin

from xsklearn.transformers.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from xsklearn.util import tokenize_if_not_yet


class TextVectorizer(TransformerMixin):  # type: ignore
    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer or WhitespaceTokenizer()

    def fit(
        self,
        X: Union[List[str], List[List[str]]],
        y: Any = None,
    ) -> TextVectorizer:
        X = tokenize_if_not_yet(X, self.tokenizer)
        return self._fit(X, y)

    def _fit(self, texts: List[List[str]], y: Any = None) -> TextVectorizer:
        return self

    def transform(
        self,
        X: Union[List[str], List[List[str]]],
    ) -> numpy.ndarray:
        X = tokenize_if_not_yet(X, self.tokenizer)
        return self._transoform(X)

    def _transoform(self, texts: List[List[str]]) -> numpy.ndarray:
        raise NotImplementedError
