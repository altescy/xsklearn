from __future__ import annotations

from typing import Any, List

from sklearn.base import TransformerMixin


class Tokenizer(TransformerMixin):  # type: ignore
    def __call__(self, texts: List[str]) -> List[List[str]]:
        return self.transform(texts)

    def fit(self, X: List[str], y: Any = None) -> Tokenizer:
        return self

    def transform(self, X: List[str]) -> List[List[str]]:
        raise NotImplementedError
