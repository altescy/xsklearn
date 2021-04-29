from __future__ import annotations

from typing import Any, List

from sklearn.base import TransformerMixin


class TextTransformer(TransformerMixin):  # type: ignore
    def fit(self, X: List[str], y: Any = None) -> TextTransformer:
        return self

    def transform(self, X: List[str]) -> List[str]:
        raise NotImplementedError


class Lowercase(TextTransformer):
    def transform(self, X: List[str]) -> List[str]:
        return [x.lower() for x in X]
