from __future__ import annotations

from typing import Any, List

import numpy
from sklearn.base import TransformerMixin


class TextScorer(TransformerMixin):  # type: ignore
    def fit(self, X: List[str], y: Any = None) -> TextScorer:
        return self

    def transform(self, X: List[str]) -> numpy.ndarray:
        raise NotImplementedError
