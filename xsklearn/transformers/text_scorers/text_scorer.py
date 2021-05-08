from __future__ import annotations

from typing import Any, List, Union

import numpy
from sklearn.base import TransformerMixin


class TextScorer(TransformerMixin):  # type: ignore
    def fit(
        self,
        X: Union[List[str], List[List[str]]],
        y: Any = None,
    ) -> TextScorer:
        return self

    def transform(self, X: Union[List[str], List[List[str]]]) -> numpy.ndarray:
        raise NotImplementedError
