from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Union, cast

import numpy

from xsklearn.transformers.text_scorers.sentence_counter import SentenceCouter
from xsklearn.transformers.text_scorers.syllable_counter import SyllableCounter
from xsklearn.transformers.text_scorers.text_scorer import TextScorer
from xsklearn.transformers.text_scorers.token_counter import TokenCounter


class FleschScorer(TextScorer):
    def __init__(
        self,
        token_counter: Optional[TextScorer] = None,
        sentence_counter: Optional[TextScorer] = None,
        syllable_counter: Optional[TextScorer] = None,
    ) -> None:
        super().__init__()
        self.token_counter = token_counter or TokenCounter()
        self.sentence_counter = sentence_counter or SentenceCouter()
        self.syllable_counter = syllable_counter or SyllableCounter()

    def fit(
        self,
        X: Union[List[str], List[List[str]]],
        y: Any = None,
    ) -> FleschScorer:
        self.token_counter.fit(X, y)
        self.sentence_counter.fit(X, y)
        self.syllable_counter.fit(X, y)
        return self

    def transform(self, X: Union[List[str], List[List[str]]]) -> numpy.ndarray:
        token_counts = self.token_counter.transform(X)
        sentence_counts = self.sentence_counter.transform(X)
        syllable_counts = self.syllable_counter.transform(X)

        scores = (
            0.39 * (token_counts / sentence_counts)
            + 11.8 * (syllable_counts / token_counts)
            - 15.59
        )
        return cast(numpy.ndarray, scores)
