from typing import Any, Callable, List, Optional

import numpy

from xsklearn.transformers.text_scorers.text_scorer import TextScorer


def _default_tokenizer(text: str) -> List[str]:
    return text.strip().split()


class TokenCounter(TextScorer):
    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer or _default_tokenizer

    def fit(self, X: List[str], y: Any = None) -> TextScorer:
        return self

    def transform(self, X: List[str]) -> numpy.ndarray:
        return numpy.array([len(self.tokenizer(text)) for text in X])
