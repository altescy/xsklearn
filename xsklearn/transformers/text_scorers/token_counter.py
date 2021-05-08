from typing import Callable, List, Optional, Union

import numpy

from xsklearn.transformers.text_scorers.text_scorer import TextScorer
from xsklearn.transformers.tokenizers import WhitespaceTokenizer
from xsklearn.util import tokenize_if_not_yet


class TokenCounter(TextScorer):
    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer or WhitespaceTokenizer()

    def transform(self, X: Union[List[str], List[List[str]]]) -> numpy.ndarray:
        tokenized_text = tokenize_if_not_yet(X, self.tokenizer)
        return numpy.array([len(tokens) for tokens in tokenized_text])
