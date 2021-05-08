from typing import Callable, List, Optional, Union

import numpy

from xsklearn.transformers.text_scorers.text_scorer import TextScorer
from xsklearn.transformers.tokenizers import WhitespaceTokenizer
from xsklearn.util import tokenize_if_not_yet


class SyllableCounter(TextScorer):
    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer or WhitespaceTokenizer()

    def _count_syllable(self, token: str) -> int:
        if not token:
            return 0

        vowels = "aeiouy"

        token = token.lower()
        count = 0
        if token[0] in vowels:
            count += 1
        for index in range(1, len(token)):
            if token[index] in vowels and token[index - 1] not in vowels:
                count += 1

        if token.endswith("e"):
            count -= 1

        if count == 0:
            count += 1

        return count

    def transform(self, X: Union[List[str], List[List[str]]]) -> numpy.ndarray:
        tokenized_text = tokenize_if_not_yet(X, self.tokenizer)
        return numpy.array(
            [
                sum(self._count_syllable(token) for token in tokens)
                for tokens in tokenized_text
            ]
        )
