import re
from typing import Callable, List, Optional, Union, cast

import numpy

from xsklearn.transformers.text_scorers.text_scorer import TextScorer


class SentenceCouter(TextScorer):
    def __init__(
        self,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        super().__init__()
        self.sentence_splitter = sentence_splitter or self._default_sentence_splitter

    @staticmethod
    def _default_sentence_splitter(text: str) -> List[str]:
        return [sentence for sentence in re.split(r"[\.\!\?\;]", text) if sentence]

    def _convert_token_list_to_sentence(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def transform(self, X: Union[List[str], List[List[str]]]) -> numpy.ndarray:
        if isinstance(X[0], list):
            X = cast(List[List[str]], X)
            texts = [self._convert_token_list_to_sentence(tokens) for tokens in X]
        else:
            texts = cast(List[str], X)

        splitted_texts = [self.sentence_splitter(text) for text in texts]
        return numpy.array([len(sentences) for sentences in splitted_texts])
