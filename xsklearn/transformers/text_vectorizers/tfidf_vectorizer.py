from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Union

import numpy
import scipy.sparse

from xsklearn.transformers.text_vectorizers.text_vectorizer import TextVectorizer


class TfidfVectorizer(TextVectorizer):
    def __init__(
        self,
        max_df: Union[int, float] = 1.0,
        min_df: Union[int, float] = 1,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        super().__init__(tokenizer)
        self.max_df = max_df
        self.min_df = min_df
        self._document_frequency: Dict[str, int] = {}
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

    def get_output_dim(self) -> int:
        return len(self._token_to_id)

    def _fit(self, texts: List[List[str]], y: Any = None) -> TextVectorizer:
        for tokens in texts:
            for token in set(tokens):
                if token not in self._document_frequency:
                    self._document_frequency[token] = 0

                self._document_frequency[token] += 1

        max_df = (
            self.max_df if isinstance(self.max_df, int) else len(texts) * self.max_df
        )
        min_df = (
            self.min_df if isinstance(self.min_df, int) else len(texts) * self.min_df
        )

        ignored_tokens: List[str] = []
        for token, df in self._document_frequency.items():
            if not (min_df <= df <= max_df):
                ignored_tokens.append(token)

        for token in ignored_tokens:
            del self._document_frequency[token]

        for token_id, token in enumerate(sorted(self._document_frequency)):
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token

        return self

    def _transoform(self, texts: List[List[str]]) -> numpy.ndarray:
        num_examples = len(texts)
        vocabulary_size = len(self._token_to_id)

        outputs = scipy.sparse.csr_matrix((num_examples, vocabulary_size))
        for text_index, tokens in enumerate(texts):
            term_frequencies = Counter(tokens)
            for token in tokens:
                if token not in self._document_frequency:
                    continue
                token_id = self._token_to_id[token]
                tf = term_frequencies[token]
                df = self._document_frequency[token]
                tfidf = tf / (df + 1e-10)
                outputs[text_index, token_id] = tfidf

        return outputs  # type: ignore

    def get_document_frequency(self, token: str) -> int:
        return self._document_frequency.get(token, 0)

    def get_vocabulary(self) -> List[str]:
        return list(self._token_to_id)

    def token_to_idex(self, token: str) -> int:
        return self._token_to_id[token]

    def index_to_token(self, index: int) -> str:
        return self._id_to_token[index]
