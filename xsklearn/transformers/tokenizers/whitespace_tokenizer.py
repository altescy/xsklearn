from typing import List

from xsklearn.transformers.tokenizers.tokenizer import Tokenizer


class WhitespaceTokenizer(Tokenizer):
    def transform(self, X: List[str]) -> List[List[str]]:
        return [x.strip().split() for x in X]
