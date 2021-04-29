from typing import List

from xsklearn.transformers.tokenizers.tokenizer import Tokenizer

try:
    from janome.tokenizer import Tokenizer as _JanomeTokenizer
except ImportError:
    _JanomeTokenizer = None


class JanomeTokenizer(Tokenizer):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        if _JanomeTokenizer is None:
            raise ImportError(
                "Failed to import janome. Make sure " "janome is successfully installed"
            )

        super().__init__()
        self._janome_tokenizer = _JanomeTokenizer(*args, **kwargs)

    def _tokenize(self, text: str) -> List[str]:
        return [
            str(token) for token in self._janome_tokenizer.tokenize(text, wakati=True)
        ]

    def transform(self, X: List[str]) -> List[List[str]]:
        return [self._tokenize(x) for x in X]
