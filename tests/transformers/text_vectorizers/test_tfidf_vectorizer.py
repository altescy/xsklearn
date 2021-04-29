from xsklearn.transformers.text_vectorizers import TfidfVectorizer


class TestTfidfVectorizer:
    def setup(self) -> None:
        self.inputs = [
            ["this", "is", "a", "first", "sentence", "."],
            ["this", "is", "a", "second", "sentence", "."],
        ]

    def test_transform(self) -> None:
        vectorizer = TfidfVectorizer()
        outputs = vectorizer.fit_transform(self.inputs)
        assert outputs.shape == (2, 7)
