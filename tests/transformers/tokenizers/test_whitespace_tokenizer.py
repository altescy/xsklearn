from xsklearn.transformers.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


def test_whitespace_tokenizer() -> None:
    inputs = ["hello world", "foo bar baz"]

    tokenizer = WhitespaceTokenizer()
    results = tokenizer.transform(inputs)

    assert len(results) == 2
    assert results[0] == ["hello", "world"]
    assert results[1] == ["foo", "bar", "baz"]
