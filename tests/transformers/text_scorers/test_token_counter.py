from xsklearn.transformers.text_scorers import TokenCounter


def test_token_counter() -> None:
    model = TokenCounter()
    output = model.fit_transform(["this is a first sentence .", "foo bar baz"])

    assert output.shape == (2,)
    assert output.tolist() == [6, 3]
