from xsklearn.transformers.text_scorers import SentenceCouter


def test_sentence_counter() -> None:
    model = SentenceCouter()
    output = model.fit_transform(["first sentence! second sentence.", "foo? bar; baz."])

    assert output.shape == (2,)
    assert output.tolist() == [2, 3]
