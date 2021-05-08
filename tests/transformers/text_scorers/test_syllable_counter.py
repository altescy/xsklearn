from xsklearn.transformers.text_scorers import SyllableCounter


def test_syllable_counter() -> None:
    model = SyllableCounter()
    output = model.fit_transform(["this is a first sentence", "hello world"])

    assert output.shape == (2,)
    assert output.tolist() == [6, 3]
