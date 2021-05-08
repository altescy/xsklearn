import numpy

from xsklearn.transformers.text_scorers import FleschScorer


def test_flesch_scorer() -> None:
    text = "The Australian platypus is seemingly a hybrid of a mammal and reptilian creature"
    model = FleschScorer()
    output = model.fit_transform([text])

    correct_token_count = 13
    correct_syllable_count = 24
    correct_score = (
        0.39 * correct_token_count
        + 11.8 * (correct_syllable_count / correct_token_count)
        - 15.59
    )

    assert output.shape == (1,)
    numpy.testing.assert_almost_equal(numpy.array([correct_score]), output)
