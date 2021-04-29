from xsklearn.transformers.text_transformers import Lowercase


def test_lowercase() -> None:
    inputs = ["HeLLo WoRLD!", "fOO bAr bAz"]

    lower = Lowercase()
    results = lower.fit_transform(inputs)

    assert len(results) == 2
    assert results[0] == "hello world!"
    assert results[1] == "foo bar baz"
