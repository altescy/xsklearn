from xsklearn.transformers.tokenizers import JanomeTokenizer


def test_janome_tokenizer() -> None:
    inputs = ["すもももももももものうち"]

    tokenizer = JanomeTokenizer()
    outputs = tokenizer.transform(inputs)

    assert len(outputs) == 1
    assert outputs[0] == ["すもも", "も", "もも", "も", "もも", "の", "うち"]
