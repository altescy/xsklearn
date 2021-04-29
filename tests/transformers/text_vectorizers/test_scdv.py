from xsklearn.transformers.text_vectorizers.scdv import SCDV


class TestSCDV:
    def setup(self) -> None:
        self.inputs = [
            ["this", "is", "a", "first", "sentence", "."],
            ["this", "is", "a", "second", "sentence", "."],
        ]

    def test_fit(self) -> None:
        model = SCDV()
        outputs = model.fit_transform(self.inputs)

        assert len(outputs) == 2
        assert outputs.shape == (2, model.get_output_dim())
