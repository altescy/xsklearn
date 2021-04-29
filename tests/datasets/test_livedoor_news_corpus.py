import tempfile

from xsklearn.datasets.livedoor_news_corpus import fetch_livedoor_news_corpus


def test_fetch_livedoor_news_corpus() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        data = fetch_livedoor_news_corpus(tempdir)

    assert isinstance(data, list)
    assert set(data[0]) == {"media", "url", "published_at", "title", "body"}
