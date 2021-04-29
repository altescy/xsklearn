import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from xsklearn.datasets import fetch_livedoor_news_corpus
from xsklearn.transformers.text_vectorizers import SCDV
from xsklearn.transformers.tokenizers import JanomeTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("fetch livedor news corpus...")
    data = fetch_livedoor_news_corpus()

    X = [item["body"] for item in data]
    y = [item["media"] for item in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X_train = X_train[:1000]
    y_train = y_train[:1000]

    logger.info(
        "data size (train / test): %s / %s",
        len(X_train),
        len(X_test),
    )

    classifier = make_pipeline(
        JanomeTokenizer(),
        SCDV(),
        RandomForestClassifier(),
    )
    logger.info("classifier: %s", repr(classifier))

    logger.info("start training...")
    classifier.fit(X_train, y_train)

    score = classifier.score(X_test, y_test)

    logger.info("test accuracy: %s", score)


if __name__ == "__main__":
    main()
