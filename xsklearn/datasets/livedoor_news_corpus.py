import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from xsklearn.util import cached_path


def fetch_livedoor_news_corpus(
    cache_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, str]]:
    URL = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"

    filename = cached_path(URL, cache_dir=cache_dir)

    with tempfile.TemporaryDirectory() as _tempdir:
        tempdir = Path(_tempdir)

        with tarfile.open(filename) as tar:
            tar.extractall(tempdir)

        file_paths = [
            path
            for path in tempdir.glob("text/*/*.txt")
            if not path.match("LICENSE.txt")
        ]

        data = []
        for path in file_paths:
            with path.open() as f:
                media = path.parent.name
                url = f.readline().strip()
                published_at = f.readline().strip()
                title = f.readline().strip()
                body = f.read().strip()

                data.append(
                    {
                        "media": media,
                        "url": url,
                        "published_at": published_at,
                        "title": title,
                        "body": body,
                    }
                )

    return data
