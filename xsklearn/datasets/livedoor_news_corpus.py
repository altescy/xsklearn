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
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, tempdir)

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
