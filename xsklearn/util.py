import hashlib
import logging
import os
from pathlib import Path
from typing import IO, Any, Callable, List, Optional, Union, cast

import requests
from tqdm import tqdm

from xsklearn.settings import CACHE_DIRRECTORY

logger = logging.getLogger(__name__)


def tokenize_if_not_yet(
    texts: Union[List[str], List[List[str]]],
    tokenizer: Callable[[str], List[str]],
) -> List[List[str]]:
    if isinstance(texts[0], str):
        texts = cast(List[str], texts)
        return [tokenizer(text) for text in texts]

    return cast(List[List[str]], texts)


def cached_path(
    url_or_filename: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    if not is_url(str(url_or_filename)):
        return Path(url_or_filename)

    url = str(url_or_filename)

    cache_dir = Path(cache_dir or CACHE_DIRRECTORY)

    os.makedirs(cache_dir, exist_ok=True)

    cache_path = cache_dir / _get_cached_filename(url_or_filename)
    if cache_path.exists():
        logger.info("use cache for %s: %s", str(url_or_filename), str(cache_path))
        return cache_path

    with open(cache_path, "wb") as fp:
        _http_get(url, fp)

    return cache_path


def is_url(text: str) -> bool:
    return text.startswith("http")


def _http_get(url: str, temp_file: IO[Any]) -> None:
    with requests.Session() as session:
        req = session.get(url, stream=True)
        req.raise_for_status()
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, desc="downloading")
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                temp_file.write(chunk)
        progress.close()


def _get_cached_filename(path: Union[str, Path]) -> str:
    encoded_path = str(path).encode()
    name = hashlib.md5(encoded_path).hexdigest()
    return name
