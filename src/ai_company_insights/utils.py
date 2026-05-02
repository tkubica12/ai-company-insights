from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from urllib.parse import urlparse


def normalize_company_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\b(a\.?\s*s\.?|s\.?\s*r\.?\s*o\.?|spol\.?\s*s\s*r\.?\s*o\.?)\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_probable_ico(value: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", value.strip()))


def host_from_url(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.").lower()


def truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."


def first_non_empty(values: Iterable[str | None]) -> str | None:
    return next((value for value in values if value), None)
