#!/usr/bin/env python
from urllib.parse import urljoin


def join_url_path(base_url: str, *path_parts: str) -> str:
    """Join a base URL with URL path segments without losing existing base path parts."""
    clean_base = base_url.rstrip("/") + "/"
    clean_path = "/".join(str(part).strip("/") for part in path_parts if part)
    return urljoin(clean_base, clean_path)
