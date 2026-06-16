"""Metadata filters (sub-phase 4.3).

Translates the request `filters` field into a Qdrant filter applied during
retrieval. Supported keys:
  - tags:   a tag or list of tags; a chunk matches if it carries any of them.
  - folder: a vault folder (posix, e.g. "projects"); exact match.
"""

from __future__ import annotations

from qdrant_client import models


def build_filter(filters: dict | None) -> models.Filter | None:
    if not filters:
        return None

    must: list[models.FieldCondition] = []

    tags = filters.get("tags")
    if isinstance(tags, str):
        tags = [tags]
    if tags:
        must.append(
            models.FieldCondition(key="tags", match=models.MatchAny(any=list(tags)))
        )

    folder = filters.get("folder")
    if folder is not None:
        must.append(
            models.FieldCondition(key="folder", match=models.MatchValue(value=folder))
        )

    return models.Filter(must=must) if must else None
