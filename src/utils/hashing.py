from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.schemas import BBox


def generate_content_hash(text: str, page_number: int, bbox: "BBox") -> str:
    """Generate a deterministic content hash anchored to text and spatial location."""
    normalized_text = " ".join((text or "").strip().lower().split())
    bbox_str = f"{bbox.x1:.2f},{bbox.y1:.2f},{bbox.x2:.2f},{bbox.y2:.2f}"
    seed = f"{normalized_text}|p:{int(page_number)}|bbox:{bbox_str}"
    return sha256(seed.encode("utf-8")).hexdigest()
