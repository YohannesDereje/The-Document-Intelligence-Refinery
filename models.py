from __future__ import annotations

from typing import List

from pydantic import BaseModel


class PageProfile(BaseModel):
	page_number: int
	char_density: float
	rect_count: int
	is_landscape: bool
	dominant_strategy: str
	detected_origin: str


class DocumentProfile(BaseModel):
	file_name: str
	total_pages: int
	overall_origin: str
	domain_hint: str
	layout_complexity: str
	pages: List[PageProfile]
