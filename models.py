from __future__ import annotations

from typing import Any, Dict, List

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


class ProvenanceChain(BaseModel):
	source_file: str
	page_number: int
	strategy_used: str
	timestamp: str


class LDU(BaseModel):
	uid: str
	content_type: str
	content_raw: str
	content_markdown: str
	provenance: ProvenanceChain


class ExtractedDocument(BaseModel):
	file_id: str
	domain_hint: str
	metadata: Dict[str, Any]
	units: List[LDU]


class ExtractionLedgerEntry(BaseModel):
	file_name: str
	strategy_selected: str
	confidence_score: float
	processing_time_ms: float
	estimated_cost_usd: float
