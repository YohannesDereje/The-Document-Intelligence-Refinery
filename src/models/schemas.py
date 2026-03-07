from __future__ import annotations

from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from src.utils.hashing import generate_content_hash


class StrategyName(str, Enum):
	STRATEGY_A = "STRATEGY_A"
	STRATEGY_B = "STRATEGY_B"
	STRATEGY_C = "STRATEGY_C"


DocumentType = Literal["born_digital", "scanned", "hybrid", "unknown"]
UnitType = Literal["text", "table", "header", "figure"]
LayoutComplexity = Literal["Low", "Medium", "High"]
CostTier = Literal["LOW", "MEDIUM", "HIGH"]


class BBox(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_assignment=True)

	x1: float
	y1: float
	x2: float
	y2: float

	@model_validator(mode="after")
	def validate_ranges(self) -> "BBox":
		if self.x2 < self.x1 or self.y2 < self.y1:
			raise ValueError("BBox coordinates must satisfy x2 >= x1 and y2 >= y1.")

		values = (self.x1, self.y1, self.x2, self.y2)
		normalized = all(0.0 <= value <= 1.0 for value in values)
		pixel_like = all(value >= 0.0 for value in values)

		if not normalized and not pixel_like:
			raise ValueError("BBox must be normalized [0.0-1.0] or non-negative pixel coordinates.")

		return self


class PageIndex(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_assignment=True)

	page_number: int
	bbox: BBox
	node_id: str = ""
	node_type: Literal["root", "section", "subsection", "chunk", "page"] = "page"
	title: str = ""
	summary: str = ""
	metadata: Dict[str, Any] = Field(default_factory=dict)
	chunk_hashes: List[str] = Field(default_factory=list)
	children: List["PageIndex"] = Field(default_factory=list)

	@field_validator("page_number")
	@classmethod
	def page_number_positive(cls, value: int) -> int:
		if value < 0:
			raise ValueError("page_number must be non-negative.")
		return value


class PageProfile(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_assignment=True)

	page_number: int
	char_density: float
	rect_count: int
	is_landscape: bool
	dominant_strategy: StrategyName
	detected_origin: DocumentType

	@field_validator("page_number")
	@classmethod
	def page_number_positive(cls, value: int) -> int:
		if value <= 0:
			raise ValueError("page_number must be positive.")
		return value


class DocumentProfile(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_assignment=True)

	file_name: str
	total_pages: int
	overall_origin: DocumentType
	origin_type: Literal["born_digital", "scanned_image", "hybrid", "unknown"] = "unknown"
	domain_hint: str
	layout_complexity: LayoutComplexity
	cost_tier: CostTier = "MEDIUM"
	triage_confidence: float = 0.5
	mixed_mode: bool = False
	form_fillable: bool = False
	document_type: DocumentType = "unknown"
	pages: List[PageProfile]

	@field_validator("total_pages")
	@classmethod
	def total_pages_non_negative(cls, value: int) -> int:
		if value < 0:
			raise ValueError("total_pages cannot be negative.")
		return value

	@field_validator("triage_confidence")
	@classmethod
	def confidence_range(cls, value: float) -> float:
		if value < 0.0 or value > 1.0:
			raise ValueError("triage_confidence must be between 0.0 and 1.0.")
		return value


class ProvenanceChain(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_assignment=True)

	source_file: str
	page_number: int
	strategy_used: StrategyName
	timestamp: str
	strategy_escalation_path: List[StrategyName] = Field(default_factory=list)
	content_hash: str = ""

	@field_validator("page_number")
	@classmethod
	def page_number_positive(cls, value: int) -> int:
		if value <= 0:
			raise ValueError("page_number must be positive.")
		return value

	@model_validator(mode="after")
	def compute_or_validate_hash(self) -> "ProvenanceChain":
		seed = (
			f"{self.source_file}|{self.page_number}|{self.strategy_used.value}|"
			f"{self.timestamp}|{','.join(item.value for item in self.strategy_escalation_path)}"
		)
		expected = sha256(seed.encode("utf-8")).hexdigest()
		if not self.content_hash:
			self.content_hash = expected
		elif self.content_hash != expected:
			raise ValueError("ProvenanceChain.content_hash does not match computed provenance hash.")
		return self


class LDU(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_assignment=True)

	uid: str
	unit_type: UnitType = Field(validation_alias="content_type")
	content_raw: str
	content_markdown: str
	bbox: BBox = Field(default_factory=lambda: BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0))
	provenance: ProvenanceChain
	content_hash: str = ""

	@property
	def content_type(self) -> UnitType:
		return self.unit_type

	@field_validator("content_raw", "content_markdown")
	@classmethod
	def content_non_empty(cls, value: str) -> str:
		if not value or not value.strip():
			raise ValueError("content strings cannot be empty or whitespace.")
		return value

	@model_validator(mode="after")
	def validate_content_hash(self) -> "LDU":
		expected = generate_content_hash(
			text=self.content_raw,
			page_number=self.provenance.page_number,
			bbox=self.bbox,
		)
		if not self.content_hash:
			self.content_hash = expected
		elif self.content_hash != expected:
			raise ValueError("LDU.content_hash does not match spatial content hash.")
		return self


class ExtractedDocument(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_assignment=True)

	file_id: str
	domain_hint: str
	document_type: DocumentType = "unknown"
	metadata: Dict[str, Any]
	units: List[LDU]
	page_index: List[PageIndex] = Field(default_factory=list)


class SemanticChunk(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_assignment=True)

	content: str
	page_numbers: List[int]
	bbox_bounds: BBox
	section_context: str
	token_count: int
	content_hash: str = ""
	page_index: List[PageIndex] = Field(default_factory=list)

	@field_validator("content", "section_context")
	@classmethod
	def non_empty_text(cls, value: str) -> str:
		if not value or not value.strip():
			raise ValueError("chunk text fields cannot be empty.")
		return value

	@field_validator("page_numbers")
	@classmethod
	def non_empty_pages(cls, value: List[int]) -> List[int]:
		if not value:
			raise ValueError("page_numbers must contain at least one page.")
		if any(page <= 0 for page in value):
			raise ValueError("page_numbers must be positive integers.")
		return sorted(set(value))

	@field_validator("token_count")
	@classmethod
	def positive_tokens(cls, value: int) -> int:
		if value <= 0:
			raise ValueError("token_count must be positive.")
		return value

	@model_validator(mode="after")
	def validate_content_hash(self) -> "SemanticChunk":
		reference_page = self.page_numbers[0]
		expected = generate_content_hash(
			text=self.content,
			page_number=reference_page,
			bbox=self.bbox_bounds,
		)
		if not self.content_hash:
			self.content_hash = expected
		elif self.content_hash != expected:
			raise ValueError("SemanticChunk.content_hash does not match spatial content hash.")
		return self


class ExtractionAttempt(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_assignment=True)

	strategy: StrategyName
	status: Literal["success", "fail"]
	confidence_score: float
	processing_time_ms: float
	estimated_cost_usd: float
	prompt_tokens: int = 0
	image_tokens: int = 0
	completion_tokens: int = 0

	@field_validator("confidence_score")
	@classmethod
	def confidence_range(cls, value: float) -> float:
		if value < 0.0 or value > 1.0:
			raise ValueError("confidence_score must be between 0.0 and 1.0.")
		return value

	@field_validator("processing_time_ms", "estimated_cost_usd")
	@classmethod
	def non_negative_metric(cls, value: float) -> float:
		if value < 0.0:
			raise ValueError("attempt metrics must be non-negative.")
		return value

	@field_validator("prompt_tokens", "image_tokens", "completion_tokens")
	@classmethod
	def non_negative_tokens(cls, value: int) -> int:
		if value < 0:
			raise ValueError("token counters must be non-negative.")
		return value


class ExtractionResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_assignment=True)

	page_number: int
	origin_type: str
	attempts: List[ExtractionAttempt]
	final_strategy: StrategyName
	final_status: Literal["success", "fail"]
	final_confidence_score: float
	escalation_path: str
	cumulative_processing_time_ms: float = 0.0
	cumulative_cost_usd: float = 0.0
	total_cost_usd: float = 0.0
	final_success_cost_usd: float = 0.0
	final_vlm_cost_usd: float = 0.0

	@field_validator("page_number")
	@classmethod
	def page_number_positive(cls, value: int) -> int:
		if value <= 0:
			raise ValueError("page_number must be positive.")
		return value

	@field_validator("final_confidence_score")
	@classmethod
	def final_confidence_range(cls, value: float) -> float:
		if value < 0.0 or value > 1.0:
			raise ValueError("final_confidence_score must be between 0.0 and 1.0.")
		return value

	@field_validator(
		"cumulative_processing_time_ms",
		"cumulative_cost_usd",
		"total_cost_usd",
		"final_success_cost_usd",
		"final_vlm_cost_usd",
	)
	@classmethod
	def non_negative_aggregate_metrics(cls, value: float) -> float:
		if value < 0.0:
			raise ValueError("aggregate cost/time metrics must be non-negative.")
		return value


class ExtractionLedgerEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_assignment=True)

	file_name: str
	page_number: int = 0
	origin_type: str = "unknown"
	strategy_selected: StrategyName
	confidence_score: float
	processing_time_ms: float
	estimated_cost_usd: float
	final_success_cost_usd: float = 0.0
	final_vlm_cost_usd: float = 0.0
	attempt_chain: List[ExtractionAttempt] = Field(default_factory=list)
	escalation_path: str = ""

	@field_validator("page_number")
	@classmethod
	def ledger_page_number_non_negative(cls, value: int) -> int:
		if value < 0:
			raise ValueError("page_number cannot be negative.")
		return value

	@field_validator(
		"processing_time_ms",
		"estimated_cost_usd",
		"final_success_cost_usd",
		"final_vlm_cost_usd",
	)
	@classmethod
	def ledger_metrics_non_negative(cls, value: float) -> float:
		if value < 0.0:
			raise ValueError("ledger metrics must be non-negative.")
		return value


PageIndex.model_rebuild()
