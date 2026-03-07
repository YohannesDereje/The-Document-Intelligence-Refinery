from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Iterable, List
import warnings

from models import BBox, ExtractedDocument, LDU, PageIndex, SemanticChunk
from src.config import ConfigLoader
from src.utils.hashing import generate_content_hash


@dataclass
class _AtomicUnit:
    content: str
    content_type: str
    section_context: str
    page_numbers: list[int]
    bbox_bounds: BBox
    is_structural_boundary: bool


class ChunkValidator:
    def __init__(self, min_chunk_size: int) -> None:
        self.min_chunk_size = max(1, int(min_chunk_size))

    def validate_and_merge(self, chunks: list[SemanticChunk]) -> list[SemanticChunk]:
        if not chunks:
            return chunks

        merged: list[SemanticChunk] = []
        index = 0
        while index < len(chunks):
            current = chunks[index]
            is_final_chunk = index == (len(chunks) - 1)

            if current.token_count >= self.min_chunk_size or is_final_chunk:
                merged.append(current)
                index += 1
                continue

            warnings.warn(
                (
                    "ChunkValidator detected orphan chunk "
                    f"({current.token_count} tokens < {self.min_chunk_size}) and merged it."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

            if index + 1 < len(chunks):
                merged_chunk = self._merge_pair(current, chunks[index + 1])
                merged.append(merged_chunk)
                index += 2
            elif merged:
                merged[-1] = self._merge_pair(merged[-1], current)
                index += 1
            else:
                merged.append(current)
                index += 1

        return merged

    @staticmethod
    def _merge_pair(first: SemanticChunk, second: SemanticChunk) -> SemanticChunk:
        content = f"{first.content}\n\n{second.content}".strip()
        pages = sorted(set(first.page_numbers + second.page_numbers))
        bbox = BBox(
            x1=min(first.bbox_bounds.x1, second.bbox_bounds.x1),
            y1=min(first.bbox_bounds.y1, second.bbox_bounds.y1),
            x2=max(first.bbox_bounds.x2, second.bbox_bounds.x2),
            y2=max(first.bbox_bounds.y2, second.bbox_bounds.y2),
        )
        token_count = first.token_count + second.token_count

        # Keep section context of first chunk for stable retrieval ordering.
        page_index = ChunkingEngine.build_page_index(pages=pages, units=[first, second])
        return SemanticChunk(
            content=content,
            page_numbers=pages,
            bbox_bounds=bbox,
            section_context=first.section_context,
            token_count=token_count,
            page_index=page_index,
        )


class ChunkingEngine:
    def __init__(self, rules_path: str | Path = "rubric/extraction_rules.yaml") -> None:
        self.logger = logging.getLogger(__name__)
        self.config_loader = ConfigLoader(rules_path)

        self.max_chunk_size = int(self.config_loader.get("chunking_constitution.max_chunk_size", 1000))
        self.min_chunk_size = int(self.config_loader.get("chunking_constitution.min_chunk_size", 50))
        self.semantic_boundaries = [
            str(item).lower()
            for item in self.config_loader.get("chunking_constitution.semantic_boundary_indicators", [])
        ]

        self.header_min_font_size = float(
            self.config_loader.get("chunking_constitution.header_detection.min_font_size", 12.0)
        )
        self.bold_weight_threshold = int(
            self.config_loader.get("chunking_constitution.header_detection.bold_weight_threshold", 600)
        )

        self.validator = ChunkValidator(min_chunk_size=self.min_chunk_size)

    @staticmethod
    def _token_count(text: str) -> int:
        return max(1, len([token for token in text.split() if token]))

    @staticmethod
    def _unit_order_key(unit: LDU) -> tuple[int, str]:
        return int(unit.provenance.page_number), unit.uid

    def _extract_style_metadata(self, unit: LDU) -> dict[str, Any]:
        raw_text = unit.content_raw.strip()
        if not raw_text:
            return {}
        if raw_text.startswith("{") and raw_text.endswith("}"):
            try:
                payload = json.loads(raw_text)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                return {}
        return {}

    def _is_header_boundary(self, unit: LDU, current_section: str) -> bool:
        if unit.content_type == "header":
            return True

        text = unit.content_markdown.strip()
        metadata = self._extract_style_metadata(unit)
        font_size = float(metadata.get("font_size", metadata.get("size", 0.0)) or 0.0)
        font_weight = int(metadata.get("font_weight", metadata.get("weight", 0)) or 0)
        is_bold = bool(metadata.get("bold", metadata.get("is_bold", False)))

        if font_size >= self.header_min_font_size and (is_bold or font_weight >= self.bold_weight_threshold):
            return True

        lower_text = text.lower()
        if any(marker in lower_text for marker in self.semantic_boundaries):
            return True

        if text.startswith("#") and text != current_section:
            return True

        return False

    @staticmethod
    def _merge_bboxes(units: Iterable[LDU]) -> BBox:
        boxes = [unit.bbox for unit in units]
        if not boxes:
            return BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)

        return BBox(
            x1=min(box.x1 for box in boxes),
            y1=min(box.y1 for box in boxes),
            x2=max(box.x2 for box in boxes),
            y2=max(box.y2 for box in boxes),
        )

    @staticmethod
    def build_page_index(pages: list[int], units: list[Any]) -> list[PageIndex]:
        per_page_boxes: dict[int, list[BBox]] = {page: [] for page in pages}

        for unit in units:
            if isinstance(unit, LDU):
                page = int(unit.provenance.page_number)
                per_page_boxes.setdefault(page, []).append(unit.bbox)
            elif isinstance(unit, SemanticChunk):
                for page in unit.page_numbers:
                    per_page_boxes.setdefault(page, []).append(unit.bbox_bounds)

        page_index: list[PageIndex] = []
        for page in sorted(per_page_boxes):
            boxes = per_page_boxes.get(page, [])
            if not boxes:
                continue
            page_index.append(
                PageIndex(
                    page_number=page,
                    bbox=BBox(
                        x1=min(box.x1 for box in boxes),
                        y1=min(box.y1 for box in boxes),
                        x2=max(box.x2 for box in boxes),
                        y2=max(box.y2 for box in boxes),
                    ),
                )
            )

        return page_index

    def _to_atomic_units(self, document: ExtractedDocument) -> list[_AtomicUnit]:
        ordered_units = sorted(document.units, key=self._unit_order_key)
        atomic_units: list[_AtomicUnit] = []

        current_section = "Document Root"
        index = 0
        while index < len(ordered_units):
            unit = ordered_units[index]

            if unit.content_type == "table":
                table_units = [unit]
                index += 1
                while index < len(ordered_units) and ordered_units[index].content_type == "table":
                    table_units.append(ordered_units[index])
                    index += 1

                table_content = "\n\n".join(item.content_markdown for item in table_units).strip()
                table_pages = sorted({int(item.provenance.page_number) for item in table_units})
                table_bbox = self._merge_bboxes(table_units)
                atomic_units.append(
                    _AtomicUnit(
                        content=table_content,
                        content_type="table",
                        section_context=current_section,
                        page_numbers=table_pages,
                        bbox_bounds=table_bbox,
                        is_structural_boundary=False,
                    )
                )
                continue

            boundary = self._is_header_boundary(unit, current_section)
            if boundary:
                current_section = unit.content_markdown.strip()[:200] or current_section

            atomic_units.append(
                _AtomicUnit(
                    content=unit.content_markdown.strip(),
                    content_type=str(unit.content_type),
                    section_context=current_section,
                    page_numbers=[int(unit.provenance.page_number)],
                    bbox_bounds=unit.bbox,
                    is_structural_boundary=boundary,
                )
            )
            index += 1

        return [item for item in atomic_units if item.content]

    def _emit_chunk(self, units: list[_AtomicUnit]) -> SemanticChunk:
        content = "\n\n".join(unit.content for unit in units).strip()
        pages = sorted({page for unit in units for page in unit.page_numbers})

        bbox = BBox(
            x1=min(unit.bbox_bounds.x1 for unit in units),
            y1=min(unit.bbox_bounds.y1 for unit in units),
            x2=max(unit.bbox_bounds.x2 for unit in units),
            y2=max(unit.bbox_bounds.y2 for unit in units),
        )

        section_context = units[0].section_context if units else "Document Root"
        token_count = self._token_count(content)
        content_hash = generate_content_hash(text=content, page_number=pages[0], bbox=bbox)

        chunk = SemanticChunk(
            content=content,
            page_numbers=pages,
            bbox_bounds=bbox,
            section_context=section_context,
            token_count=token_count,
            # Provenance Verification: binds text + page anchor + bbox so reused text
            # in different documents/locations yields different hashes.
            content_hash=content_hash,
        )
        chunk = chunk.model_copy(update={"page_index": self.build_page_index(pages=pages, units=[chunk])})
        return chunk

    def _split_oversized_text_unit(self, unit: _AtomicUnit) -> list[_AtomicUnit]:
        if unit.content_type == "table":
            return [unit]

        words = unit.content.split()
        if not words:
            return [unit]

        output: list[_AtomicUnit] = []
        start = 0
        while start < len(words):
            end = min(len(words), start + self.max_chunk_size)
            content = " ".join(words[start:end]).strip()
            if not content:
                break
            output.append(
                _AtomicUnit(
                    content=content,
                    content_type=unit.content_type,
                    section_context=unit.section_context,
                    page_numbers=list(unit.page_numbers),
                    bbox_bounds=unit.bbox_bounds,
                    is_structural_boundary=False,
                )
            )
            start = end

        return output or [unit]

    def chunk_document(self, document: ExtractedDocument) -> list[SemanticChunk]:
        atomic_units = self._to_atomic_units(document)
        if not atomic_units:
            return []

        chunks: list[SemanticChunk] = []
        current_units: list[_AtomicUnit] = []
        current_tokens = 0

        for unit in atomic_units:
            candidate_units = [unit]
            unit_tokens = self._token_count(unit.content)
            if unit_tokens > self.max_chunk_size and unit.content_type != "table":
                candidate_units = self._split_oversized_text_unit(unit)

            for candidate in candidate_units:
                candidate_tokens = self._token_count(candidate.content)

                if current_units and candidate.is_structural_boundary:
                    chunks.append(self._emit_chunk(current_units))
                    current_units = []
                    current_tokens = 0

                exceeds = (current_tokens + candidate_tokens) > self.max_chunk_size
                if current_units and exceeds:
                    chunks.append(self._emit_chunk(current_units))
                    current_units = []
                    current_tokens = 0

                current_units.append(candidate)
                current_tokens += candidate_tokens

        if current_units:
            chunks.append(self._emit_chunk(current_units))

        validated = self.validator.validate_and_merge(chunks)
        self.logger.info("ChunkingEngine created %s semantic chunks for file_id=%s", len(validated), document.file_id)
        return validated
