from __future__ import annotations

import json
import logging
import inspect
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List

import pdfplumber
import yaml
from docling.document_converter import DocumentConverter

try:
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption
except Exception:  # pragma: no cover
    PyPdfiumDocumentBackend = None
    InputFormat = None
    PdfFormatOption = None

from models import LDU, ProvenanceChain

from .base_strategy import BaseExtractor
from .vision_extractor import VisionExtractor


class DoclingDocumentAdapter:
    """Normalizes Docling conversion outputs to a stable piece iteration interface."""

    def __init__(self, conversion_result: Any) -> None:
        self.conversion_result = conversion_result
        self.document = self._resolve_document(conversion_result)

    @staticmethod
    def _resolve_document(conversion_result: Any) -> Any:
        for attr in ("document", "doc", "result"):
            if hasattr(conversion_result, attr):
                return getattr(conversion_result, attr)
        return conversion_result

    @staticmethod
    def _collection_from_object(obj: Any, name: str) -> list[Any]:
        if not hasattr(obj, name):
            return []
        value = getattr(obj, name)
        if isinstance(value, list):
            return value
        if value is None:
            return []
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _detect_page_number(piece: Any) -> int:
        for attr in ("page_no", "page_number", "page", "page_idx", "page_index"):
            if hasattr(piece, attr):
                try:
                    number = int(getattr(piece, attr))
                    if number >= 1:
                        return number
                    return number + 1
                except (TypeError, ValueError):
                    pass

        provenance = getattr(piece, "provenance", None)
        if provenance is not None:
            for attr in ("page_no", "page_number", "page", "page_idx", "page_index"):
                if hasattr(provenance, attr):
                    try:
                        number = int(getattr(provenance, attr))
                        if number >= 1:
                            return number
                        return number + 1
                    except (TypeError, ValueError):
                        pass

        return 1

    @staticmethod
    def piece_type(piece: Any) -> str:
        raw_type = str(getattr(piece, "type", "") or getattr(piece, "label", "")).lower()
        class_name = piece.__class__.__name__.lower()
        if "table" in raw_type or "table" in class_name:
            return "table"
        if "figure" in raw_type or "image" in raw_type or "figure" in class_name:
            return "figure"
        return "text"

    def iter_pieces(self, page_numbers: set[int] | None = None) -> Iterable[tuple[int, Any]]:
        candidate_pieces: list[Any] = []

        for field in ("texts", "tables", "figures", "items", "elements", "content"):
            candidate_pieces.extend(self._collection_from_object(self.document, field))

        if not candidate_pieces and hasattr(self.document, "pages"):
            pages = getattr(self.document, "pages") or []
            for page in pages:
                for field in ("texts", "tables", "figures", "items", "elements", "content"):
                    candidate_pieces.extend(self._collection_from_object(page, field))

        for piece in candidate_pieces:
            page_no = self._detect_page_number(piece)
            if page_numbers is None or page_no in page_numbers:
                yield page_no, piece

    def piece_to_markdown(self, piece: Any) -> str:
        if self.piece_type(piece) == "table":
            for method_name in ("export_to_markdown", "to_markdown", "as_markdown"):
                if hasattr(piece, method_name):
                    method = getattr(piece, method_name)
                    if callable(method):
                        return str(method() or "")
            return self.piece_to_text(piece)

        return self.piece_to_text(piece)

    def piece_to_text(self, piece: Any) -> str:
        for attr in ("text", "content", "raw_text", "value"):
            if hasattr(piece, attr):
                value = getattr(piece, attr)
                if value is not None:
                    return str(value)

        for method_name in ("export_to_text", "to_text", "as_text"):
            if hasattr(piece, method_name):
                method = getattr(piece, method_name)
                if callable(method):
                    value = method()
                    if value is not None:
                        return str(value)

        return str(piece)

    def piece_to_json(self, piece: Any) -> str:
        payload: Any
        if hasattr(piece, "model_dump") and callable(piece.model_dump):
            payload = piece.model_dump()
        elif hasattr(piece, "to_dict") and callable(piece.to_dict):
            payload = piece.to_dict()
        elif hasattr(piece, "export_to_dict") and callable(piece.export_to_dict):
            payload = piece.export_to_dict()
        elif hasattr(piece, "__dict__"):
            payload = {k: v for k, v in vars(piece).items() if not k.startswith("_")}
        else:
            payload = str(piece)

        return json.dumps(payload, ensure_ascii=False, default=str)


class LayoutExtractor(BaseExtractor):
    def __init__(self, rules_path: Path | None = None, warning_confidence_threshold: float = 0.7) -> None:
        self.logger = logging.getLogger(__name__)
        self.warning_confidence_threshold = warning_confidence_threshold
        self.min_digital_density = 0.0005

        default_rules_path = Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
        self.rules_path = rules_path or default_rules_path
        self.min_digital_density = self._load_min_digital_density()

        if PyPdfiumDocumentBackend is not None and InputFormat is not None and PdfFormatOption is not None:
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(backend=PyPdfiumDocumentBackend),
                }
            )
            self.logger.info("LayoutExtractor initialized with PyPdfiumDocumentBackend")
        else:
            self.converter = DocumentConverter()
            self.logger.warning(
                "PyPdfiumDocumentBackend not available; falling back to default Docling backend."
            )

        self.vision_fallback: VisionExtractor | None = None

    def _get_vision_fallback(self) -> VisionExtractor:
        if self.vision_fallback is None:
            self.vision_fallback = VisionExtractor(
                api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
                model_name="google/gemini-flash-1.5",
            )
        return self.vision_fallback

    def _load_min_digital_density(self) -> float:
        if not self.rules_path.exists():
            return 0.0005

        raw_yaml = self.rules_path.read_text(encoding="utf-8")
        normalized_yaml = raw_yaml.replace("\t", "  ")
        config = yaml.safe_load(normalized_yaml) or {}
        thresholds = config.get("thresholds", {}) if isinstance(config, dict) else {}
        return float(thresholds.get("MIN_DIGITAL_DENSITY", 0.0005))

    @staticmethod
    def _page_density_map(pdf_path: Path, page_numbers: set[int] | None) -> dict[int, float]:
        densities: dict[int, float] = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                if page_numbers is not None and page_number not in page_numbers:
                    continue
                text = page.extract_text() or ""
                area = float(page.width or 0.0) * float(page.height or 0.0)
                densities[page_number] = (len(text) / area) if area > 0 else 0.0
        return densities

    def _convert_pdf(self, pdf_path: Path, selected_pages: set[int] | None) -> Any:
        if not selected_pages:
            return self.converter.convert(str(pdf_path))

        ordered_pages = sorted(selected_pages)
        supported_params = set(inspect.signature(self.converter.convert).parameters.keys())

        candidate_kwargs: list[dict[str, Any]] = [
            {"page_numbers": ordered_pages},
            {"pages": ordered_pages},
            {"page_range": ordered_pages},
            {"first_page": min(ordered_pages), "last_page": max(ordered_pages)},
        ]

        for kwargs in candidate_kwargs:
            filtered = {k: v for k, v in kwargs.items() if k in supported_params}
            if not filtered:
                continue
            try:
                return self.converter.convert(str(pdf_path), **filtered)
            except Exception as exc:
                self.logger.warning("Docling conversion attempt failed with %s: %s", filtered, exc)

        self.logger.warning("Falling back to full-document Docling conversion for %s", pdf_path.name)
        return self.converter.convert(str(pdf_path))

    def extract(self, pdf_path: Path, page_numbers: List[int]) -> List[LDU]:
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF path: {pdf_path}")

        selected_pages = set(page_numbers) if page_numbers else None
        density_map = self._page_density_map(pdf_path, selected_pages)

        try:
            conversion_result = self._convert_pdf(pdf_path, selected_pages)
        except Exception as exc:
            self.logger.exception(
                "Docling Strategy B failed for %s. Falling back to Strategy C Vision. Error: %s",
                pdf_path.name,
                exc,
            )
            vision_fallback = self._get_vision_fallback()
            return vision_fallback.extract(pdf_path=pdf_path, page_numbers=page_numbers)

        adapter = DoclingDocumentAdapter(conversion_result)

        units: List[LDU] = []
        for index, (page_no, piece) in enumerate(adapter.iter_pieces(selected_pages), start=1):
            content_type = adapter.piece_type(piece)

            if content_type == "table":
                content_markdown = adapter.piece_to_markdown(piece)
                content_raw = adapter.piece_to_json(piece)
            else:
                content_markdown = adapter.piece_to_markdown(piece)
                content_raw = adapter.piece_to_text(piece)

            page_density = density_map.get(page_no, 0.0)
            confidence = self.calculate_confidence(
                {
                    "parsed_ok": True,
                    "char_density": page_density,
                    "content_type": content_type,
                }
            )
            if confidence < self.warning_confidence_threshold:
                self.logger.warning(
                    "Low layout confidence %.3f on %s page %s (%s)",
                    confidence,
                    pdf_path.name,
                    page_no,
                    content_type,
                )

            units.append(
                LDU(
                    uid=f"{pdf_path.stem}_p{page_no}_{index}",
                    content_type=content_type,
                    content_raw=content_raw,
                    content_markdown=content_markdown,
                    provenance=ProvenanceChain(
                        source_file=pdf_path.name,
                        page_number=page_no,
                        strategy_used="STRATEGY_B",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                )
            )

        return units

    def calculate_confidence(self, page_data: Any) -> float:
        if isinstance(page_data, dict):
            parsed_ok = bool(page_data.get("parsed_ok", False))
            density = float(page_data.get("char_density", page_data.get("density", 0.0)) or 0.0)
            content_type = str(page_data.get("content_type", "text")).lower()
        else:
            parsed_ok = bool(getattr(page_data, "parsed_ok", False))
            density = float(
                getattr(page_data, "char_density", getattr(page_data, "density", 0.0)) or 0.0
            )
            content_type = str(getattr(page_data, "content_type", "text")).lower()

        confidence = 0.9 if parsed_ok else 0.5

        if density < self.min_digital_density:
            confidence -= 0.35

        if content_type == "table":
            confidence += 0.05

        return max(0.0, min(1.0, confidence))
