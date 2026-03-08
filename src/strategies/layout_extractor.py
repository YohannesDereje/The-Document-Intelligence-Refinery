from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pdfplumber

from models import BBox, LDU, ProvenanceChain
from src.config import ConfigLoader

from .base_strategy import BaseExtractor
from .vision_extractor import VisionExtractor

try:
    from docling.document_converter import DocumentConverter
except Exception:  # pragma: no cover
    DocumentConverter = None

try:
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption
except Exception:  # pragma: no cover
    PyPdfiumDocumentBackend = None
    InputFormat = None
    PdfFormatOption = None

try:
    from docling.datamodel.pipeline_options import PdfPipelineOptions
except Exception:  # pragma: no cover
    PdfPipelineOptions = None

try:
    from docling.datamodel.pipeline_options import TesseractCliOcrOptions
except Exception:  # pragma: no cover
    TesseractCliOcrOptions = None

try:
    from docling.datamodel.pipeline_options import TesseractOcrOptions
except Exception:  # pragma: no cover
    TesseractOcrOptions = None

pytesseract = None
try:
    import pytesseract as _pytesseract  # type: ignore[reportMissingImports]

    pytesseract = _pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


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
                    raw_number = int(getattr(piece, attr))
                    return raw_number if raw_number >= 1 else raw_number + 1
                except (TypeError, ValueError):
                    pass

        provenance = getattr(piece, "provenance", None)
        if provenance is not None:
            for attr in ("page_no", "page_number", "page", "page_idx", "page_index"):
                if hasattr(provenance, attr):
                    try:
                        raw_number = int(getattr(provenance, attr))
                        return raw_number if raw_number >= 1 else raw_number + 1
                    except (TypeError, ValueError):
                        pass

        return 1

    @staticmethod
    def piece_type(piece: Any) -> str:
        raw_type = str(getattr(piece, "type", "") or getattr(piece, "label", "")).lower()
        class_name = piece.__class__.__name__.lower()
        if "table" in raw_type or "table" in class_name:
            return "table"
        if "header" in raw_type or "title" in raw_type:
            return "header"
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
            payload = {key: value for key, value in vars(piece).items() if not key.startswith("_")}
        else:
            payload = str(piece)

        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def piece_bbox(piece: Any, page_width: float, page_height: float) -> BBox:
        bbox_value = None
        for attr in ("bbox", "box", "rect"):
            if hasattr(piece, attr):
                bbox_value = getattr(piece, attr)
                if bbox_value is not None:
                    break

        coords: Optional[Tuple[float, float, float, float]] = None
        if isinstance(bbox_value, (tuple, list)) and len(bbox_value) >= 4:
            coords = (
                float(bbox_value[0]),
                float(bbox_value[1]),
                float(bbox_value[2]),
                float(bbox_value[3]),
            )
        elif bbox_value is not None:
            x1 = getattr(bbox_value, "x0", getattr(bbox_value, "left", None))
            y1 = getattr(bbox_value, "y0", getattr(bbox_value, "top", None))
            x2 = getattr(bbox_value, "x1", getattr(bbox_value, "right", None))
            y2 = getattr(bbox_value, "y1", getattr(bbox_value, "bottom", None))
            if None not in (x1, y1, x2, y2):
                coords = (float(x1), float(y1), float(x2), float(y2))

        if coords is None:
            return BBox(x1=0.0, y1=0.0, x2=max(page_width, 1.0), y2=max(page_height, 1.0))

        return BBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])


class LayoutExtractor(BaseExtractor):
    def __init__(
        self,
        rules_path: Path | None = None,
        warning_confidence_threshold: float = 0.7,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.warning_confidence_threshold = warning_confidence_threshold

        default_rules_path = Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
        self.rules_path = rules_path or default_rules_path
        self.config_loader = ConfigLoader(self.rules_path)

        self.min_digital_density = float(
            self.config_loader.get("thresholds.char_density.min_digital", 0.0005)
        )
        self.ocr_min_text_clarity = float(
            self.config_loader.get("thresholds.ocr_min_text_clarity", 0.35)
        )
        self.ocr_policy: Dict[str, Any] = dict(self.config_loader.get("ocr_policy", {}))
        self.ocr_execution_mode = str(self.ocr_policy.get("execution_mode", "auto")).strip().lower()
        if self.ocr_execution_mode not in {"auto", "docling", "pytesseract"}:
            self.ocr_execution_mode = "auto"
        self.tesseract_cmd = self._resolve_tesseract_cmd()
        self.tesseract_runtime_ok = self._verify_tesseract_runtime(self.tesseract_cmd)
        self.last_page_ocr_confidence: Dict[int, float] = {}

        self.converter = self._build_converter()
        self.vision_fallback: VisionExtractor | None = None

    def _build_converter(self) -> Any:
        if DocumentConverter is None:
            raise RuntimeError("Docling is not available. Install docling to use LayoutExtractor.")

        if PyPdfiumDocumentBackend is None or InputFormat is None or PdfFormatOption is None:
            self.logger.warning(
                "Docling backend extension unavailable; using default converter without explicit backend config."
            )
            return DocumentConverter()

        format_option_kwargs: Dict[str, Any] = {"backend": PyPdfiumDocumentBackend}
        pipeline_options = self._build_ocr_pipeline_options()
        if pipeline_options is not None:
            format_option_kwargs["pipeline_options"] = pipeline_options

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(**format_option_kwargs),
            }
        )

        self.logger.info(
            "LayoutExtractor initialized with OCR-enabled Docling backend (engine=%s, tesseract_ok=%s).",
            str(self.ocr_policy.get("engine", "tesseract")),
            self.tesseract_runtime_ok,
        )
        return converter

    def _build_ocr_pipeline_options(self) -> Any:
        if not bool(self.ocr_policy.get("force_enable", True)):
            return None

        if PdfPipelineOptions is None:
            self.logger.warning("PdfPipelineOptions not available; cannot explicitly enable OCR flags.")
            return None

        options = PdfPipelineOptions()

        for attr in ("do_ocr", "ocr_enabled", "enable_ocr"):
            if hasattr(options, attr):
                setattr(options, attr, True)

        ocr_option = self._build_tesseract_ocr_options()
        if ocr_option is not None and hasattr(options, "ocr_options"):
            setattr(options, "ocr_options", ocr_option)

        return options

    def _build_tesseract_ocr_options(self) -> Any:
        ocr_languages = self.ocr_policy.get("languages", ["eng"])
        if not isinstance(ocr_languages, list):
            ocr_languages = ["eng"]

        tesseract_cmd = self.tesseract_cmd

        for option_class in (TesseractCliOcrOptions, TesseractOcrOptions):
            if option_class is None:
                continue
            try:
                option = option_class()
                for attr in ("lang", "languages", "language"):
                    if hasattr(option, attr):
                        setattr(option, attr, ocr_languages)
                for attr in ("tesseract_cmd", "cmd", "command"):
                    if tesseract_cmd and hasattr(option, attr):
                        setattr(option, attr, tesseract_cmd)
                return option
            except Exception:
                continue

        return None

    def _resolve_tesseract_cmd(self) -> str:
        override = str(os.getenv("TESSERACT_CMD", "")).strip()
        if override and Path(override).exists():
            return override

        discovered = shutil.which("tesseract")
        if discovered:
            return discovered

        common_paths = [
            Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
            Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
        ]
        for candidate in common_paths:
            if candidate.exists():
                return str(candidate)
        return "tesseract"

    def _verify_tesseract_runtime(self, tesseract_cmd: str) -> bool:
        if not tesseract_cmd:
            self.logger.warning("Tesseract command is empty; OCR confidence may be degraded.")
            return False

        command_exists = Path(tesseract_cmd).exists() if (":" in tesseract_cmd or "/" in tesseract_cmd or "\\" in tesseract_cmd) else bool(shutil.which(tesseract_cmd))
        if not command_exists:
            self.logger.warning("Tesseract binary was not found at runtime: %s", tesseract_cmd)
            return False

        try:
            completed = subprocess.run(
                [tesseract_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if completed.returncode != 0:
                self.logger.warning(
                    "Tesseract binary check failed (code=%s): %s",
                    completed.returncode,
                    (completed.stderr or completed.stdout or "").strip(),
                )
                return False
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Unable to execute tesseract binary '%s': %s", tesseract_cmd, exc)
            return False

        if pytesseract is not None:
            try:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            except Exception:
                pass
        else:
            self.logger.warning("pytesseract is not installed; Strategy B confidence falls back to text-clarity signals.")

        return True

    @staticmethod
    def _mean_tesseract_confidence(data: Dict[str, Any]) -> float | None:
        confidences = data.get("conf", []) if isinstance(data, dict) else []
        parsed: list[float] = []
        for value in confidences:
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if score >= 0.0:
                parsed.append(score)

        if not parsed:
            return None

        return max(0.0, min(1.0, (sum(parsed) / len(parsed)) / 100.0))

    def _compute_page_ocr_confidence_map(self, pdf_path: Path, page_numbers: set[int] | None) -> Dict[int, float]:
        if pytesseract is None or not self.tesseract_runtime_ok:
            return {}

        page_confidence: Dict[int, float] = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                if page_numbers is not None and page_number not in page_numbers:
                    continue

                try:
                    page_image = page.to_image(resolution=220)
                    pil_image = page_image.original
                    data = pytesseract.image_to_data(
                        pil_image,
                        output_type=pytesseract.Output.DICT,
                        lang="+".join(self.ocr_policy.get("languages", ["eng"])),
                    )
                    confidence = self._mean_tesseract_confidence(data)
                    if confidence is not None:
                        page_confidence[page_number] = confidence
                except Exception as exc:
                    self.logger.warning(
                        "pytesseract OCR confidence probe failed on %s page %s: %s",
                        pdf_path.name,
                        page_number,
                        exc,
                    )

        return page_confidence

    def _extract_with_pytesseract(self, pdf_path: Path, page_numbers: set[int] | None) -> List[LDU]:
        if pytesseract is None or not self.tesseract_runtime_ok:
            raise RuntimeError("pytesseract runtime is unavailable.")

        ocr_languages = self.ocr_policy.get("languages", ["eng"])
        if not isinstance(ocr_languages, list):
            ocr_languages = ["eng"]
        lang = "+".join(str(item) for item in ocr_languages if str(item).strip()) or "eng"

        units: List[LDU] = []
        self.last_page_ocr_confidence = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                if page_numbers is not None and page_number not in page_numbers:
                    continue

                try:
                    page_image = page.to_image(resolution=220)
                    pil_image = page_image.original

                    data = pytesseract.image_to_data(
                        pil_image,
                        output_type=pytesseract.Output.DICT,
                        lang=lang,
                    )
                    confidence = self._mean_tesseract_confidence(data)
                    if confidence is not None:
                        self.last_page_ocr_confidence[page_number] = confidence

                    raw_text = pytesseract.image_to_string(pil_image, lang=lang) or ""
                    text = raw_text.strip()
                    if not text:
                        continue

                    width = float(page.width or 1.0)
                    height = float(page.height or 1.0)
                    units.append(
                        LDU(
                            uid=f"{pdf_path.stem}_p{page_number}_ocr_1",
                            content_type="text",
                            content_raw=text,
                            content_markdown=text,
                            bbox=BBox(x1=0.0, y1=0.0, x2=max(width, 1.0), y2=max(height, 1.0)),
                            provenance=ProvenanceChain(
                                source_file=pdf_path.name,
                                page_number=page_number,
                                strategy_used="STRATEGY_B",
                                timestamp=datetime.now(timezone.utc).isoformat(),
                            ),
                        )
                    )
                except Exception as exc:
                    self.logger.warning(
                        "pytesseract OCR extraction failed on %s page %s: %s",
                        pdf_path.name,
                        page_number,
                        exc,
                    )

        return units

    def _get_vision_fallback(self) -> VisionExtractor:
        if self.vision_fallback is None:
            self.vision_fallback = VisionExtractor(
                api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
                model_name="openai/gpt-4o-mini",
            )
        return self.vision_fallback

    @staticmethod
    def _page_geometry_map(pdf_path: Path, page_numbers: set[int] | None) -> dict[int, tuple[float, float]]:
        geometry: dict[int, tuple[float, float]] = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                if page_numbers is not None and page_number not in page_numbers:
                    continue
                geometry[page_number] = (float(page.width or 0.0), float(page.height or 0.0))
        return geometry

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
            filtered = {key: value for key, value in kwargs.items() if key in supported_params}
            if not filtered:
                continue
            try:
                return self.converter.convert(str(pdf_path), **filtered)
            except Exception as exc:
                self.logger.warning("Docling conversion attempt failed with %s: %s", filtered, exc)

        self.logger.warning("Falling back to full-document Docling conversion for %s", pdf_path.name)
        return self.converter.convert(str(pdf_path))

    @staticmethod
    def _text_clarity_score(text: str) -> float:
        if not text:
            return 0.0

        visible_chars = [char for char in text if not char.isspace()]
        if not visible_chars:
            return 0.0

        alnum_ratio = sum(1 for char in visible_chars if char.isalnum()) / len(visible_chars)
        word_lengths = [len(word) for word in text.split() if word]
        avg_word_len = (sum(word_lengths) / len(word_lengths)) if word_lengths else 0.0
        word_score = min(1.0, avg_word_len / 5.0)

        return max(0.0, min(1.0, (0.7 * alnum_ratio) + (0.3 * word_score)))

    def calculate_ocr_confidence(self, page_data: Dict[str, Any]) -> float:
        parsed_ok = bool(page_data.get("parsed_ok", False))
        raw_text_clarity = page_data.get("text_clarity")
        text_clarity = float(raw_text_clarity) if raw_text_clarity is not None else (1.0 if parsed_ok else 0.0)
        raw_engine_confidence = page_data.get("ocr_engine_confidence")
        engine_confidence = float(raw_engine_confidence) if raw_engine_confidence is not None else None
        table_count = int(page_data.get("table_count", 0) or 0)
        default_text_count = 1 if parsed_ok else 0
        text_count = int(page_data.get("text_count", default_text_count) or 0)
        page_density = float(page_data.get("char_density", 0.0) or 0.0)
        min_digital_density = float(getattr(self, "min_digital_density", 0.0005))
        min_text_clarity = float(getattr(self, "ocr_min_text_clarity", 0.25))

        table_signal = 1.0 if table_count > 0 else (0.65 if text_count > 0 else 0.2)
        density_signal = min(1.0, page_density / max(min_digital_density * 1.5, 1e-9))

        if engine_confidence is not None:
            confidence = (0.5 * engine_confidence) + (0.3 * text_clarity) + (0.15 * table_signal) + (0.05 * density_signal)
        else:
            confidence = (0.6 * text_clarity) + (0.25 * table_signal) + (0.15 * density_signal)
        if not parsed_ok:
            confidence *= 0.5
        if text_clarity < min_text_clarity:
            confidence *= 0.7

        return max(0.0, min(1.0, confidence))

    def extract(self, pdf_path: Path, page_numbers: List[int]) -> List[LDU]:
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF path: {pdf_path}")

        selected_pages = set(page_numbers) if page_numbers else None
        density_map = self._page_density_map(pdf_path, selected_pages)
        geometry_map = self._page_geometry_map(pdf_path, selected_pages)

        if self.ocr_execution_mode in {"auto", "pytesseract"}:
            try:
                pytesseract_units = self._extract_with_pytesseract(pdf_path, selected_pages)
                if pytesseract_units:
                    return pytesseract_units
                if self.ocr_execution_mode == "pytesseract":
                    return []
            except Exception as exc:
                if self.ocr_execution_mode == "pytesseract":
                    self.logger.exception("pytesseract-only Strategy B failed for %s: %s", pdf_path.name, exc)
                    return []
                self.logger.warning(
                    "pytesseract pre-pass failed for %s; falling back to docling backend: %s",
                    pdf_path.name,
                    exc,
                )

        self.last_page_ocr_confidence = self._compute_page_ocr_confidence_map(pdf_path, selected_pages)

        try:
            conversion_result = self._convert_pdf(pdf_path, selected_pages)
        except Exception as exc:
            self.logger.exception(
                "Docling Strategy B failed for %s. Returning failure to router for explicit escalation. Error: %s",
                pdf_path.name,
                exc,
            )
            raise RuntimeError(
                f"LayoutExtractor OCR backend failed for {pdf_path.name}; router should escalate to Strategy C."
            ) from exc

        adapter = DoclingDocumentAdapter(conversion_result)

        units: List[LDU] = []
        page_table_counts: dict[int, int] = {}
        page_text_counts: dict[int, int] = {}
        page_text_blobs: dict[int, list[str]] = {}

        for index, (page_no, piece) in enumerate(adapter.iter_pieces(selected_pages), start=1):
            content_type = adapter.piece_type(piece)

            if content_type == "table":
                content_markdown = adapter.piece_to_markdown(piece)
                content_raw = adapter.piece_to_json(piece)
            else:
                content_markdown = adapter.piece_to_markdown(piece)
                content_raw = adapter.piece_to_text(piece)

            content_markdown = content_markdown or content_raw
            if not content_markdown.strip():
                continue

            width, height = geometry_map.get(page_no, (1.0, 1.0))
            bbox = adapter.piece_bbox(piece, page_width=width, page_height=height)

            page_table_counts[page_no] = page_table_counts.get(page_no, 0) + (1 if content_type == "table" else 0)
            page_text_counts[page_no] = page_text_counts.get(page_no, 0) + (1 if content_type in {"text", "header"} else 0)
            page_text_blobs.setdefault(page_no, []).append(content_markdown)

            units.append(
                LDU(
                    uid=f"{pdf_path.stem}_p{page_no}_{index}",
                    content_type=content_type,
                    content_raw=content_raw,
                    content_markdown=content_markdown,
                    bbox=bbox,
                    provenance=ProvenanceChain(
                        source_file=pdf_path.name,
                        page_number=page_no,
                        strategy_used="STRATEGY_B",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                )
            )

        for page_no, text_chunks in page_text_blobs.items():
            page_density = density_map.get(page_no, 0.0)
            clarity = self._text_clarity_score("\n".join(text_chunks))
            ocr_confidence = self.calculate_ocr_confidence(
                {
                    "parsed_ok": bool(text_chunks),
                    "char_density": page_density,
                    "table_count": page_table_counts.get(page_no, 0),
                    "text_count": page_text_counts.get(page_no, 0),
                    "text_clarity": clarity,
                    "ocr_engine_confidence": self.last_page_ocr_confidence.get(page_no),
                }
            )
            if ocr_confidence < self.warning_confidence_threshold:
                self.logger.warning(
                    "Low OCR confidence %.3f on %s page %s",
                    ocr_confidence,
                    pdf_path.name,
                    page_no,
                )

        return units

    def calculate_confidence(self, page_data: Any) -> float:
        if isinstance(page_data, dict):
            return self.calculate_ocr_confidence(page_data)
        parsed_ok = bool(getattr(page_data, "parsed_ok", False))
        return self.calculate_ocr_confidence({"parsed_ok": parsed_ok})

    def get_last_page_ocr_confidence(self, page_number: int) -> float | None:
        return self.last_page_ocr_confidence.get(page_number)
