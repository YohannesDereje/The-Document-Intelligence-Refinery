from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import pdfplumber
import yaml

from models import LDU, ProvenanceChain

from .base_strategy import BaseExtractor


class FastTextExtractor(BaseExtractor):
    def __init__(
        self,
        rules_path: Path | None = None,
        warning_confidence_threshold: float = 0.7,
    ) -> None:
        default_rules_path = Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
        self.rules_path = rules_path or default_rules_path
        self.warning_confidence_threshold = warning_confidence_threshold
        self.logger = logging.getLogger(__name__)
        self.min_digital_density = self._load_min_digital_density()

    def _load_min_digital_density(self) -> float:
        if not self.rules_path.exists():
            return 0.0005

        raw_yaml = self.rules_path.read_text(encoding="utf-8")
        normalized_yaml = raw_yaml.replace("\t", "  ")
        config = yaml.safe_load(normalized_yaml) or {}
        thresholds = config.get("thresholds", {}) if isinstance(config, dict) else {}
        return float(thresholds.get("MIN_DIGITAL_DENSITY", 0.0005))

    def _extract_text_blocks(self, page: Any) -> List[str]:
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
        if words:
            grouped_lines: dict[int, List[str]] = {}
            for word in words:
                line_key = int(round(float(word.get("top", 0.0))))
                grouped_lines.setdefault(line_key, []).append(str(word.get("text", "")).strip())

            blocks = [" ".join(tokens).strip() for _, tokens in sorted(grouped_lines.items())]
            return [block for block in blocks if block]

        fallback_text = page.extract_text() or ""
        fallback_blocks = [segment.strip() for segment in fallback_text.split("\n\n")]
        return [block for block in fallback_blocks if block]

    def extract(self, pdf_path: Path, page_numbers: List[int]) -> List[LDU]:
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF path: {pdf_path}")

        units: List[LDU] = []
        normalized_pages = sorted(set(page_numbers))

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_number in normalized_pages:
                if page_number < 1 or page_number > total_pages:
                    self.logger.warning(
                        "Skipping out-of-range page %s for file %s (total pages: %s)",
                        page_number,
                        pdf_path.name,
                        total_pages,
                    )
                    continue

                page = pdf.pages[page_number - 1]
                text = page.extract_text() or ""
                char_count = len(text)
                width = float(page.width or 0.0)
                height = float(page.height or 0.0)

                confidence = self.calculate_confidence(
                    {
                        "char_count": char_count,
                        "width": width,
                        "height": height,
                        "chars": page.chars,
                    }
                )
                if confidence < self.warning_confidence_threshold:
                    self.logger.warning(
                        "Low fast-text confidence %.3f on %s page %s",
                        confidence,
                        pdf_path.name,
                        page_number,
                    )

                blocks = self._extract_text_blocks(page)
                for block_index, block_text in enumerate(blocks, start=1):
                    uid = f"{pdf_path.stem}_p{page_number}_{block_index}"
                    provenance = ProvenanceChain(
                        source_file=pdf_path.name,
                        page_number=page_number,
                        strategy_used="STRATEGY_A",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    units.append(
                        LDU(
                            uid=uid,
                            content_type="text",
                            content_raw=block_text,
                            content_markdown=block_text,
                            provenance=provenance,
                        )
                    )

        return units

    def calculate_confidence(self, page_data: Any) -> float:
        if hasattr(page_data, "chars") and hasattr(page_data, "width") and hasattr(page_data, "height"):
            extracted_text = page_data.extract_text() or ""
            char_count = len(extracted_text)
            width = float(page_data.width or 0.0)
            height = float(page_data.height or 0.0)
            chars = page_data.chars or []
        elif isinstance(page_data, dict):
            char_count = int(page_data.get("char_count", 0))
            width = float(page_data.get("width", 0.0) or 0.0)
            height = float(page_data.get("height", 0.0) or 0.0)
            chars = page_data.get("chars", []) or []
        else:
            return 0.0

        area = width * height
        density = (char_count / area) if area > 0 else 0.0

        has_font_metadata = any(
            isinstance(char_item, dict) and (char_item.get("fontname") or char_item.get("size"))
            for char_item in chars
        )

        density_score = min(1.0, density / max(self.min_digital_density * 4.0, 1e-9))
        metadata_score = 1.0 if has_font_metadata else 0.35

        confidence = (0.75 * density_score) + (0.25 * metadata_score)
        if density < self.min_digital_density:
            confidence *= 0.35

        return max(0.0, min(1.0, confidence))
