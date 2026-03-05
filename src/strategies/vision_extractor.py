from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import fitz
from openai import OpenAI

from models import LDU, ProvenanceChain

from .base_strategy import BaseExtractor


class BudgetExceededError(RuntimeError):
    pass


class VisionExtractor(BaseExtractor):
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "openai/gpt-4o-mini",
        max_budget: float = 5.0,
        dpi: int = 140,
    ) -> None:
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("Missing API key. Provide api_key or set OPENROUTER_API_KEY.")

        self.client = OpenAI(api_key=resolved_key, base_url="https://openrouter.ai/api/v1")
        self.model_name = model_name
        self.max_budget = max_budget
        self.total_spend = 0.0
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)

    def _estimate_call_cost(self, image_bytes: bytes, prompt_text: str) -> float:
        image_mb = len(image_bytes) / (1024 * 1024)
        prompt_kchars = len(prompt_text) / 1000
        estimated = (0.010 * image_mb) + (0.0015 * prompt_kchars) + 0.003
        return max(0.003, round(estimated, 6))

    def _check_budget_or_raise(self, estimated_cost: float, page_number: int) -> None:
        remaining = self.max_budget - self.total_spend
        if estimated_cost > remaining:
            raise BudgetExceededError(
                f"Budget exceeded before page {page_number}. Estimated ${estimated_cost:.4f}, remaining ${remaining:.4f}."
            )

    def _render_page_png(self, pdf_path: Path, page_number: int) -> bytes:
        with fitz.open(pdf_path) as document:
            if page_number < 1 or page_number > document.page_count:
                raise ValueError(
                    f"Page {page_number} is out of range for {pdf_path.name} (1-{document.page_count})."
                )

            page = document.load_page(page_number - 1)
            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            return pix.tobytes("png")

    def _build_messages(self, image_bytes: bytes) -> list[dict[str, Any]]:
        system_prompt = (
            "You are a document intelligence extraction engine. "
            "Return strict JSON only."
        )

        user_prompt = (
            "Extract all text, tables, and headers. For tables, provide them in valid GitHub-Flavored Markdown. "
            "Respond in JSON with this schema: "
            "{\"sections\": [{\"content_type\": \"text|table|header|figure\", "
            "\"content_raw\": \"string or object\", \"content_markdown\": \"string\", "
            "\"uncertainty\": false}], \"uncertainty\": false}"
        )

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ]

    @staticmethod
    def _extract_json_text(raw_content: str) -> str:
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end >= start:
            return cleaned[start : end + 1]
        return cleaned

    def _call_vlm(self, image_bytes: bytes, page_number: int) -> Dict[str, Any]:
        messages = self._build_messages(image_bytes)
        prompt_text = json.dumps(messages, default=str)
        estimated_cost = self._estimate_call_cost(image_bytes, prompt_text)
        self._check_budget_or_raise(estimated_cost, page_number)

        self.logger.info(
            "Vision call estimate for page %s: $%.4f (spent: $%.4f / $%.2f)",
            page_number,
            estimated_cost,
            self.total_spend,
            self.max_budget,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            self.logger.exception(
                "Vision API call failed on page %s with model %s: %s",
                page_number,
                self.model_name,
                exc,
            )
            return {
                "sections": [
                    {
                        "content_type": "text",
                        "content_raw": f"Vision API error: {exc}",
                        "content_markdown": "",
                        "uncertainty": True,
                    }
                ],
                "uncertainty": True,
            }

        self.total_spend += estimated_cost
        raw = response.choices[0].message.content or "{}"
        json_payload = self._extract_json_text(raw)

        try:
            return json.loads(json_payload)
        except json.JSONDecodeError:
            return {
                "sections": [
                    {
                        "content_type": "text",
                        "content_raw": raw,
                        "content_markdown": raw,
                        "uncertainty": True,
                    }
                ],
                "uncertainty": True,
            }

    def extract(self, pdf_path: Path, page_numbers: List[int]) -> List[LDU]:
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF path: {pdf_path}")

        units: List[LDU] = []
        normalized_pages = sorted(set(page_numbers))

        for page_number in normalized_pages:
            image_bytes = self._render_page_png(pdf_path, page_number)
            payload = self._call_vlm(image_bytes=image_bytes, page_number=page_number)

            page_uncertainty = bool(payload.get("uncertainty", False))
            confidence = self.calculate_confidence({"uncertainty": page_uncertainty})

            sections = payload.get("sections", [])
            if not isinstance(sections, list):
                sections = []

            for index, section in enumerate(sections, start=1):
                if not isinstance(section, dict):
                    continue

                content_type = str(section.get("content_type", "text"))
                content_raw_value = section.get("content_raw", "")
                content_markdown = str(section.get("content_markdown") or "")

                if isinstance(content_raw_value, (dict, list)):
                    content_raw = json.dumps(content_raw_value, ensure_ascii=False)
                else:
                    content_raw = str(content_raw_value)

                if not content_markdown:
                    content_markdown = content_raw

                if content_type.lower() == "table" and not content_markdown.strip():
                    content_markdown = content_raw

                if confidence < 0.95:
                    self.logger.warning(
                        "Vision extraction uncertainty on %s page %s (confidence %.2f)",
                        pdf_path.name,
                        page_number,
                        confidence,
                    )

                units.append(
                    LDU(
                        uid=f"{pdf_path.stem}_p{page_number}_{index}",
                        content_type=content_type,
                        content_raw=content_raw,
                        content_markdown=content_markdown,
                        provenance=ProvenanceChain(
                            source_file=pdf_path.name,
                            page_number=page_number,
                            strategy_used="STRATEGY_C",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                )

        return units

    def calculate_confidence(self, page_data: Any) -> float:
        uncertainty = False
        if isinstance(page_data, dict):
            uncertainty = bool(page_data.get("uncertainty", False))
        else:
            uncertainty = bool(getattr(page_data, "uncertainty", False))

        return 0.7 if uncertainty else 0.95
