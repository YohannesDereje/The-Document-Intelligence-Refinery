from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz
from openai import OpenAI

from models import LDU, ProvenanceChain
from src.config import ConfigLoader

from .base_strategy import BaseExtractor


class BudgetExceededError(RuntimeError):
    pass


class ConfigurationError(RuntimeError):
    pass


class VisionExtractor(BaseExtractor):
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "openai/gpt-4o-mini",
        max_budget: float = 5.0,
        dpi: int = 140,
        rules_path: str | Path = "rubric/extraction_rules.yaml",
    ) -> None:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        resolved_key = api_key or openrouter_key or openai_key
        if not resolved_key:
            raise ConfigurationError("Missing API key. Provide api_key or set OPENROUTER_API_KEY.")

        self.config_loader = ConfigLoader(rules_path)
        self.provider_name = "openrouter"
        self.provider_base_url = "https://openrouter.ai/api/v1"

        if api_key and openai_key and api_key == openai_key:
            self.provider_name = "openai"
            self.provider_base_url = "https://api.openai.com/v1"
        elif api_key and openrouter_key and api_key == openrouter_key:
            self.provider_name = "openrouter"
            self.provider_base_url = "https://openrouter.ai/api/v1"
        elif not api_key and openai_key and not openrouter_key:
            self.provider_name = "openai"
            self.provider_base_url = "https://api.openai.com/v1"
        elif "/" not in model_name and (openai_key or (api_key and not openrouter_key)):
            self.provider_name = "openai"
            self.provider_base_url = "https://api.openai.com/v1"

        try:
            self.client = OpenAI(api_key=resolved_key, base_url=self.provider_base_url)
        except Exception as exc:
            raise ConfigurationError(f"Failed to initialize {self.provider_name} client: {exc}") from exc
        if self.provider_name == "openai" and "/" in model_name:
            self.model_name = model_name.split("/")[-1]
        else:
            self.model_name = model_name
        self.max_budget = max_budget
        self.total_spend = 0.0
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        self.last_call_cost_usd = 0.0
        self.last_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "image_tokens": 0,
            "completion_tokens": 0,
        }

        guards = dict(self.config_loader.get("economic_guards", {}))
        self.vlm_input_cost_per_1k_tokens = self._load_positive_float(
            guards, "vlm_input_cost_per_1k_tokens", 0.00030
        )
        self.vlm_image_cost_per_1k_tokens = self._load_positive_float(
            guards, "vlm_image_cost_per_1k_tokens", 0.00120
        )
        self.vlm_output_cost_per_1k_tokens = self._load_positive_float(
            guards, "vlm_output_cost_per_1k_tokens", 0.00060
        )
        self.vlm_image_tile_size = int(self._load_positive_float(guards, "vlm_image_tile_size", 512.0))
        self.vlm_tokens_per_image_tile = int(
            self._load_positive_float(guards, "vlm_tokens_per_image_tile", 85.0)
        )
        self.vlm_chars_per_token = self._load_positive_float(guards, "vlm_chars_per_token", 4.0)
        self.vlm_max_output_tokens = int(self._load_positive_float(guards, "vlm_max_output_tokens", 1200.0))

    def _switch_to_openai_fallback(self) -> bool:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return False

        try:
            self.client = OpenAI(api_key=openai_key, base_url="https://api.openai.com/v1")
            self.provider_name = "openai"
            self.provider_base_url = "https://api.openai.com/v1"
            if "/" in self.model_name:
                self.model_name = self.model_name.split("/")[-1]
            return True
        except Exception:
            return False

    def _load_positive_float(self, config: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = float(config.get(key, default))
            if value > 0.0:
                return value
        except (TypeError, ValueError):
            pass

        self.logger.warning(
            "Invalid economic guard '%s' in extraction_rules.yaml; using fallback %.6f",
            key,
            default,
        )
        return float(default)

    def _estimate_text_tokens(self, text: str) -> int:
        if not text:
            return 1
        return max(1, int(len(text) / max(self.vlm_chars_per_token, 1.0)))

    def _estimate_image_tokens(self, width_px: int, height_px: int) -> int:
        tile_size = max(1, self.vlm_image_tile_size)
        tile_count_x = max(1, (width_px + tile_size - 1) // tile_size)
        tile_count_y = max(1, (height_px + tile_size - 1) // tile_size)
        return tile_count_x * tile_count_y * max(1, self.vlm_tokens_per_image_tile)

    def calculate_final_cost(self, input_tokens: int, image_tokens: int, output_tokens: int) -> float:
        input_cost = (max(0, input_tokens) / 1000.0) * self.vlm_input_cost_per_1k_tokens
        image_cost = (max(0, image_tokens) / 1000.0) * self.vlm_image_cost_per_1k_tokens
        output_cost = (max(0, output_tokens) / 1000.0) * self.vlm_output_cost_per_1k_tokens
        return round(input_cost + image_cost + output_cost, 6)

    def _check_budget_or_raise(self, estimated_cost: float, page_number: int) -> None:
        remaining = self.max_budget - self.total_spend
        if estimated_cost > remaining:
            raise BudgetExceededError(
                f"Budget exceeded before page {page_number}. Estimated ${estimated_cost:.4f}, remaining ${remaining:.4f}."
            )

    def _render_page_png(self, pdf_path: Path, page_number: int) -> Tuple[bytes, int, int]:
        with fitz.open(pdf_path) as document:
            if page_number < 1 or page_number > document.page_count:
                raise ValueError(
                    f"Page {page_number} is out of range for {pdf_path.name} (1-{document.page_count})."
                )

            page = document.load_page(page_number - 1)
            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            return pix.tobytes("png"), int(pix.width), int(pix.height)

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

    def _call_vlm(
        self,
        image_bytes: bytes,
        page_number: int,
        width_px: int,
        height_px: int,
    ) -> Dict[str, Any]:
        messages = self._build_messages(image_bytes)
        prompt_text = json.dumps(messages, default=str)
        estimated_input_tokens = self._estimate_text_tokens(prompt_text)
        estimated_image_tokens = self._estimate_image_tokens(width_px, height_px)
        estimated_output_tokens = 256
        estimated_cost = self.calculate_final_cost(
            input_tokens=estimated_input_tokens,
            image_tokens=estimated_image_tokens,
            output_tokens=estimated_output_tokens,
        )
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
                max_tokens=self.vlm_max_output_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            error_text = str(exc).lower()
            if self.provider_name == "openrouter" and (
                "insufficient credits" in error_text or "error code: 402" in error_text
            ):
                if self._switch_to_openai_fallback():
                    self.logger.warning(
                        "OpenRouter credits exhausted; retrying Strategy C with OpenAI provider and model %s.",
                        self.model_name,
                    )
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=self.vlm_max_output_tokens,
                        response_format={"type": "json_object"},
                    )
                else:
                    self.logger.exception(
                        "Vision API connectivity/config failure on page %s with model %s: %s",
                        page_number,
                        self.model_name,
                        exc,
                    )
                    raise ConfigurationError(
                        f"Vision API cannot be reached for page {page_number} using model {self.model_name}: {exc}"
                    ) from exc
            else:
                self.logger.exception(
                    "Vision API connectivity/config failure on page %s with model %s: %s",
                    page_number,
                    self.model_name,
                    exc,
                )
                raise ConfigurationError(
                    f"Vision API cannot be reached for page {page_number} using model {self.model_name}: {exc}"
                ) from exc

        raw = response.choices[0].message.content or "{}"
        usage = getattr(response, "usage", None)

        input_tokens = estimated_input_tokens
        if usage is not None and getattr(usage, "prompt_tokens", None) is not None:
            input_tokens = max(1, int(getattr(usage, "prompt_tokens")))

        output_tokens = self._estimate_text_tokens(raw)
        if usage is not None and getattr(usage, "completion_tokens", None) is not None:
            output_tokens = max(1, int(getattr(usage, "completion_tokens")))

        image_tokens = estimated_image_tokens

        final_cost = self.calculate_final_cost(
            input_tokens=input_tokens,
            image_tokens=image_tokens,
            output_tokens=output_tokens,
        )

        self.total_spend += final_cost
        self.last_call_cost_usd = final_cost
        self.last_usage = {
            "prompt_tokens": max(0, int(input_tokens)),
            "image_tokens": max(0, int(image_tokens)),
            "completion_tokens": max(0, int(output_tokens)),
        }

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
            image_bytes, width_px, height_px = self._render_page_png(pdf_path, page_number)
            payload = self._call_vlm(
                image_bytes=image_bytes,
                page_number=page_number,
                width_px=width_px,
                height_px=height_px,
            )

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
