from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber

from models import DocumentProfile, ExtractedDocument, ExtractionLedgerEntry, LDU
from src.config import ConfigLoader
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_extractor import LayoutExtractor
from src.strategies.vision_extractor import VisionExtractor


class ExtractionRouter:
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        confidence_threshold: float | None = None,
        max_budget: float | None = None,
        per_page_timeout_seconds: float | None = None,
        rules_path: str | Path = "rubric/extraction_rules.yaml",
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.config_loader = ConfigLoader(rules_path)

        self.vlm_budget_cap = float(self.config_loader.get("economic_guards.vlm_budget_cap", 30.0))
        self.default_dpi = int(self.config_loader.get("economic_guards.default_dpi", 140))
        self.cost_tier_to_model = dict(self.config_loader.get("strategy_mapping.cost_tier_to_model", {}))

        self.confidence_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else float(self.config_loader.get("thresholds.triage_confidence_gate", 0.70))
        )
        self.per_page_timeout_seconds = (
            float(per_page_timeout_seconds)
            if per_page_timeout_seconds is not None
            else float(self.config_loader.get("economic_guards.per_page_timeout_seconds", 60.0))
        )

        effective_max_budget = float(max_budget) if max_budget is not None else self.vlm_budget_cap
        default_model = str(self.cost_tier_to_model.get("HIGH", "openai/gpt-4o-mini"))
        effective_model = model_name or default_model

        self.fast_text_extractor = FastTextExtractor()
        self.layout_extractor = LayoutExtractor()
        self.vision_extractor = VisionExtractor(
            api_key=api_key,
            model_name=effective_model,
            max_budget=effective_max_budget,
            dpi=self.default_dpi,
        )

        self.strategy_order = ["STRATEGY_A", "STRATEGY_B", "STRATEGY_C"]
        self.strategy_map = {
            "STRATEGY_A": self.fast_text_extractor,
            "STRATEGY_B": self.layout_extractor,
            "STRATEGY_C": self.vision_extractor,
        }

        self.ledger_path = Path(".refinery") / "extraction_ledger.jsonl"
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def _normalize_strategy(self, strategy: str) -> str:
        normalized = strategy.strip().upper()
        if normalized in self.strategy_map:
            return normalized
        return "STRATEGY_C"

    def _strategy_chain(self, starting_strategy: str) -> List[str]:
        normalized = self._normalize_strategy(starting_strategy)
        start_index = self.strategy_order.index(normalized)
        return self.strategy_order[start_index:]

    def _append_ledger_entry(self, entry: ExtractionLedgerEntry) -> None:
        with self.ledger_path.open("a", encoding="utf-8") as ledger_file:
            ledger_file.write(entry.model_dump_json())
            ledger_file.write("\n")

    @staticmethod
    def _page_metrics(pdf_path: Path, page_number: int) -> Dict[str, Any]:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                raise ValueError(f"Page {page_number} out of range for {pdf_path.name}")

            page = pdf.pages[page_number - 1]
            text = page.extract_text() or ""
            char_count = len(text)
            width = float(page.width or 0.0)
            height = float(page.height or 0.0)
            area = width * height
            char_density = (char_count / area) if area > 0 else 0.0

            return {
                "char_count": char_count,
                "width": width,
                "height": height,
                "char_density": char_density,
                "chars": page.chars,
                "parsed_ok": True,
            }

    def _confidence_for_strategy(self, strategy: str, extractor: Any, page_data: Dict[str, Any], units: List[LDU]) -> float:
        if strategy == "STRATEGY_A":
            return float(extractor.calculate_confidence(page_data))

        if strategy == "STRATEGY_B":
            layout_data = {
                "parsed_ok": bool(units),
                "char_density": page_data.get("char_density", 0.0),
                "content_type": "table" if any(unit.content_type == "table" for unit in units) else "text",
            }
            return float(extractor.calculate_confidence(layout_data))

        uncertainty = not bool(units)
        return float(extractor.calculate_confidence({"uncertainty": uncertainty}))

    def _extract_page(self, extractor: Any, pdf_path: Path, page_number: int) -> List[LDU]:
        return extractor.extract(pdf_path=pdf_path, page_numbers=[page_number])

    @staticmethod
    def _is_vlm_model(value: str) -> bool:
        return "/" in value

    def _resolve_vision_model_for_profile(self, profile: DocumentProfile) -> str:
        tier = str(getattr(profile, "cost_tier", "HIGH") or "HIGH").upper()
        mapped = str(self.cost_tier_to_model.get(tier, self.vision_extractor.model_name))
        if self._is_vlm_model(mapped):
            return mapped
        return self.vision_extractor.model_name

    def _extract_page_with_timeout(self, extractor: Any, pdf_path: Path, page_number: int) -> List[LDU]:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._extract_page, extractor, pdf_path, page_number)
        try:
            return future.result(timeout=self.per_page_timeout_seconds)
        except TimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Strategy timed out after {self.per_page_timeout_seconds:.0f}s on page {page_number}."
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def process_document(self, pdf_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF path: {pdf_path}")

        self.vision_extractor.model_name = self._resolve_vision_model_for_profile(profile)

        all_units: List[LDU] = []
        escalations = 0

        for page_profile in sorted(profile.pages, key=lambda item: item.page_number):
            page_number = int(page_profile.page_number)
            page_metrics = self._page_metrics(pdf_path, page_number)

            strategy_chain = self._strategy_chain(page_profile.dominant_strategy)
            selected_units: List[LDU] = []
            selected_strategy = strategy_chain[-1]
            selected_confidence = 0.0
            selected_cost = 0.0
            page_had_success = False

            for strategy_name in strategy_chain:
                extractor = self.strategy_map[strategy_name]
                strategy_start = time.perf_counter()
                vision_spend_before = float(self.vision_extractor.total_spend)

                if strategy_name == "STRATEGY_C" and self.vision_extractor.total_spend >= self.vlm_budget_cap:
                    units = []
                    confidence = 0.0
                    self.logger.error(
                        "VLM stop-loss triggered at $%.4f (cap: $%.4f) on %s page %s. Skipping Strategy C.",
                        self.vision_extractor.total_spend,
                        self.vlm_budget_cap,
                        pdf_path.name,
                        page_number,
                    )
                else:
                    try:
                        units = self._extract_page_with_timeout(extractor, pdf_path, page_number)
                        confidence = self._confidence_for_strategy(strategy_name, extractor, page_metrics, units)
                    except TimeoutError:
                        units = []
                        confidence = 0.0
                        self.logger.warning(
                            "Timeout on %s page %s via %s after %.0fs. Escalating to next strategy.",
                            pdf_path.name,
                            page_number,
                            strategy_name,
                            self.per_page_timeout_seconds,
                        )
                    except Exception as exc:
                        units = []
                        confidence = 0.0
                        self.logger.exception(
                            "Strategy failure on %s page %s via %s. Escalating to next strategy. Error: %s",
                            pdf_path.name,
                            page_number,
                            strategy_name,
                            exc,
                        )

                processing_time_ms = (time.perf_counter() - strategy_start) * 1000.0
                vision_spend_after = float(self.vision_extractor.total_spend)
                estimated_cost_usd = max(0.0, vision_spend_after - vision_spend_before)

                ledger_entry = ExtractionLedgerEntry(
                    file_name=pdf_path.name,
                    strategy_selected=strategy_name,
                    confidence_score=confidence,
                    processing_time_ms=processing_time_ms,
                    estimated_cost_usd=estimated_cost_usd,
                )
                self._append_ledger_entry(ledger_entry)

                selected_units = units
                selected_strategy = strategy_name
                selected_confidence = confidence
                selected_cost = estimated_cost_usd
                if units:
                    page_had_success = True

                if confidence >= self.confidence_threshold or strategy_name == "STRATEGY_C":
                    break

                escalations += 1
                self.logger.warning(
                    "Low confidence %.3f for %s page %s via %s. Escalating to next strategy.",
                    confidence,
                    pdf_path.name,
                    page_number,
                    strategy_name,
                )

            self.logger.info(
                "Page %s finalized with %s (confidence=%.3f, units=%s, cost=$%.4f)",
                page_number,
                selected_strategy,
                selected_confidence,
                len(selected_units),
                selected_cost,
            )

            if not page_had_success:
                self.logger.error(
                    "All strategies failed for %s page %s. Continuing to next page.",
                    pdf_path.name,
                    page_number,
                )

            all_units.extend(selected_units)

        metadata = {
            "overall_origin": profile.overall_origin,
            "layout_complexity": profile.layout_complexity,
            "total_pages": profile.total_pages,
            "units_count": len(all_units),
            "escalations": escalations,
        }

        return ExtractedDocument(
            file_id=pdf_path.stem,
            domain_hint=profile.domain_hint,
            metadata=metadata,
            units=all_units,
        )
