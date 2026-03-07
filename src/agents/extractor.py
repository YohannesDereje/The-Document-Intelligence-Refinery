from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pdfplumber

from models import (
    BBox,
    DocumentProfile,
    ExtractedDocument,
    ExtractionAttempt,
    ExtractionLedgerEntry,
    ExtractionResult,
    LDU,
)
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
        self.rules_path = Path(rules_path)
        self.config_loader = ConfigLoader(self.rules_path)

        self.vlm_budget_cap = float(self.config_loader.get("economic_guards.vlm_budget_cap", 30.0))
        self.default_dpi = int(self.config_loader.get("economic_guards.default_dpi", 140))
        self.cost_tier_to_model = dict(self.config_loader.get("strategy_mapping.cost_tier_to_model", {}))

        self.confidence_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else float(self.config_loader.get("thresholds.triage_confidence_gate", 0.70))
        )
        self.strategy_b_escalation_confidence = float(
            self.config_loader.get("thresholds.strategy_b_escalation_confidence", 0.75)
        )
        self.per_page_timeout_seconds = (
            float(per_page_timeout_seconds)
            if per_page_timeout_seconds is not None
            else float(self.config_loader.get("economic_guards.per_page_timeout_seconds", 60.0))
        )

        effective_max_budget = float(max_budget) if max_budget is not None else self.vlm_budget_cap
        default_model = str(self.cost_tier_to_model.get("HIGH", "openai/gpt-4o-mini"))
        effective_model = model_name or default_model

        self.fast_text_extractor = FastTextExtractor(rules_path=self.rules_path)
        self.layout_extractor = LayoutExtractor(rules_path=self.rules_path)
        self.vision_extractor = VisionExtractor(
            api_key=api_key,
            model_name=effective_model,
            max_budget=effective_max_budget,
            dpi=self.default_dpi,
            rules_path=self.rules_path,
        )

        self.strategy_order = ["STRATEGY_A", "STRATEGY_B", "STRATEGY_C"]
        self.strategy_map = {
            "STRATEGY_A": self.fast_text_extractor,
            "STRATEGY_B": self.layout_extractor,
            "STRATEGY_C": self.vision_extractor,
        }

        self.ledger_path = Path(".refinery") / "extraction_ledger.jsonl"
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.total_cost_usd = 0.0

    def _normalize_strategy(self, strategy: str) -> str:
        normalized = strategy.strip().upper()
        if normalized in self.strategy_map:
            return normalized
        return "STRATEGY_C"

    def _strategy_chain(self, starting_strategy: str) -> List[str]:
        normalized = self._normalize_strategy(starting_strategy)
        start_index = self.strategy_order.index(normalized)
        return self.strategy_order[start_index:]

    @staticmethod
    def _origin_type_from_profile(profile: DocumentProfile) -> str:
        profile_data = profile.model_dump()
        origin_candidate = profile_data.get("origin_type")
        if not origin_candidate:
            origin_candidate = profile_data.get("overall_origin") or profile_data.get("document_type")

        normalized = str(origin_candidate or "unknown").strip().lower()
        if normalized in {"scanned", "scanned_image", "image_scan"}:
            return "scanned_image"
        if normalized in {"born_digital", "digital"}:
            return "born_digital"
        if normalized in {"hybrid", "mixed"}:
            return "hybrid"
        return "unknown"

    @staticmethod
    def _strategy_label(strategy_name: str) -> str:
        return strategy_name.replace("STRATEGY_", "Strategy ")

    @staticmethod
    def _is_vlm_model(value: str) -> bool:
        return "/" in value

    def _resolve_vision_model_for_profile(self, profile: DocumentProfile) -> str:
        tier = str(getattr(profile, "cost_tier", "HIGH") or "HIGH").upper()
        mapped = str(self.cost_tier_to_model.get(tier, self.vision_extractor.model_name))
        if self._is_vlm_model(mapped):
            return mapped
        return self.vision_extractor.model_name

    def _append_ledger_entry(self, entry: ExtractionLedgerEntry) -> None:
        if entry.attempt_chain:
            cumulative_cost = self.calculate_final_cost(entry.attempt_chain)
            cumulative_time = self._cumulative_processing_time(entry.attempt_chain)
            final_success_cost = self._final_success_cost(entry.attempt_chain)
            final_vlm_cost = self._final_vlm_cost(entry.attempt_chain)
            entry = entry.model_copy(
                update={
                    "estimated_cost_usd": cumulative_cost,
                    "processing_time_ms": cumulative_time,
                    "final_success_cost_usd": final_success_cost,
                    "final_vlm_cost_usd": final_vlm_cost,
                }
            )

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

    def _confidence_for_strategy(
        self,
        strategy: str,
        extractor: Any,
        page_data: Dict[str, Any],
        units: List[LDU],
    ) -> float:
        if strategy == "STRATEGY_A":
            return float(extractor.calculate_confidence(page_data))

        if strategy == "STRATEGY_B":
            joined_text = "\n".join(unit.content_markdown for unit in units if unit.content_type in {"text", "header", "table"})
            layout_data = {
                "parsed_ok": bool(units),
                "char_density": page_data.get("char_density", 0.0),
                "table_count": sum(1 for unit in units if unit.content_type == "table"),
                "text_count": sum(1 for unit in units if unit.content_type in {"text", "header"}),
                "text_clarity": extractor._text_clarity_score(joined_text) if hasattr(extractor, "_text_clarity_score") else 0.0,
            }
            return float(extractor.calculate_ocr_confidence(layout_data))

        uncertainty = not bool(units)
        return float(extractor.calculate_confidence({"uncertainty": uncertainty}))

    def _extract_page(self, extractor: Any, pdf_path: Path, page_number: int) -> List[LDU]:
        return extractor.extract(pdf_path=pdf_path, page_numbers=[page_number])

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

    @staticmethod
    def _path_string(origin_type: str, attempts: Sequence[ExtractionAttempt]) -> str:
        chunks: list[str] = [origin_type]
        for attempt in attempts:
            status_label = "Success" if attempt.status == "success" else "Fail"
            chunks.append(f"{attempt.strategy.value.replace('STRATEGY_', 'Strategy ')} ({status_label})")
        return " -> ".join(chunks)

    @staticmethod
    def calculate_final_cost(attempts: Sequence[ExtractionAttempt]) -> float:
        return round(sum(max(0.0, float(attempt.estimated_cost_usd)) for attempt in attempts), 6)

    @staticmethod
    def _cumulative_processing_time(attempts: Sequence[ExtractionAttempt]) -> float:
        return round(sum(max(0.0, float(attempt.processing_time_ms)) for attempt in attempts), 3)

    @staticmethod
    def _final_success_cost(attempts: Sequence[ExtractionAttempt]) -> float:
        for attempt in reversed(attempts):
            if attempt.status == "success":
                return round(max(0.0, float(attempt.estimated_cost_usd)), 6)
        return 0.0

    @staticmethod
    def _final_vlm_cost(attempts: Sequence[ExtractionAttempt]) -> float:
        for attempt in reversed(attempts):
            if attempt.status == "success" and attempt.strategy.value == "STRATEGY_C":
                return round(max(0.0, float(attempt.estimated_cost_usd)), 6)
        return 0.0

    def _required_confidence(self, strategy_name: str, origin_type: str) -> float:
        if strategy_name == "STRATEGY_B" and origin_type == "scanned_image":
            return self.strategy_b_escalation_confidence
        if strategy_name == "STRATEGY_C":
            return 0.0
        return self.confidence_threshold

    @staticmethod
    def _normalize_unit(unit: LDU, page_number: int, strategy_name: str, path: list[str]) -> LDU:
        bbox = unit.bbox if unit.bbox is not None else BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        normalized_provenance = unit.provenance.model_copy(
            update={
                "page_number": page_number,
                "strategy_used": strategy_name,
                "strategy_escalation_path": path,
            }
        )
        return unit.model_copy(update={"bbox": bbox, "provenance": normalized_provenance})

    def process_document(self, pdf_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF path: {pdf_path}")

        self.vision_extractor.model_name = self._resolve_vision_model_for_profile(profile)

        all_units: List[LDU] = []
        page_results: List[ExtractionResult] = []
        escalations = 0
        origin_type = self._origin_type_from_profile(profile)

        for page_profile in sorted(profile.pages, key=lambda item: item.page_number):
            page_number = int(page_profile.page_number)
            page_metrics = self._page_metrics(pdf_path, page_number)

            if origin_type == "scanned_image":
                strategy_chain = ["STRATEGY_B", "STRATEGY_C"]
            else:
                strategy_chain = self._strategy_chain(page_profile.dominant_strategy)

            selected_units: List[LDU] = []
            selected_strategy = strategy_chain[-1]
            selected_confidence = 0.0
            attempt_chain: List[ExtractionAttempt] = []

            for strategy_name in strategy_chain:
                extractor = self.strategy_map[strategy_name]
                strategy_start = time.perf_counter()
                vision_spend_before = float(self.vision_extractor.total_spend)
                prompt_tokens = 0
                image_tokens = 0
                completion_tokens = 0

                if strategy_name == "STRATEGY_C":
                    self.logger.info("Directing to Strategy C (VLM) for page %s", page_number)

                if strategy_name == "STRATEGY_C" and self.vision_extractor.total_spend >= self.vlm_budget_cap:
                    units: List[LDU] = []
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
                        if strategy_name == "STRATEGY_C" and hasattr(self.vision_extractor, "last_usage"):
                            usage = dict(getattr(self.vision_extractor, "last_usage", {}))
                            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                            image_tokens = int(usage.get("image_tokens", 0) or 0)
                            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
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

                required_confidence = self._required_confidence(strategy_name, origin_type)
                status = "success" if units and confidence >= required_confidence else "fail"

                attempt = ExtractionAttempt(
                    strategy=strategy_name,
                    status=status,
                    confidence_score=confidence,
                    processing_time_ms=processing_time_ms,
                    estimated_cost_usd=estimated_cost_usd,
                    prompt_tokens=prompt_tokens,
                    image_tokens=image_tokens,
                    completion_tokens=completion_tokens,
                )
                attempt_chain.append(attempt)

                selected_units = units
                selected_strategy = strategy_name
                selected_confidence = confidence

                if status == "success" or strategy_name == "STRATEGY_C":
                    break

                escalations += 1
                self.logger.warning(
                    "Low confidence %.3f for %s page %s via %s. Escalating to next strategy.",
                    confidence,
                    pdf_path.name,
                    page_number,
                    strategy_name,
                )

            strategy_path = [attempt.strategy.value for attempt in attempt_chain]
            normalized_units = [
                self._normalize_unit(unit, page_number=page_number, strategy_name=selected_strategy, path=strategy_path)
                for unit in selected_units
            ]

            all_units.extend(normalized_units)

            final_status = "success" if normalized_units else "fail"
            path_string = self._path_string(origin_type, attempt_chain)
            cumulative_cost_usd = self.calculate_final_cost(attempt_chain)
            cumulative_processing_time_ms = self._cumulative_processing_time(attempt_chain)
            final_success_cost_usd = self._final_success_cost(attempt_chain)
            final_vlm_cost_usd = self._final_vlm_cost(attempt_chain)

            page_result = ExtractionResult(
                page_number=page_number,
                origin_type=origin_type,
                attempts=attempt_chain,
                final_strategy=selected_strategy,
                final_status=final_status,
                final_confidence_score=selected_confidence,
                escalation_path=path_string,
                cumulative_processing_time_ms=cumulative_processing_time_ms,
                cumulative_cost_usd=cumulative_cost_usd,
                total_cost_usd=cumulative_cost_usd,
                final_success_cost_usd=final_success_cost_usd,
                final_vlm_cost_usd=final_vlm_cost_usd,
            )
            page_results.append(page_result)
            self.total_cost_usd += cumulative_cost_usd

            ledger_entry = ExtractionLedgerEntry(
                file_name=pdf_path.name,
                page_number=page_number,
                origin_type=origin_type,
                strategy_selected=selected_strategy,
                confidence_score=selected_confidence,
                processing_time_ms=cumulative_processing_time_ms,
                estimated_cost_usd=cumulative_cost_usd,
                final_success_cost_usd=final_success_cost_usd,
                final_vlm_cost_usd=final_vlm_cost_usd,
                attempt_chain=attempt_chain,
                escalation_path=path_string,
            )
            self._append_ledger_entry(ledger_entry)

            self.logger.info(
                "Page %s finalized with %s (confidence=%.3f, units=%s, path=%s)",
                page_number,
                selected_strategy,
                selected_confidence,
                len(normalized_units),
                path_string,
            )

            if not normalized_units:
                self.logger.error(
                    "All strategies failed for %s page %s. Continuing to next page.",
                    pdf_path.name,
                    page_number,
                )

        metadata = {
            "overall_origin": profile.overall_origin,
            "layout_complexity": profile.layout_complexity,
            "total_pages": profile.total_pages,
            "units_count": len(all_units),
            "escalations": escalations,
            "extraction_results": [result.model_dump() for result in page_results],
        }

        return ExtractedDocument(
            file_id=pdf_path.stem,
            domain_hint=profile.domain_hint,
            metadata=metadata,
            units=all_units,
        )
