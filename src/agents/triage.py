from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, List

import pdfplumber

from models import DocumentProfile, PageProfile
from src.config import ConfigLoader


class BaseDomainStrategy(ABC):
    @abstractmethod
    def infer(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        pass


class KeywordDomainStrategy(BaseDomainStrategy):
    def __init__(self, registry: list[dict[str, Any]]) -> None:
        self.registry = registry

    def infer(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        corpus = f"{text} {' '.join(str(v) for v in (metadata or {}).values())}".lower()
        for entry in self.registry:
            domain_hint = str(entry.get("domain_hint", "General"))
            keywords = [str(keyword).lower() for keyword in entry.get("domain_keywords", [])]
            if any(keyword in corpus for keyword in keywords):
                return domain_hint
        return "General"


class MetadataDomainStrategy(BaseDomainStrategy):
    def __init__(self, registry: list[dict[str, Any]]) -> None:
        self.registry = registry

    def infer(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        if not metadata:
            return "General"

        metadata_blob = " ".join(str(value) for value in metadata.values()).lower()
        for entry in self.registry:
            domain_hint = str(entry.get("domain_hint", "General"))
            keywords = [str(keyword).lower() for keyword in entry.get("domain_keywords", [])]
            if any(keyword in metadata_blob for keyword in keywords):
                return domain_hint
        return "General"


class TriageAgent:
    def __init__(
        self,
        rules_path: str | Path = "rubric/extraction_rules.yaml",
        domain_strategies: Iterable[BaseDomainStrategy] | None = None,
    ) -> None:
        self.rules_path = Path(rules_path)
        self.config_loader = ConfigLoader(self.rules_path)
        self.rules_config = self.config_loader.config

        self.routing_rules = self._load_and_sort_rules(self.rules_config)

        self.min_digital_density = float(self.config_loader.get("thresholds.char_density.min_digital", 0.0005))
        self.table_rect_threshold = int(self.config_loader.get("thresholds.rect_count.table_high", 50))
        self.complexity_medium_rect_threshold = int(
            self.config_loader.get("thresholds.rect_count.medium_complexity", 10)
        )
        self.triage_confidence_gate = float(self.config_loader.get("thresholds.triage_confidence_gate", 0.70))
        self.mixed_mode_sample_size = int(self.config_loader.get("thresholds.mixed_mode_sample_size", 6))
        self.contradictory_image_ratio_threshold = float(
            self.config_loader.get("thresholds.contradictory_image_ratio_threshold", 0.30)
        )

        self.domain_registry: list[dict[str, Any]] = list(self.config_loader.get("domain_registry", []))
        self.cost_tier_to_model: dict[str, str] = dict(
            self.config_loader.get("strategy_mapping.cost_tier_to_model", {})
        )

        self.domain_strategies = list(
            domain_strategies
            or [
                KeywordDomainStrategy(self.domain_registry),
                MetadataDomainStrategy(self.domain_registry),
            ]
        )

    @staticmethod
    def _build_condition_from_when(when: dict[str, Any]) -> str:
        metric = str(when.get("metric", "False"))
        operator = str(when.get("operator", "eq")).lower()
        value = when.get("value")

        if metric == "page_width_gt_height":
            return "width > height"

        operator_map = {
            "eq": "==",
            "ne": "!=",
            "gt": ">",
            "gte": ">=",
            "lt": "<",
            "lte": "<=",
        }
        expression_operator = operator_map.get(operator, "==")
        return f"{metric} {expression_operator} {repr(value)}"

    @staticmethod
    def _normalize_strategy(action_value: Any) -> str:
        if isinstance(action_value, dict):
            strategy_value = str(action_value.get("strategy", "C"))
        else:
            strategy_value = str(action_value)

        normalized = strategy_value.strip().upper()
        if normalized in {"A", "B", "C"}:
            return f"STRATEGY_{normalized}"
        if normalized.startswith("STRATEGY_"):
            return normalized
        return "STRATEGY_C"

    def _load_and_sort_rules(self, rules_config: dict[str, Any]) -> list[dict[str, Any]]:
        raw_rules = rules_config.get("routing_rules") or rules_config.get("rules") or []
        if not isinstance(raw_rules, list):
            return []

        normalized_rules: list[dict[str, Any]] = []
        for index, rule in enumerate(raw_rules, start=1):
            if not isinstance(rule, dict):
                continue

            condition = rule.get("condition")
            if not condition:
                when_block = rule.get("when", {})
                if isinstance(when_block, dict):
                    condition = self._build_condition_from_when(when_block)
                else:
                    condition = "False"

            normalized_rules.append(
                {
                    "name": str(rule.get("name", f"rule_{index}")),
                    "priority": int(rule.get("priority", index)),
                    "condition": str(condition),
                    "action": self._normalize_strategy(rule.get("action", "STRATEGY_C")),
                    "reason": str(rule.get("reason") or rule.get("description") or "Matched routing rule."),
                }
            )

        return sorted(normalized_rules, key=lambda rule: int(rule.get("priority", 9999)))

    def _evaluate_rule(self, condition: str, context: dict[str, Any]) -> bool:
        eval_context = {
            **context,
            "MIN_DIGITAL_DENSITY": self.min_digital_density,
            "TABLE_RECT_THRESHOLD": self.table_rect_threshold,
            "TRIAGE_CONFIDENCE_GATE": self.triage_confidence_gate,
        }
        try:
            return bool(eval(condition, {"__builtins__": {}}, eval_context))
        except Exception:
            return False

    def _infer_detected_origin(self, char_density: float) -> str:
        if char_density > self.min_digital_density:
            return "born_digital"
        return "scanned"

    def _infer_layout_complexity(self, pages: list[PageProfile]) -> str:
        if not pages:
            return "Low"

        average_rects = sum(page.rect_count for page in pages) / len(pages)
        landscape_pages = sum(1 for page in pages if page.is_landscape)

        if average_rects >= self.table_rect_threshold or landscape_pages >= max(1, len(pages) // 3):
            return "High"
        if average_rects >= self.complexity_medium_rect_threshold:
            return "Medium"
        return "Low"

    def _sample_page_indices(self, total_pages: int) -> List[int]:
        if total_pages <= 0:
            return []
        if total_pages <= self.mixed_mode_sample_size:
            return list(range(total_pages))

        step = max(1, total_pages // self.mixed_mode_sample_size)
        indices = list(range(0, total_pages, step))[: self.mixed_mode_sample_size]
        if (total_pages - 1) not in indices:
            indices[-1] = total_pages - 1
        return sorted(set(indices))

    def _detect_mixed_mode(self, page_metrics: List[dict[str, Any]]) -> bool:
        if not page_metrics:
            return False

        sampled = [page_metrics[index] for index in self._sample_page_indices(len(page_metrics))]
        has_digital = any(metric["char_density"] > self.min_digital_density for metric in sampled)
        has_scan_like = any(
            metric["char_density"] <= self.min_digital_density and metric["image_count"] > 0 for metric in sampled
        )
        return has_digital and has_scan_like

    @staticmethod
    def _detect_form_fillable(pdf: Any) -> bool:
        try:
            document = getattr(pdf, "doc", None)
            if document is None:
                return False
            catalog = getattr(document, "catalog", {}) or {}
            acro_form = catalog.get("AcroForm")
            if not acro_form:
                return False
            if "XFA" in str(acro_form):
                return True
            fields = acro_form.get("Fields") if isinstance(acro_form, dict) else None
            return bool(fields)
        except Exception:
            return False

    def get_domain_hint(self, text: str) -> str:
        return self._resolve_domain_hint(text=text, metadata={})

    def _resolve_domain_hint(self, text: str, metadata: dict[str, Any]) -> str:
        for strategy in self.domain_strategies:
            hint = strategy.infer(text=text, metadata=metadata)
            if hint != "General":
                return hint
        return "General"

    @staticmethod
    def _promote_cost_tier(tier: str) -> str:
        if tier == "LOW":
            return "MEDIUM"
        if tier == "MEDIUM":
            return "HIGH"
        return "HIGH"

    def _map_cost_tier(self, origin_type: str, layout_complexity: str, domain_hint: str, triage_confidence: float | None = None) -> str:
        if origin_type == "scanned" or layout_complexity == "High":
            base_tier = "HIGH"
        elif domain_hint in {"Banking/Security", "Public_Sector"} or layout_complexity == "Medium":
            base_tier = "MEDIUM"
        else:
            base_tier = "LOW"

        if triage_confidence is not None and triage_confidence < self.triage_confidence_gate:
            base_tier = self._promote_cost_tier(base_tier)

        return base_tier

    def _compute_triage_confidence(
        self,
        overall_origin: str,
        mixed_mode: bool,
        contradiction_ratio: float,
        scanned_ratio: float,
        form_fillable: bool,
    ) -> float:
        confidence = 0.82

        if overall_origin == "scanned" and scanned_ratio >= 0.8:
            confidence = 0.94
        if mixed_mode:
            confidence -= 0.18
        if contradiction_ratio >= self.contradictory_image_ratio_threshold:
            confidence -= 0.15
        if form_fillable:
            confidence -= 0.05

        return max(0.05, min(0.99, confidence))

    def process_pdf(self, file_path: str | Path) -> DocumentProfile:
        pdf_path = Path(file_path)
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF file path: {pdf_path}")

        page_profiles: list[PageProfile] = []
        page_metrics: list[dict[str, Any]] = []
        document_text_parts: list[str] = []
        self.last_page_reasons: dict[int, str] = {}

        with pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata or {}
            form_fillable = self._detect_form_fillable(pdf)

            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                document_text_parts.append(text)

                width = float(page.width or 0.0)
                height = float(page.height or 0.0)
                page_area = width * height
                char_count = len(text)
                char_density = (char_count / page_area) if page_area > 0 else 0.0
                rect_count = len(page.rects)
                image_count = len(page.images)
                is_landscape = width > height

                eval_context: dict[str, Any] = {
                    "char_density": char_density,
                    "rects": rect_count,
                    "rect_count": rect_count,
                    "width": width,
                    "height": height,
                }

                selected_action = "STRATEGY_C"
                selected_reason = "No explicit rule matched; defaulting to OCR/Vision routing."
                for rule in self.routing_rules:
                    condition = str(rule.get("condition", "False"))
                    if self._evaluate_rule(condition, eval_context):
                        selected_action = str(rule.get("action", "STRATEGY_C"))
                        selected_reason = str(rule.get("reason", "Matched routing rule."))
                        break

                self.last_page_reasons[page_number] = selected_reason
                page_profiles.append(
                    PageProfile(
                        page_number=page_number,
                        char_density=char_density,
                        rect_count=rect_count,
                        is_landscape=is_landscape,
                        dominant_strategy=selected_action,
                        detected_origin=self._infer_detected_origin(char_density),
                    )
                )
                page_metrics.append(
                    {
                        "char_density": char_density,
                        "image_count": image_count,
                        "rect_count": rect_count,
                    }
                )

        full_text = "\n".join(document_text_parts)
        domain_hint = self._resolve_domain_hint(full_text, metadata)

        scanned_pages = sum(1 for page in page_profiles if page.detected_origin == "scanned")
        scanned_ratio = scanned_pages / max(1, len(page_profiles))
        overall_origin = "scanned" if scanned_pages > (len(page_profiles) / 2) else "born_digital"

        layout_complexity = self._infer_layout_complexity(page_profiles)
        mixed_mode = self._detect_mixed_mode(page_metrics)
        contradictory_pages = sum(
            1
            for metric in page_metrics
            if metric["char_density"] > self.min_digital_density and metric["image_count"] > 0
        )
        contradiction_ratio = contradictory_pages / max(1, len(page_metrics))

        triage_confidence = self._compute_triage_confidence(
            overall_origin=overall_origin,
            mixed_mode=mixed_mode,
            contradiction_ratio=contradiction_ratio,
            scanned_ratio=scanned_ratio,
            form_fillable=form_fillable,
        )
        cost_tier = self._map_cost_tier(
            origin_type=overall_origin,
            layout_complexity=layout_complexity,
            domain_hint=domain_hint,
            triage_confidence=triage_confidence,
        )

        normalized_origin_type = "scanned_image" if overall_origin == "scanned" else overall_origin
        if mixed_mode:
            normalized_origin_type = "hybrid"

        return DocumentProfile(
            file_name=pdf_path.name,
            total_pages=len(page_profiles),
            overall_origin=overall_origin,
            origin_type=normalized_origin_type,
            domain_hint=domain_hint,
            layout_complexity=layout_complexity,
            pages=page_profiles,
            document_type="hybrid" if mixed_mode else overall_origin,
            triage_confidence=triage_confidence,
            cost_tier=cost_tier,
            mixed_mode=mixed_mode,
            form_fillable=form_fillable,
        )


__all__ = [
    "BaseDomainStrategy",
    "KeywordDomainStrategy",
    "MetadataDomainStrategy",
    "TriageAgent",
]
