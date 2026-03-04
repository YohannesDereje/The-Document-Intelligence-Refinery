from __future__ import annotations

from pathlib import Path
from typing import Any

import pdfplumber
import yaml

from models import DocumentProfile, PageProfile


class TriageAgent:
    def __init__(self, rules_path: str | Path = "rubric/extraction_rules.yaml") -> None:
        self.rules_path = Path(rules_path)
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_path}")

        self.rules_config = self._load_rules(self.rules_path)
        self.thresholds: dict[str, Any] = self.rules_config.get("thresholds", {})
        self.routing_rules = self._load_and_sort_rules(self.rules_config)

    @staticmethod
    def _load_rules(rules_path: Path) -> dict[str, Any]:
        raw_yaml = rules_path.read_text(encoding="utf-8")
        normalized_yaml = raw_yaml.replace("\t", "  ")
        loaded = yaml.safe_load(normalized_yaml) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Rules YAML must load as a mapping/object.")
        return loaded

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

            normalized_rule = {
                "name": str(rule.get("name", f"rule_{index}")),
                "priority": int(rule.get("priority", index)),
                "condition": str(condition),
                "action": self._normalize_strategy(rule.get("action", "STRATEGY_C")),
                "reason": str(rule.get("reason") or rule.get("description") or "Matched routing rule."),
            }
            normalized_rules.append(normalized_rule)

        return sorted(normalized_rules, key=lambda rule: int(rule.get("priority", 9999)))

    @staticmethod
    def get_domain_hint(text: str) -> str:
        normalized = text.lower()

        banking_security_keywords = ["bank", "cbe", "disclosure", "vulnerability"]
        public_sector_keywords = ["ministry", "fta", "woreda", "survey"]

        if any(keyword in normalized for keyword in banking_security_keywords):
            return "Banking/Security"

        if any(keyword in normalized for keyword in public_sector_keywords):
            return "Public_Sector"

        return "General"

    def _evaluate_rule(self, condition: str, context: dict[str, Any]) -> bool:
        eval_context = {**self.thresholds, **context}
        try:
            return bool(eval(condition, {"__builtins__": {}}, eval_context))
        except Exception:
            return False

    def _infer_detected_origin(self, char_density: float) -> str:
        min_digital_density = float(self.thresholds.get("MIN_DIGITAL_DENSITY", 0.0005))
        if char_density > min_digital_density:
            return "born_digital"
        return "scanned"

    def _infer_layout_complexity(self, pages: list[PageProfile]) -> str:
        if not pages:
            return "Low"

        average_rects = sum(page.rect_count for page in pages) / len(pages)
        landscape_pages = sum(1 for page in pages if page.is_landscape)

        if average_rects > 50 or landscape_pages >= max(1, len(pages) // 3):
            return "High"

        if average_rects > 10:
            return "Medium"

        return "Low"

    def process_pdf(self, file_path: str | Path) -> DocumentProfile:
        pdf_path = Path(file_path)
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF file path: {pdf_path}")

        page_profiles: list[PageProfile] = []
        document_text_parts: list[str] = []
        self.last_page_reasons: dict[int, str] = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                document_text_parts.append(text)

                width = float(page.width or 0.0)
                height = float(page.height or 0.0)
                page_area = width * height

                char_count = len(text)
                char_density = (char_count / page_area) if page_area > 0 else 0.0
                rect_count = len(page.rects)
                is_landscape = width > height

                eval_context: dict[str, Any] = {
                    "char_density": char_density,
                    "rects": rect_count,
                    "rect_count": rect_count,
                    "width": width,
                    "height": height,
                    **self.thresholds,
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
                page_profile = PageProfile(
                    page_number=page_number,
                    char_density=char_density,
                    rect_count=rect_count,
                    is_landscape=is_landscape,
                    dominant_strategy=selected_action,
                    detected_origin=self._infer_detected_origin(char_density),
                )
                page_profiles.append(page_profile)

        full_text = "\n".join(document_text_parts)
        domain_hint = self.get_domain_hint(full_text)

        scanned_pages = sum(1 for page in page_profiles if page.detected_origin == "scanned")
        overall_origin = "scanned" if scanned_pages > (len(page_profiles) / 2) else "born_digital"

        document_profile = DocumentProfile(
            file_name=pdf_path.name,
            total_pages=len(page_profiles),
            overall_origin=overall_origin,
            domain_hint=domain_hint,
            layout_complexity=self._infer_layout_complexity(page_profiles),
            pages=page_profiles,
        )

        return document_profile


if __name__ == "__main__":
    agent = TriageAgent(rules_path="rubric/extraction_rules.yaml")
    sample_path = Path("data/fta_performance_survey_final_report_2022.pdf")
    profile = agent.process_pdf(sample_path)
    print(profile.model_dump_json(indent=2))
