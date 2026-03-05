from __future__ import annotations

from pathlib import Path

from models import PageProfile
from src.agents.triage import TriageAgent


RULES_PATH = Path(__file__).resolve().parents[1] / "rubric" / "extraction_rules.yaml"


def _agent() -> TriageAgent:
    return TriageAgent(rules_path=RULES_PATH)


def test_get_domain_hint_keywords() -> None:
    agent = _agent()

    assert agent.get_domain_hint("CBE internal security notice") == "Banking/Security"
    assert agent.get_domain_hint("Bank disclosure update") == "Banking/Security"
    assert agent.get_domain_hint("Ministry report for FY2025") == "Public_Sector"
    assert agent.get_domain_hint("Woreda operational survey") == "Public_Sector"


def test_infer_detected_origin_by_density() -> None:
    agent = _agent()

    assert agent._infer_detected_origin(0.05) == "born_digital"
    assert agent._infer_detected_origin(0.0001) == "scanned"


def test_infer_layout_complexity_high_rect_count() -> None:
    agent = _agent()
    pages = [
        PageProfile(
            page_number=1,
            char_density=0.003,
            rect_count=120,
            is_landscape=False,
            dominant_strategy="STRATEGY_B",
            detected_origin="born_digital",
        )
    ]

    assert agent._infer_layout_complexity(pages) == "High"
