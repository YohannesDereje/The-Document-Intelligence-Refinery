from __future__ import annotations

from pathlib import Path

from src.agents.triage import TriageAgent


RULES_PATH = Path(__file__).resolve().parents[1] / "rubric" / "extraction_rules.yaml"


def _agent() -> TriageAgent:
    return TriageAgent(rules_path=RULES_PATH)


def test_detect_mixed_mode_true_from_sampled_pages() -> None:
    agent = _agent()

    metrics = [
        {"char_density": 0.005, "image_count": 0, "rect_count": 5},
        {"char_density": 0.0, "image_count": 2, "rect_count": 0},
        {"char_density": 0.004, "image_count": 1, "rect_count": 8},
        {"char_density": 0.0, "image_count": 1, "rect_count": 1},
    ]

    assert agent._detect_mixed_mode(metrics) is True


def test_detect_mixed_mode_false_for_uniform_digital() -> None:
    agent = _agent()

    metrics = [
        {"char_density": 0.006, "image_count": 0, "rect_count": 4},
        {"char_density": 0.007, "image_count": 0, "rect_count": 9},
        {"char_density": 0.004, "image_count": 0, "rect_count": 3},
    ]

    assert agent._detect_mixed_mode(metrics) is False


def test_cost_tier_high_for_scanned_or_high_complexity() -> None:
    agent = _agent()

    assert agent._map_cost_tier("scanned", "Low", "General") == "HIGH"
    assert agent._map_cost_tier("born_digital", "High", "General") == "HIGH"


def test_cost_tier_medium_and_low_paths() -> None:
    agent = _agent()

    assert agent._map_cost_tier("born_digital", "Medium", "General") == "MEDIUM"
    assert agent._map_cost_tier("born_digital", "Low", "Public_Sector") == "MEDIUM"
    assert agent._map_cost_tier("born_digital", "Low", "General") == "LOW"
