from __future__ import annotations

from src.strategies.layout_extractor import LayoutExtractor


def _layout_extractor_for_unit_test() -> LayoutExtractor:
    extractor = LayoutExtractor.__new__(LayoutExtractor)
    extractor.min_digital_density = 0.0005
    return extractor


def test_layout_confidence_high_density_parsed_ok() -> None:
    extractor = _layout_extractor_for_unit_test()

    confidence = extractor.calculate_confidence(
        {
            "parsed_ok": True,
            "char_density": 0.02,
            "content_type": "text",
        }
    )

    assert confidence > 0.8


def test_layout_confidence_units_empty_below_threshold() -> None:
    extractor = _layout_extractor_for_unit_test()

    # Emulates router behavior when units == [] -> parsed_ok = False
    confidence = extractor.calculate_confidence(
        {
            "parsed_ok": False,
            "char_density": 0.02,
            "content_type": "text",
        }
    )

    assert confidence < 0.65
