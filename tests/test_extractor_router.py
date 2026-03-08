from __future__ import annotations

from types import SimpleNamespace

from src.agents.extractor import ExtractionRouter


def _router_for_unit_test() -> ExtractionRouter:
    router = ExtractionRouter.__new__(ExtractionRouter)
    router.strategy_order = ["STRATEGY_A", "STRATEGY_B", "STRATEGY_C"]
    router.strategy_map = {
        "STRATEGY_A": object(),
        "STRATEGY_B": object(),
        "STRATEGY_C": object(),
    }
    return router


def test_strategy_chain_for_hybrid_forces_a_b_c_order() -> None:
    router = _router_for_unit_test()

    chain = router._strategy_chain_for_page(origin_type="hybrid", dominant_strategy="STRATEGY_C")

    assert chain == ["STRATEGY_A", "STRATEGY_B", "STRATEGY_C"]


def test_strategy_chain_for_scanned_image_forces_b_then_c() -> None:
    router = _router_for_unit_test()

    chain = router._strategy_chain_for_page(origin_type="scanned_image", dominant_strategy="STRATEGY_A")

    assert chain == ["STRATEGY_B", "STRATEGY_C"]


def test_strategy_chain_after_a_starts_with_b_before_c() -> None:
    router = _router_for_unit_test()

    chain = router._strategy_chain_for_page(origin_type="born_digital", dominant_strategy="STRATEGY_A")

    assert chain == ["STRATEGY_A", "STRATEGY_B", "STRATEGY_C"]


def test_strategy_b_confidence_uses_ocr_engine_signal() -> None:
    router = _router_for_unit_test()

    class StubLayoutExtractor:
        def __init__(self) -> None:
            self.last_payload = None

        @staticmethod
        def _text_clarity_score(_: str) -> float:
            return 0.55

        @staticmethod
        def get_last_page_ocr_confidence(page_number: int) -> float:
            assert page_number == 2
            return 0.87

        def calculate_ocr_confidence(self, payload):
            self.last_payload = payload
            return float(payload["ocr_engine_confidence"])

    extractor = StubLayoutExtractor()
    units = [SimpleNamespace(content_markdown="scanned text", content_type="text")]

    score = router._confidence_for_strategy(
        "STRATEGY_B",
        extractor,
        page_data={"page_number": 2, "char_density": 0.0},
        units=units,
    )

    assert score == 0.87
    assert extractor.last_payload["ocr_engine_confidence"] == 0.87
