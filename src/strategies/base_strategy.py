from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

from models import LDU, ProvenanceChain


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, pdf_path: Path, page_numbers: List[int]) -> List[LDU]:
        pass

    @abstractmethod
    def calculate_confidence(self, page_data: Any) -> float:
        pass
