from __future__ import annotations

from typing import Any

import pytest

from models import BBox, SemanticChunk
from src.agents import qa_agent
from src.agents.qa_agent import QAGraphAgent, _chunks_from_vector_store, run_qa
from src.utils.hashing import generate_content_hash


def _chunk(content: str, section: str, page: int, content_hash: str | None = None) -> SemanticChunk:
    bbox = BBox(x1=0.1, y1=0.1, x2=0.9, y2=0.2)
    resolved_hash = content_hash or generate_content_hash(text=content, page_number=page, bbox=bbox)
    return SemanticChunk(
        content=content,
        page_numbers=[page],
        bbox_bounds=bbox,
        section_context=section,
        token_count=max(1, len(content.split())),
        content_hash=resolved_hash,
    )


class _FakePageIndexBuilder:
    def __init__(self) -> None:
        self.root = None

    def traverse_query(self, query: str) -> list[SemanticChunk]:
        _ = query
        return [_chunk("Section summary evidence text.", "Finance > Forecasting", 2)]

    def build_tree(self, chunks: list[SemanticChunk], persist: bool = True) -> Any:
        _ = chunks
        _ = persist
        self.root = None
        return None


class _FakeVectorStore:
    def __init__(self) -> None:
        self.collection = self

    def semantic_search(self, query: str, top_k: int = 5) -> list[SemanticChunk]:
        _ = query
        _ = top_k
        return [_chunk("Revenue forecast improved.", "Finance > Forecasting", 3)]

    def get(self, include: list[str]) -> dict[str, list[Any]]:
        _ = include
        content = "Revenue forecast improved."
        bbox = BBox(x1=0.1, y1=0.1, x2=0.9, y2=0.2)
        content_hash = generate_content_hash(text=content, page_number=3, bbox=bbox)
        return {
            "documents": [content],
            "metadatas": [
                {
                    "content_hash": content_hash,
                    "page_numbers": "3",
                    "section_context": "Finance > Forecasting",
                    "bbox_x1": 0.1,
                    "bbox_y1": 0.1,
                    "bbox_x2": 0.9,
                    "bbox_y2": 0.2,
                    "token_count": 3,
                }
            ],
        }


@pytest.mark.skipif(qa_agent.StateGraph is None or qa_agent.tool is None, reason="langgraph/langchain-core missing")
def test_qa_agent_fallback_answer_includes_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QAGraphAgent, "_build_llm_client", staticmethod(lambda: None))

    agent = QAGraphAgent(page_index_builder=_FakePageIndexBuilder(), vector_store=_FakeVectorStore())
    response = agent.ask("What changed in revenue forecast?")
    expected_hash = generate_content_hash(
        text="Revenue forecast improved.",
        page_number=3,
        bbox=BBox(x1=0.1, y1=0.1, x2=0.9, y2=0.2),
    )

    assert f"content_hash={expected_hash}" in response["answer"]
    assert "page=3" in response["answer"]
    assert response["provenance_chain"]
    assert response["provenance_chain"][0]["content_hash"] == expected_hash
    assert response["provenance_chain"][0]["page_number"] == 3


def test_chunks_from_vector_store_reconstructs_content_hash() -> None:
    chunks = _chunks_from_vector_store(_FakeVectorStore())
    expected_hash = generate_content_hash(
        text="Revenue forecast improved.",
        page_number=3,
        bbox=BBox(x1=0.1, y1=0.1, x2=0.9, y2=0.2),
    )

    assert len(chunks) == 1
    assert chunks[0].content_hash == expected_hash
    assert chunks[0].page_numbers == [3]


def test_run_qa_integration_with_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubVectorStore(_FakeVectorStore):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = args
            _ = kwargs
            super().__init__()

    class _StubPageIndexBuilder(_FakePageIndexBuilder):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = args
            _ = kwargs
            super().__init__()

    class _StubQAGraphAgent:
        def __init__(self, page_index_builder: Any, vector_store: Any) -> None:
            _ = page_index_builder
            _ = vector_store

        def ask(self, query: str) -> dict[str, Any]:
            return {"answer": f"ok:{query}", "provenance_chain": []}

    monkeypatch.setattr(qa_agent, "VectorStore", _StubVectorStore)
    monkeypatch.setattr(qa_agent, "PageIndexBuilder", _StubPageIndexBuilder)
    monkeypatch.setattr(qa_agent, "QAGraphAgent", _StubQAGraphAgent)

    result = run_qa("Where is the forecast evidence?")
    assert result == {"answer": "ok:Where is the forecast evidence?", "provenance_chain": []}
