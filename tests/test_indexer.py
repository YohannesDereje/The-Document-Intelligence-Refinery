from __future__ import annotations

from pathlib import Path

from models import BBox, SemanticChunk
from src.agents.indexer import PageIndexBuilder


def _chunk(content: str, section_context: str, page: int, x1: float = 0.1) -> SemanticChunk:
    return SemanticChunk(
        content=content,
        page_numbers=[page],
        bbox_bounds=BBox(x1=x1, y1=0.1, x2=0.9, y2=0.2),
        section_context=section_context,
        token_count=max(1, len(content.split())),
    )


def test_index_tree_build_and_query(tmp_path: Path) -> None:
    output_file = tmp_path / "page_index.json"

    builder = PageIndexBuilder(
        summary_model="openai/gpt-4o-mini",
        top_k_sections=3,
        persistence_path=output_file,
    )

    chunks = [
        _chunk(
            "Revenue forecasting improved after introducing weekly variance tracking for finance KPIs.",
            "Finance > Forecasting",
            page=1,
            x1=0.1,
        ),
        _chunk(
            "Customer churn interventions reduced ticket backlog in support workflows.",
            "Operations > Support",
            page=2,
            x1=0.2,
        ),
        _chunk(
            "Cash-flow planning now includes scenario stress tests for treasury reviews.",
            "Finance > Treasury",
            page=3,
            x1=0.3,
        ),
    ]

    root = builder.build_tree(chunks=chunks, persist=True)

    assert root.node_type == "root"
    assert output_file.exists()

    selected = builder.traverse_query("finance forecast and cash flow")
    assert selected
    assert all("Finance" in chunk.section_context for chunk in selected)
