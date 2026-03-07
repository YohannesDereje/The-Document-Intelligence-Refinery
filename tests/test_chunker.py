from __future__ import annotations

from pathlib import Path

from models import BBox, ExtractedDocument, LDU, ProvenanceChain, StrategyName
from src.agents.chunker import ChunkingEngine


def _unit(
    uid: str,
    content_type: str,
    markdown: str,
    page: int,
    x1: float = 0.1,
    y1: float = 0.1,
    x2: float = 0.9,
    y2: float = 0.2,
) -> LDU:
    return LDU(
        uid=uid,
        content_type=content_type,
        content_raw=markdown,
        content_markdown=markdown,
    provenance=ProvenanceChain(
      source_file="doc-table.pdf",
      page_number=page,
      strategy_used=StrategyName.STRATEGY_B,
      timestamp="2026-01-01T00:00:00Z",
      strategy_escalation_path=[StrategyName.STRATEGY_B],
    ),
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _rules_file(tmp_path: Path) -> Path:
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """
chunking_constitution:
  max_chunk_size: 30
  min_chunk_size: 5
  preserve_structure: true
  semantic_boundary_indicators:
    - "introduction"
    - "conclusion"
  force_split_on:
    - "header"
    - "table"
  never_split_on:
    - "table"
  overlap_tokens: 0
  header_detection:
    min_font_size: 12
    bold_weight_threshold: 600
""".strip(),
        encoding="utf-8",
    )
    return rules


def test_table_blocks_are_not_split(tmp_path: Path) -> None:
    rules_path = _rules_file(tmp_path)
    engine = ChunkingEngine(rules_path=rules_path)

    long_table = " | ".join([f"col{i}" for i in range(1, 60)])

    units = [
        _unit("u1", "header", "Introduction", page=1),
        _unit("u2", "text", "This opening paragraph provides context for the table below.", page=1),
        _unit("u3", "table", f"Table 1\n{long_table}", page=1, y1=0.3, y2=0.6),
        _unit("u4", "table", "Continuation row A | value A", page=1, y1=0.61, y2=0.7),
        _unit("u5", "text", "Final remarks after the table.", page=1, y1=0.72, y2=0.8),
    ]

    document = ExtractedDocument(
      file_id="doc-table",
      domain_hint="unknown",
      metadata={"source": "unit-test"},
      units=units,
    )
    chunks = engine.chunk_document(document)

    table_chunks = [chunk for chunk in chunks if "Table 1" in chunk.content or "Continuation row A" in chunk.content]

    assert len(table_chunks) == 1
    assert "Table 1" in table_chunks[0].content
    assert "Continuation row A" in table_chunks[0].content
    assert table_chunks[0].token_count > engine.max_chunk_size
