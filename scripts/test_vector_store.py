from __future__ import annotations

from models import BBox, SemanticChunk
from src.agents.vector_store import VectorStore


def _chunk(content: str, section: str, page: int, content_hash: str) -> SemanticChunk:
    return SemanticChunk(
        content=content,
        page_numbers=[page],
        bbox_bounds=BBox(x1=0.1, y1=0.1, x2=0.9, y2=0.2),
        section_context=section,
        token_count=max(1, len(content.split())),
        content_hash=content_hash,
    )


def main() -> None:
    store = VectorStore(persist_directory=".refinery/vector_db", collection_name="semantic_chunks_test")

    chunks = [
        _chunk(
            content="Revenue forecast variance decreased after weekly finance reviews.",
            section="Finance > Forecasting",
            page=2,
            content_hash="hash-finance-001",
        ),
        _chunk(
            content="Support ticket backlog reduced after workflow automation in operations.",
            section="Operations > Support",
            page=5,
            content_hash="hash-ops-001",
        ),
    ]

    indexed = store.populate_store(chunks, reset=True)
    print(f"Indexed chunks: {indexed}")

    hits = store.semantic_search("finance revenue forecast", top_k=1)
    if not hits:
        raise RuntimeError("No vector search hits were returned.")

    print("Top hit content_hash:", hits[0].content_hash)
    assert hits[0].content_hash == "hash-finance-001", "Unexpected content_hash returned from search"
    print("Vector store smoke test passed.")


if __name__ == "__main__":
    main()
