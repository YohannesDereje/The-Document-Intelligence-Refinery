from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import BBox, SemanticChunk
from src.agents.vector_store import VectorStore


def _unit_to_chunk(unit: dict, default_section: str) -> SemanticChunk | None:
    content = str(unit.get("content_markdown") or unit.get("content_raw") or "").strip()
    if not content:
        return None

    provenance = unit.get("provenance", {}) if isinstance(unit.get("provenance", {}), dict) else {}
    page_number = int(provenance.get("page_number", 1) or 1)
    if page_number <= 0:
        page_number = 1

    bbox_raw = unit.get("bbox", {}) if isinstance(unit.get("bbox", {}), dict) else {}
    bbox = BBox(
        x1=float(bbox_raw.get("x1", 0.0)),
        y1=float(bbox_raw.get("y1", 0.0)),
        x2=float(bbox_raw.get("x2", 1.0)),
        y2=float(bbox_raw.get("y2", 1.0)),
    )

    unit_type = str(unit.get("unit_type", "text")).strip() or "text"
    section_context = f"{default_section} > {unit_type}"
    token_count = max(1, len(content.split()))

    return SemanticChunk(
        content=content,
        page_numbers=[page_number],
        bbox_bounds=bbox,
        section_context=section_context,
        token_count=token_count,
    )


def _collect_chunks(profiles_dir: Path, max_pages_per_doc: int | None) -> list[SemanticChunk]:
    refined_files = sorted(profiles_dir.glob("*_refined.json"))
    if not refined_files:
        raise FileNotFoundError(f"No refined profiles found in: {profiles_dir}")

    all_chunks: list[SemanticChunk] = []

    for refined_path in refined_files:
        payload = json.loads(refined_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            print(f"Skipped {refined_path.name}: expected JSON object")
            continue

        file_id = str(payload.get("file_id", refined_path.stem)).strip() or refined_path.stem
        raw_units = payload.get("units", []) if isinstance(payload.get("units", []), list) else []

        chunks: list[SemanticChunk] = []
        for unit in raw_units:
            if not isinstance(unit, dict):
                continue
            chunk = _unit_to_chunk(unit, default_section=file_id)
            if chunk is None:
                continue
            if max_pages_per_doc is not None and int(chunk.page_numbers[0]) > max_pages_per_doc:
                continue
            chunks.append(chunk)

        all_chunks.extend(chunks)
        print(f"Loaded {refined_path.name}: {len(chunks)} chunk(s)")

    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build .refinery/vector_db from refined extraction outputs.")
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=PROJECT_ROOT / ".refinery" / "profiles",
        help="Directory containing *_refined.json outputs.",
    )
    parser.add_argument(
        "--vector-db-dir",
        type=Path,
        default=PROJECT_ROOT / ".refinery" / "vector_db",
        help="Target Chroma persistence directory.",
    )
    parser.add_argument(
        "--rules-path",
        type=Path,
        default=PROJECT_ROOT / "rubric" / "extraction_rules.yaml",
        help="Unused compatibility flag; retained for stable CLI usage.",
    )
    parser.add_argument(
        "--max-pages-per-doc",
        type=int,
        default=None,
        help="Optional page cap while chunking refined documents.",
    )
    args = parser.parse_args()

    if not args.profiles_dir.exists():
        raise FileNotFoundError(f"Profiles directory not found: {args.profiles_dir}")
    chunks = _collect_chunks(
        profiles_dir=args.profiles_dir,
        max_pages_per_doc=args.max_pages_per_doc,
    )
    if not chunks:
        raise RuntimeError("No semantic chunks were generated from refined profiles.")

    store = VectorStore(persist_directory=args.vector_db_dir, collection_name="semantic_chunks")
    indexed_count = store.populate_store(chunks=chunks, reset=True)

    print(f"Indexed chunks: {indexed_count}")
    print(f"Vector DB path: {args.vector_db_dir.resolve()}")


if __name__ == "__main__":
    main()
