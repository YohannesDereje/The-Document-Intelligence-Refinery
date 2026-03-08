from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from src.agents.chunker import ChunkingEngine
from src.agents.extractor import ExtractionRouter
from src.agents.indexer import PageIndexBuilder
from src.agents.triage import TriageAgent

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REFINERY_DIR = PROJECT_ROOT / ".refinery"
OUTPUT_DIR = REFINERY_DIR / "profiles"
PAGEINDEX_DIR = REFINERY_DIR / "pageindex"
LEDGER_PATH = REFINERY_DIR / "extraction_ledger.jsonl"
RULES_PATH = PROJECT_ROOT / "rubric" / "extraction_rules.yaml"
MAX_PAGES_PER_DOC = 21


def _iter_with_progress(pdf_files: list[Path]):
    if tqdm is not None:
        return tqdm(pdf_files, desc="Refinery Pipeline", unit="doc")
    return pdf_files


def _safe_pageindex_stem(file_name: str, fallback_stem: str) -> str:
    raw = Path(str(file_name).strip()).stem if str(file_name).strip() else fallback_stem
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return sanitized or fallback_stem


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    os.makedirs(REFINERY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PAGEINDEX_DIR, exist_ok=True)
    if not LEDGER_PATH.exists():
        LEDGER_PATH.write_text("", encoding="utf-8")

    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found in: {DATA_DIR}")

    triage_agent = TriageAgent(rules_path=RULES_PATH, max_pages_per_doc=MAX_PAGES_PER_DOC)

    # Prefer explicit key override, then OpenAI, then OpenRouter.
    api_key = (
        os.getenv("VISION_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )
    extraction_router = ExtractionRouter(
        api_key=api_key,
        rules_path=RULES_PATH,
        max_pages_per_doc=MAX_PAGES_PER_DOC,
    )
    chunking_engine = ChunkingEngine(rules_path=RULES_PATH, max_pages_per_doc=MAX_PAGES_PER_DOC)

    total_processed = 0
    total_pages = 0
    failures: list[tuple[str, str]] = []

    iterator = _iter_with_progress(pdf_files)
    total_docs = len(pdf_files)
    print(f"Pipeline Start: found {total_docs} document(s) in {DATA_DIR}")

    for index, pdf_path in enumerate(iterator, start=1):
        if tqdm is None:
            print(f"Processing {index} of {total_docs}: {pdf_path.name}...")

        output_path = OUTPUT_DIR / f"{pdf_path.stem}_refined.json"

        try:
            profile = triage_agent.process_pdf(pdf_path)
            # Global page cap: process first N pages per document consistently.
            limited_pages = profile.pages[:MAX_PAGES_PER_DOC]
            profile = profile.model_copy(
                update={
                    "pages": limited_pages,
                    "total_pages": len(limited_pages),
                }
            )
            refined = extraction_router.process_document(pdf_path=pdf_path, profile=profile)

            output_path.write_text(refined.model_dump_json(indent=4), encoding="utf-8")
            logger.info("Saved refined output: %s", output_path)

            # Build semantic chunks from extracted LDUs and persist page index per document.
            semantic_chunks = chunking_engine.chunk_document(refined)
            pageindex_stem = _safe_pageindex_stem(profile.file_name, pdf_path.stem)
            pageindex_path = PAGEINDEX_DIR / f"{pageindex_stem}.json"

            page_index_builder = PageIndexBuilder(
                rules_path=RULES_PATH,
                persistence_path=pageindex_path,
            )

            page_index_builder.build_tree(chunks=semantic_chunks, persist=False)
            written_path = page_index_builder.serialize(pageindex_path)
            logger.info(
                "Saved page index: %s (chunks=%s, pages_target=%s)",
                written_path,
                len(semantic_chunks),
                profile.total_pages,
            )

            total_processed += 1
            total_pages += profile.total_pages
        except Exception as exc:
            failures.append((pdf_path.name, str(exc)))
            logger.exception("Failed processing %s", pdf_path.name)

    total_cost = float(getattr(extraction_router, "total_cost_usd", extraction_router.vision_extractor.total_spend))

    print("\nRun Summary")
    print(f"- Total documents processed: {total_processed}")
    print(f"- Total pages: {total_pages}")
    print(f"- Total cost: ${total_cost:.4f}")

    if failures:
        print(f"- Failed documents: {len(failures)}")
        for file_name, error_message in failures:
            print(f"  - {file_name}: {error_message}")


if __name__ == "__main__":
    main()
