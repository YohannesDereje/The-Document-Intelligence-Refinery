# Document Intelligence Refinery

End-to-end PDF triage + extraction pipeline using a strategy router:

- **Strategy A**: fast text extraction (`FastTextExtractor`)
- **Strategy B**: layout-aware extraction with Docling (`LayoutExtractor`)
- **Strategy C**: vision extraction via OpenRouter VLM (`VisionExtractor`)

The pipeline performs:

1. **Phase 1 (Triage)**: profile pages and assign `dominant_strategy`
2. **Phase 2 (Extraction)**: route pages through A/B/C with escalation, ledgering, and refined output

## 1) Prerequisites

- Windows, macOS, or Linux
- Python **3.10+** (project currently tested with Python 3.13)
- Internet access for Strategy C (OpenRouter API)

## 2) Project Structure (key paths)

- `data/` → input PDF corpus
- `rubric/extraction_rules.yaml` → routing thresholds/rules
- `run_refinery.py` → full orchestration script
- `src/agents/triage.py` / `triage_agent.py` → triage logic
- `src/agents/extractor.py` → extraction router + escalation + ledger writes
- `.refinery/profiles/` → refined JSON outputs
- `.refinery/extraction_ledger.jsonl` → per-page strategy ledger

## 3) Install Dependencies

From project root:

```bash
python -m pip install -U pip
python -m pip install -e .
python -m pip install -e .[dev]
```

If your shell does not support extras cleanly, install dev tools directly:

```bash
python -m pip install pytest ipykernel python-dotenv
```

## 4) Configure API Key (.env)

Create a `.env` file in project root:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
```

`run_refinery.py` calls `load_dotenv()` and automatically reads this key.

## 5) Add Input PDFs

Put your corpus PDFs in `data/`.

The current interim configuration processes **all PDFs** found in `data/`, but limits extraction to the **first 3 pages per document**.

## 6) Run Tests

```bash
python -m pytest tests -q
```

Included tests:

- `tests/test_triage.py`
- `tests/test_confidence.py`

## 7) Run Full Pipeline

Use your active Python executable from project root:

```bash
python run_refinery.py
```

On your current Windows environment, equivalent explicit command is:

```bash
C:/Users/Yohannes/AppData/Local/Programs/Python/Python313/python.exe run_refinery.py
```

## 8) What the Run Produces

- Refined document outputs:
	- `.refinery/profiles/<file_stem>_refined.json`
- Extraction ledger (JSONL, append-only):
	- `.refinery/extraction_ledger.jsonl`

Each ledger line includes:

- `file_name`
- `strategy_selected`
- `confidence_score`
- `processing_time_ms`
- `estimated_cost_usd`

## 9) Runtime Behavior (Current)

- `ExtractionRouter` budget is set to **$30.00** in `run_refinery.py`
- Per-page strategy timeout is **60 seconds**
- Escalation flow: **A → B → C**
- If one strategy fails for a page, router escalates
- If all strategies fail for a page, it logs and continues to next page/document

## 10) Troubleshooting

### `No endpoints found` / 404 from OpenRouter

This is usually a model routing issue, not a bad key.

- Verify model slug in `src/strategies/vision_extractor.py`
- Current default is `openai/gpt-4o-mini`
- Keep OpenRouter base URL as:
	- `https://openrouter.ai/api/v1`

### `std::bad_alloc` in Docling preprocess

`LayoutExtractor` is configured to prefer `PyPdfiumDocumentBackend` and can fallback to Strategy C on conversion failure.

### Resume behavior

Current `run_refinery.py` is configured to process the full discovered corpus each run. If you want skip-on-existing-output behavior again, add an `output_path.exists()` guard in the main loop.