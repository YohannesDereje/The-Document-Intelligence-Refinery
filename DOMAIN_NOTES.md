# Document Intelligence Refinery - Domain Notes

## Corpus Classes

### 1) Pure Scans
- **Observed profile:** Character density = `0.0`, image-dominant pages, little or no extractable text layer.
- **Likely origin:** Physical scan pipelines and scan-PDF normalization workflows (often post-processed by tools such as iLovePDF).
- **Failure mode:** Text-first pipelines fail because content exists primarily as pixels.
- **Required strategy:** **Strategy C** (Vision/OCR, most complex/expensive).

### 2) Born Digital
- **Observed profile:** Character density > `0.0005`, consistent text layer, low geometric complexity.
- **Likely origin:** Native digital authoring/export (MS Word, Google Docs, desktop publishing, direct PDF export).
- **Failure mode:** Overly heavy pipelines waste latency/cost when fast text extraction is sufficient.
- **Required strategy:** **Strategy A** (Fast Text, fastest/cheapest, high-speed CPU extraction).

### 3) Hybrid / Landscape
- **Observed profile:** Mixed orientation and geometry, including landscape pages and/or high rectangle counts (`rects > 50`).
- **Likely origin:** Merged production workflows (digital originals + inserted scanned exhibits), rotation/landscape post-processing.
- **Failure mode:** Fast text extraction may lose table structure and row/column relationships.
- **Required strategy:** **Strategy B** (Structural/Table parsing to preserve layout fidelity).

## Extraction Thresholds

The following thresholds are used for deterministic routing:

- `MIN_DIGITAL_DENSITY: 0.0005`
- `TABLE_RECT_THRESHOLD: 50`

## Strategy Mapping Summary

- **Strategy A (Fast Text - Fastest/Cheapest):** Use for native digital documents with a consistent text layer and low geometric complexity; execute high-speed CPU text extraction.
- **Strategy B (Structural/Table - Medium Complexity):** Use for digital documents with `rects > 50` or landscape orientation (`width > height`); execute structural parsing to preserve table rows/columns.
- **Strategy C (Vision/OCR - Most Complex/Expensive):** Use for pure scans (`density = 0`) or image-driven pages; execute VLM/OCR to reconstruct text from pixels.