from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from triage_agent import TriageAgent


def main() -> None:
    """Bulk-process all PDFs in data/ and build triage_manifest.json."""
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    manifest_path = project_root / "triage_manifest.json"

    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    # Discover all PDF files in a stable order for reproducible manifests.
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found in: {data_dir}")

    # Initialize the triage agent once and reuse it across the entire batch.
    agent = TriageAgent(rules_path="rubric/extraction_rules.yaml")

    # Aggregate successful profiles by filename and keep failures separate.
    manifest: dict[str, dict] = {}
    failures: dict[str, str] = {}

    # Process each file with per-item exception handling so one bad file does not stop the run.
    for pdf_file in tqdm(pdf_files, desc="Triaging PDFs", unit="file"):
        try:
            profile = agent.process_pdf(pdf_file)
            manifest[pdf_file.name] = profile.model_dump()
        except Exception as exc:
            failures[pdf_file.name] = str(exc)

    # Persist only successful profile outputs as a JSON-serializable manifest.
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    # Build summary metrics from successfully processed documents.
    origin_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for profile_dict in manifest.values():
        overall_origin = str(profile_dict.get("overall_origin", "unknown"))
        domain_hint = str(profile_dict.get("domain_hint", "unknown"))

        origin_counts[overall_origin] = origin_counts.get(overall_origin, 0) + 1
        domain_counts[domain_hint] = domain_counts.get(domain_hint, 0) + 1

    print()
    print(f"Manifest saved to: {manifest_path}")
    print(f"Total documents processed: {len(manifest)}")

    print("Overall origin counts:")
    if origin_counts:
        for origin, count in sorted(origin_counts.items()):
            print(f"- {origin}: {count}")
    else:
        print("- none")

    print("Domain hint counts:")
    if domain_counts:
        for domain_hint, count in sorted(domain_counts.items()):
            print(f"- {domain_hint}: {count}")
    else:
        print("- none")

    if failures:
        print(f"Files failed during processing: {len(failures)}")
        for file_name, error_message in sorted(failures.items()):
            print(f"- {file_name}: {error_message}")


if __name__ == "__main__":
    main()
