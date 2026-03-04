from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pdfplumber

def format_table_fallback(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator_line = "-+-".join("-" * width for width in widths)
    body_lines = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    try:
        from tabulate import tabulate

        return tabulate(rows, headers=headers, tablefmt="github")
    except Exception:
        return format_table_fallback(headers, rows)


def analyze_pdf(pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], int, float]:
    metrics: list[dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        metadata = pdf.metadata or {}

        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text_length = len(text)

            width = float(page.width or 0.0)
            height = float(page.height or 0.0)
            area = width * height
            density = (text_length / area) if area > 0 else 0.0

            metrics.append(
                {
                    "page_number": page_index,
                    "text_length": text_length,
                    "width": width,
                    "height": height,
                    "char_density": density,
                    "images": len(page.images),
                    "rects": len(page.rects),
                    "curves": len(page.curves),
                }
            )

    total_pages = len(metrics)
    average_density = (
        sum(page_metrics["char_density"] for page_metrics in metrics) / total_pages if total_pages else 0.0
    )

    return metrics, metadata, total_pages, average_density


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze PDF physical characteristics and text density."
    )
    parser.add_argument("file_path", type=Path, help="Path to the input PDF file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path: Path = args.file_path

    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Invalid PDF path: {pdf_path}")

    rows, metadata, total_pages, average_density = analyze_pdf(pdf_path=pdf_path)

    if not total_pages:
        raise SystemExit("No pages available to analyze.")

    print(f"Analyzed file: {pdf_path}")
    print()

    headers = [
        "Page Number",
        "Text Length",
        "Width",
        "Height",
        "Character Density",
        "Images",
        "Rects",
        "Curves",
    ]
    table_rows = [
        [
            str(row["page_number"]),
            str(row["text_length"]),
            f"{row['width']:.2f}",
            f"{row['height']:.2f}",
            f"{row['char_density']:.6f}",
            str(row["images"]),
            str(row["rects"]),
            str(row["curves"]),
        ]
        for row in rows
    ]

    print(render_table(headers, table_rows))
    print()

    print("PDF Metadata")
    if metadata:
        for key, value in metadata.items():
            print(f"- {key}: {value}")
    else:
        print("- No metadata found")

    print()
    print(f"Total Pages Processed: {total_pages} | Average Density: {average_density:.6f}")


if __name__ == "__main__":
    main()
