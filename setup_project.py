from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    print(f"[dir]  {path}")


def ensure_file(path: Path, content: str = "") -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        print(f"[file] {path}")
    else:
        print(f"[skip] {path} (already exists)")


def main() -> None:
    root = Path(__file__).resolve().parent

    # Directory structure
    directories = [
        root / "src" / "agents",
        root / "src" / "models",
        root / "src" / "strategies",
        root / "tests",
        root / "rubric",
        root / "data",
        root / ".refinery",
        root / ".refinery" / "profiles",
    ]

    for directory in directories:
        ensure_directory(directory)

    # Make src subdirectories Python packages
    init_files = [
        root / "src" / "agents" / "__init__.py",
        root / "src" / "models" / "__init__.py",
        root / "src" / "strategies" / "__init__.py",
    ]

    for init_file in init_files:
        ensure_file(init_file)

    # Placeholder files
    ensure_file(
        root / "src" / "models" / "schemas.py",
        '"""Pydantic schemas for the Document Intelligence Refinery project."""\n',
    )
    ensure_file(
        root / "rubric" / "extraction_rules.yaml",
        "# Extraction configuration rules\n",
    )
    ensure_file(
        root / "DOMAIN_NOTES.md",
        "# Domain Notes\n\nPhase 0 documentation lives here.\n",
    )

    # Note for refinery artifacts
    print(
        "\nNote: '.refinery/' stores JSON and JSONL artifacts "
        "(for example, extraction ledger and profile outputs)."
    )


if __name__ == "__main__":
    main()
