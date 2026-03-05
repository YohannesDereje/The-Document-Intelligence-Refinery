from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    def __init__(self, config_path: str | Path = "rubric/extraction_rules.yaml") -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        raw_yaml = self.config_path.read_text(encoding="utf-8")
        normalized_yaml = raw_yaml.replace("\t", "  ")
        loaded = yaml.safe_load(normalized_yaml) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Configuration must be a YAML mapping/object.")
        self.config = loaded

    def get(self, dotted_key: str, default: Any = None) -> Any:
        cursor: Any = self.config
        for part in dotted_key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor
