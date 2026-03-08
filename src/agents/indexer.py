from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import BBox, PageIndex, SemanticChunk
from src.config import ConfigLoader

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


class PageIndexBuilder:
    def __init__(
        self,
        rules_path: str | Path = "rubric/extraction_rules.yaml",
        summary_model: str = "meta-llama/llama-3.3-70b-instruct",
        top_k_sections: int = 3,
        persistence_path: str | Path = ".refinery/page_index.json",
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.config_loader = ConfigLoader(rules_path)
        self.summary_model = summary_model
        self.top_k_sections = max(1, int(top_k_sections))
        self.persistence_path = Path(persistence_path)

        self.provider_name = "none"
        self.provider_base_url = ""

        groq_api_key = os.getenv("GROQ_API_KEY")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if groq_api_key:
            api_key = groq_api_key
            self.provider_name = "groq"
            self.provider_base_url = "https://api.groq.com/openai/v1"
            # Groq uses provider-specific model IDs; map common OpenRouter alias if needed.
            if self.summary_model == "meta-llama/llama-3.3-70b-instruct":
                self.summary_model = "llama-3.3-70b-versatile"
        elif openrouter_api_key:
            api_key = openrouter_api_key
            self.provider_name = "openrouter"
            self.provider_base_url = "https://openrouter.ai/api/v1"
        else:
            api_key = openai_api_key
            self.provider_name = "openai" if openai_api_key else "none"
            self.provider_base_url = "https://api.openai.com/v1"

        self.client: Any = None
        if api_key and OpenAI is not None and self.provider_base_url:
            try:
                self.client = OpenAI(api_key=api_key, base_url=self.provider_base_url)
            except Exception as exc:  # pragma: no cover
                self.logger.warning("Failed to initialize summarization client: %s", exc)

        self.root: Optional[PageIndex] = None
        self._section_nodes: Dict[str, PageIndex] = {}
        self._section_chunks: Dict[str, List[SemanticChunk]] = {}

    @staticmethod
    def _slug(text: str) -> str:
        lowered = text.strip().lower()
        cleaned = re.sub(r"[^a-z0-9]+", "-", lowered)
        return cleaned.strip("-") or "untitled"

    @staticmethod
    def _merge_chunk_bboxes(chunks: List[SemanticChunk]) -> BBox:
        if not chunks:
            return BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)

        return BBox(
            x1=min(chunk.bbox_bounds.x1 for chunk in chunks),
            y1=min(chunk.bbox_bounds.y1 for chunk in chunks),
            x2=max(chunk.bbox_bounds.x2 for chunk in chunks),
            y2=max(chunk.bbox_bounds.y2 for chunk in chunks),
        )

    @staticmethod
    def _split_section_path(section_context: str) -> List[str]:
        value = (section_context or "").strip()
        if not value:
            return ["Unsectioned"]

        for sep in (">", "::", "|", "/"):
            if sep in value:
                parts = [part.strip().lstrip("#").strip() for part in value.split(sep)]
                filtered = [part for part in parts if part]
                return filtered or ["Unsectioned"]

        single = value.lstrip("#").strip()
        return [single] if single else ["Unsectioned"]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _semantic_similarity(self, query: str, candidate: str) -> float:
        q = self._tokenize(query)
        c = self._tokenize(candidate)
        if not q or not c:
            return 0.0

        overlap = len(q & c)
        union = len(q | c)
        jaccard = overlap / union if union else 0.0
        coverage = overlap / max(1, len(q))
        return (0.6 * jaccard) + (0.4 * coverage)

    @staticmethod
    def _fallback_summary(section_title: str, section_text: str) -> str:
        compact = " ".join(section_text.split())
        if not compact:
            return f"Section '{section_title}' has no extracted content yet."

        sentence = re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)[0]
        if len(sentence) > 220:
            sentence = sentence[:217].rstrip() + "..."
        if sentence and sentence[-1] not in ".!?":
            sentence = sentence + "."
        return sentence

    def _llm_section_summary(self, section_title: str, section_text: str) -> str:
        if self.client is None:
            return self._fallback_summary(section_title, section_text)

        prompt = (
            "Write exactly one sentence summarizing the section for retrieval routing. "
            "Keep it factual, concise, and include key entities/topics.\n\n"
            f"Section title: {section_title}\n"
            f"Section content:\n{section_text[:4000]}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": "You summarize document sections for search indexing."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=64,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return self._fallback_summary(section_title, section_text)
            one_line = " ".join(text.split())
            if one_line[-1] not in ".!?":
                one_line += "."
            return one_line
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Section summary generation failed for '%s': %s", section_title, exc)
            return self._fallback_summary(section_title, section_text)

    def build_tree(self, chunks: List[SemanticChunk], persist: bool = True) -> PageIndex:
        sorted_chunks = sorted(chunks, key=lambda item: (item.page_numbers[0], item.section_context, item.content_hash))
        root_bbox = self._merge_chunk_bboxes(sorted_chunks)
        root_page = sorted_chunks[0].page_numbers[0] if sorted_chunks else 0
        all_pages = sorted({int(page) for chunk in sorted_chunks for page in chunk.page_numbers})
        root = PageIndex(
            page_number=root_page,
            bbox=root_bbox,
            node_id="root",
            node_type="root",
            title="Document Root",
            summary="",
            metadata={
                "total_chunks": len(sorted_chunks),
                "total_pages_covered": len(all_pages),
                "min_page": all_pages[0] if all_pages else 0,
                "max_page": all_pages[-1] if all_pages else 0,
            },
            children=[],
        )

        self._section_nodes = {}
        self._section_chunks = {}

        for chunk in sorted_chunks:
            path_parts = self._split_section_path(chunk.section_context)
            parent = root
            top_section_id = ""

            for depth, title in enumerate(path_parts):
                node_type = "section" if depth == 0 else "subsection"
                node_id = f"{'/'.join(self._slug(part) for part in path_parts[: depth + 1])}"
                existing = next((child for child in parent.children if child.node_id == node_id), None)
                if existing is None:
                    existing = PageIndex(
                        page_number=chunk.page_numbers[0],
                        bbox=chunk.bbox_bounds,
                        node_id=node_id,
                        node_type=node_type,
                        title=title,
                        summary="",
                        metadata={
                            "min_page": int(chunk.page_numbers[0]),
                            "max_page": int(chunk.page_numbers[-1]),
                            "chunk_count": 0,
                        },
                        children=[],
                    )
                    parent.children.append(existing)
                else:
                    existing.page_number = min(int(existing.page_number), int(chunk.page_numbers[0]))
                    existing.bbox = BBox(
                        x1=min(existing.bbox.x1, chunk.bbox_bounds.x1),
                        y1=min(existing.bbox.y1, chunk.bbox_bounds.y1),
                        x2=max(existing.bbox.x2, chunk.bbox_bounds.x2),
                        y2=max(existing.bbox.y2, chunk.bbox_bounds.y2),
                    )
                    existing.metadata["min_page"] = min(
                        int(existing.metadata.get("min_page", chunk.page_numbers[0])),
                        int(chunk.page_numbers[0]),
                    )
                    existing.metadata["max_page"] = max(
                        int(existing.metadata.get("max_page", chunk.page_numbers[-1])),
                        int(chunk.page_numbers[-1]),
                    )

                parent = existing
                if depth == 0:
                    top_section_id = node_id
                    self._section_nodes[top_section_id] = existing

            chunk_node = PageIndex(
                page_number=chunk.page_numbers[0],
                bbox=chunk.bbox_bounds,
                node_id=f"chunk/{chunk.content_hash[:16]}",
                node_type="chunk",
                title=f"Chunk {chunk.content_hash[:8]}",
                summary="",
                metadata={
                    "content_hash": chunk.content_hash,
                    "token_count": chunk.token_count,
                    "page_numbers": chunk.page_numbers,
                    "section_context": chunk.section_context,
                },
                chunk_hashes=[chunk.content_hash],
                children=[],
            )
            parent.children.append(chunk_node)

            if top_section_id:
                self._section_chunks.setdefault(top_section_id, []).append(chunk)
                top_node = self._section_nodes.get(top_section_id)
                if top_node is not None:
                    top_node.metadata["chunk_count"] = int(top_node.metadata.get("chunk_count", 0)) + 1

        for section_id, section_node in self._section_nodes.items():
            section_chunks = self._section_chunks.get(section_id, [])
            combined = "\n\n".join(chunk.content for chunk in section_chunks)
            summary = self._llm_section_summary(section_node.title or section_id, combined)
            section_node.summary = summary
            section_node.metadata["summary"] = summary
            pages = sorted({int(page) for chunk in section_chunks for page in chunk.page_numbers})
            section_node.metadata["total_pages_covered"] = len(pages)
            section_node.metadata["min_page"] = pages[0] if pages else 0
            section_node.metadata["max_page"] = pages[-1] if pages else 0
            section_node.chunk_hashes = [chunk.content_hash for chunk in section_chunks]

        self.root = root
        if persist:
            self.serialize()
        return root

    def traverse_query(self, query: str) -> List[SemanticChunk]:
        if not query.strip() or not self._section_nodes:
            return []

        ranked: List[tuple[float, str]] = []
        for section_id, section_node in self._section_nodes.items():
            candidate_text = f"{section_node.title} {section_node.summary}".strip()
            score = self._semantic_similarity(query, candidate_text)
            if score > 0.0:
                ranked.append((score, section_id))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected_ids = [section_id for _, section_id in ranked[: self.top_k_sections]]

        seen: set[str] = set()
        selected_chunks: List[SemanticChunk] = []
        for section_id in selected_ids:
            for chunk in self._section_chunks.get(section_id, []):
                if chunk.content_hash in seen:
                    continue
                seen.add(chunk.content_hash)
                selected_chunks.append(chunk)

        selected_chunks.sort(key=lambda item: (item.page_numbers[0], item.content_hash))
        return selected_chunks

    def serialize(self, output_path: str | Path | None = None) -> Path:
        if self.root is None:
            raise ValueError("No index tree available to serialize. Call build_tree first.")

        target = Path(output_path) if output_path is not None else self.persistence_path
        os.makedirs(target.parent, exist_ok=True)

        # Use mode="json" to coerce nested pydantic fields into JSON-safe primitives.
        payload = self.root.model_dump(mode="json")
        try:
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=True)
        except Exception as exc:
            self.logger.exception("Failed to persist page index JSON at %s: %s", target, exc)
            raise

        return target
