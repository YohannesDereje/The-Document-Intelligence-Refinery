from __future__ import annotations

import logging
import re
import importlib
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from models import BBox, SemanticChunk

try:
    import chromadb
except Exception as exc:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    _CHROMA_IMPORT_ERROR = exc
else:
    _CHROMA_IMPORT_ERROR = None

_SENTENCE_TRANSFORMER_IMPORT_ERROR: Exception | None = None


class VectorStore:
    _embedding_model: ClassVar[Any] = None
    _embedding_model_name: ClassVar[str | None] = None

    def __init__(
        self,
        persist_directory: str | Path = ".refinery/vector_db",
        collection_name: str = "semantic_chunks",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.logger = logging.getLogger(__name__)

        if chromadb is None:
            raise RuntimeError(
                f"chromadb is required for VectorStore but is not installed: {_CHROMA_IMPORT_ERROR}"
            )
        model_class = self._load_sentence_transformer_class()
        if model_class is None:
            raise RuntimeError(
                "sentence-transformers is required for VectorStore but is not installed: "
                f"{_SENTENCE_TRANSFORMER_IMPORT_ERROR}"
            )

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.model = self._get_embedding_model(self.embedding_model_name)

    @classmethod
    @staticmethod
    def _load_sentence_transformer_class() -> Any:
        global _SENTENCE_TRANSFORMER_IMPORT_ERROR
        try:
            module = importlib.import_module("sentence_transformers")
            return getattr(module, "SentenceTransformer")
        except Exception as exc:  # pragma: no cover
            _SENTENCE_TRANSFORMER_IMPORT_ERROR = exc
            return None

    @classmethod
    def _get_embedding_model(cls, model_name: str) -> Any:
        # Singleton model avoids reloading MiniLM on every search/index run.
        model_class = cls._load_sentence_transformer_class()
        if model_class is None:
            raise RuntimeError(
                "sentence-transformers is required for VectorStore but is not installed: "
                f"{_SENTENCE_TRANSFORMER_IMPORT_ERROR}"
            )
        if cls._embedding_model is None or cls._embedding_model_name != model_name:
            cls._embedding_model = model_class(model_name)
            cls._embedding_model_name = model_name
        return cls._embedding_model

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _hybrid_score(self, query: str, document: str, semantic_distance: float) -> float:
        query_tokens = self._tokenize(query)
        document_tokens = self._tokenize(document)
        if query_tokens and document_tokens:
            overlap = len(query_tokens & document_tokens)
            coverage = overlap / max(1, len(query_tokens))
        else:
            coverage = 0.0

        semantic_score = 1.0 / (1.0 + max(0.0, float(semantic_distance)))
        return (0.8 * semantic_score) + (0.2 * coverage)

    @staticmethod
    def _chunk_to_metadata(chunk: SemanticChunk) -> Dict[str, Any]:
        return {
            # Required metadata fields:
            "content_hash": chunk.content_hash,
            "page_numbers": ",".join(str(page) for page in chunk.page_numbers),
            "section_context": chunk.section_context,
            # Additional fields for exact SemanticChunk reconstruction:
            "bbox_x1": float(chunk.bbox_bounds.x1),
            "bbox_y1": float(chunk.bbox_bounds.y1),
            "bbox_x2": float(chunk.bbox_bounds.x2),
            "bbox_y2": float(chunk.bbox_bounds.y2),
            "token_count": int(chunk.token_count),
        }

    @staticmethod
    def _metadata_to_chunk(content: str, metadata: Dict[str, Any]) -> SemanticChunk:
        page_numbers_raw = str(metadata.get("page_numbers", "")).strip()
        page_numbers = [
            int(part)
            for part in page_numbers_raw.split(",")
            if part.strip().isdigit() and int(part.strip()) > 0
        ]
        if not page_numbers:
            page_numbers = [1]

        bbox = BBox(
            x1=float(metadata.get("bbox_x1", 0.0)),
            y1=float(metadata.get("bbox_y1", 0.0)),
            x2=float(metadata.get("bbox_x2", 1.0)),
            y2=float(metadata.get("bbox_y2", 1.0)),
        )

        token_count = int(metadata.get("token_count", max(1, len(content.split()))))
        content_hash = str(metadata.get("content_hash", "")).strip()

        return SemanticChunk(
            content=content,
            page_numbers=page_numbers,
            bbox_bounds=bbox,
            section_context=str(metadata.get("section_context", "Unknown")).strip() or "Unknown",
            token_count=max(1, token_count),
            content_hash=content_hash,
        )

    def _reset_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def populate_store(self, chunks: List[SemanticChunk], reset: bool = True) -> int:
        if reset:
            self._reset_collection()

        if not chunks:
            return 0

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        seen_ids: Dict[str, int] = {}
        for index, chunk in enumerate(chunks):
            base_id = chunk.content_hash or f"chunk-{index}"
            count = seen_ids.get(base_id, 0)
            seen_ids[base_id] = count + 1
            row_id = base_id if count == 0 else f"{base_id}-{count}"

            documents.append(chunk.content)
            metadatas.append(self._chunk_to_metadata(chunk))
            ids.append(row_id)

        embeddings = self.model.encode(documents, normalize_embeddings=True).tolist()

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(ids)

    def semantic_search(self, query: str, top_k: int = 5) -> List[SemanticChunk]:
        clean_query = query.strip()
        if not clean_query:
            return []

        search_k = max(1, int(top_k))
        probe_k = min(max(search_k * 4, search_k), 50)

        query_embedding = self.model.encode([clean_query], normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=probe_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        ranked_rows: List[tuple[float, str, Dict[str, Any]]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if document is None or metadata is None:
                continue
            score = self._hybrid_score(clean_query, str(document), float(distance))
            ranked_rows.append((score, str(document), dict(metadata)))

        ranked_rows.sort(key=lambda item: item[0], reverse=True)

        chunks: List[SemanticChunk] = []
        seen_hashes: set[str] = set()
        for _, document, metadata in ranked_rows:
            chunk = self._metadata_to_chunk(content=document, metadata=metadata)
            if chunk.content_hash in seen_hashes:
                continue
            seen_hashes.add(chunk.content_hash)
            chunks.append(chunk)
            if len(chunks) >= search_k:
                break

        return chunks
