from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from src.agents.indexer import PageIndexBuilder
from src.agents.qa_agent import QAGraphAgent, _chunks_from_vector_store
from src.agents.vector_store import VectorStore


class QueryInterfaceAgent:
    """Lightweight adapter exposing run()/audit_claim() for CLI usage."""

    def __init__(self, base_agent: QAGraphAgent) -> None:
        self.base_agent = base_agent

    @staticmethod
    def _infer_mode(tool_history: List[str]) -> str:
        for tool_name in tool_history:
            if tool_name in {"semantic_search", "pageindex_navigate"}:
                return tool_name
        if tool_history:
            return tool_history[0]
        return "unknown"

    def run(self, query: str) -> Dict[str, Any]:
        initial_state: Dict[str, Any] = {
            "user_query": query,
            "mode": "qa",
            "retrieved_chunks": [],
            "sql_outputs": [],
            "final_generated_answer": "",
            "final_response": {"answer": "", "provenance_chain": []},
            "audit_status": "UNVERIFIABLE",
            "audit_reasoning": "",
            "audit_contradiction_checked": False,
            "chosen_tool": "",
            "tool_history": [],
            "tool_outputs": [],
            "needs_more_retrieval": False,
            "route_to": "retriever",
            "max_tool_calls": 2,
        }

        final_state = self.base_agent.graph.invoke(initial_state)
        final_response = final_state.get("final_response", {})
        answer = str(final_response.get("answer", "")).strip() if isinstance(final_response, dict) else ""
        provenance = final_response.get("provenance_chain", []) if isinstance(final_response, dict) else []
        if not isinstance(provenance, list):
            provenance = []

        tool_history = final_state.get("tool_history", [])
        if not isinstance(tool_history, list):
            tool_history = []

        return {
            "answer": answer,
            "provenance_chain": provenance,
            "mode": self._infer_mode([str(item) for item in tool_history]),
        }

    def audit_claim(self, claim: str) -> Dict[str, Any]:
        initial_state: Dict[str, Any] = {
            "user_query": claim,
            "mode": "audit",
            "retrieved_chunks": [],
            "sql_outputs": [],
            "final_generated_answer": "",
            "final_response": {"answer": "", "provenance_chain": []},
            "audit_status": "UNVERIFIABLE",
            "audit_reasoning": "",
            "audit_contradiction_checked": False,
            "chosen_tool": "",
            "tool_history": [],
            "tool_outputs": [],
            "needs_more_retrieval": False,
            "route_to": "retriever",
            "max_tool_calls": 3,
        }

        final_state = self.base_agent.graph.invoke(initial_state)
        final_response = final_state.get("final_response", {})
        answer = str(final_response.get("answer", "")).strip() if isinstance(final_response, dict) else ""
        provenance = final_response.get("provenance_chain", []) if isinstance(final_response, dict) else []
        if not isinstance(provenance, list):
            provenance = []

        tool_history = final_state.get("tool_history", [])
        if not isinstance(tool_history, list):
            tool_history = []

        return {
            "answer": answer,
            "provenance_chain": provenance,
            "mode": self._infer_mode([str(item) for item in tool_history]),
            "status": str(final_state.get("audit_status", "UNVERIFIABLE")).strip().upper(),
        }


def _validate_refinery_state(project_root: Path) -> tuple[Path, Path, Path]:
    refinery_dir = project_root / ".refinery"
    vector_db_dir = refinery_dir / "vector_db"
    fact_db_path = refinery_dir / "fact_store.db"
    pageindex_dir = refinery_dir / "pageindex"

    missing: List[str] = []
    if not refinery_dir.exists():
        missing.append(str(refinery_dir))
    if not vector_db_dir.exists():
        missing.append(str(vector_db_dir))
    if not fact_db_path.exists():
        missing.append(str(fact_db_path))
    if not pageindex_dir.exists():
        missing.append(str(pageindex_dir))

    if missing:
        message = [
            "Missing required .refinery artifacts.",
            "Run the pipeline first so vector store, fact DB, and page index exist.",
            "Missing paths:",
        ]
        message.extend([f"- {path}" for path in missing])
        raise FileNotFoundError("\n".join(message))

    return vector_db_dir, fact_db_path, pageindex_dir


def _build_query_agent(project_root: Path) -> QueryInterfaceAgent:
    vector_db_dir, fact_db_path, _ = _validate_refinery_state(project_root)

    # Validate fact table can be opened before starting CLI.
    with sqlite3.connect(fact_db_path) as conn:
        conn.execute("SELECT 1")

    vector_store = VectorStore(persist_directory=vector_db_dir, collection_name="semantic_chunks")
    indexed_chunks = _chunks_from_vector_store(vector_store)

    page_index_builder = PageIndexBuilder(persistence_path=project_root / ".refinery" / "page_index.json")
    if indexed_chunks:
        page_index_builder.build_tree(indexed_chunks, persist=True)

    return QueryInterfaceAgent(QAGraphAgent(page_index_builder=page_index_builder, vector_store=vector_store))


def _print_response(response: Dict[str, Any]) -> None:
    answer = str(response.get("answer", "")).strip()
    mode = str(response.get("mode", "unknown")).strip() or "unknown"
    provenance = response.get("provenance_chain", [])
    if not isinstance(provenance, list):
        provenance = []

    print("\n=== ANSWER ===")
    print(answer if answer else "(no answer)")

    print("\n=== PROVENANCE ===")
    if not provenance:
        print("No provenance returned.")
    else:
        for idx, item in enumerate(provenance, start=1):
            if not isinstance(item, dict):
                continue
            document = str(item.get("document_name", "Unknown document")).strip() or "Unknown document"
            page = int(item.get("page_number", 0) or 0)
            bbox = item.get("bbox_bounds", {}) if isinstance(item.get("bbox_bounds", {}), dict) else {}
            x1 = float(bbox.get("x1", 0.0))
            y1 = float(bbox.get("y1", 0.0))
            x2 = float(bbox.get("x2", 0.0))
            y2 = float(bbox.get("y2", 0.0))
            print(f"{idx}. Document: {document} | Page: {page} | BBox: ({x1:.3f}, {y1:.3f}) -> ({x2:.3f}, {y2:.3f})")

    print("\n=== MODE ===")
    print(mode)


def main() -> None:
    project_root = Path(__file__).resolve().parent

    try:
        query_agent = _build_query_agent(project_root)
    except Exception as exc:
        print("Failed to initialize query CLI.")
        print(str(exc))
        return

    while True:
        user_input = input("Enter Query (or 'exit'): ").strip()
        if user_input.lower() == "exit":
            print("Exiting.")
            break
        if not user_input:
            continue

        try:
            if user_input.startswith("Verify:"):
                claim = user_input[len("Verify:") :].strip()
                if not claim:
                    print("Provide a claim after 'Verify:'.")
                    continue
                response = query_agent.audit_claim(claim)
            else:
                response = query_agent.run(user_input)
        except Exception as exc:
            print("Query failed.")
            print(str(exc))
            continue

        _print_response(response)


if __name__ == "__main__":
    main()
