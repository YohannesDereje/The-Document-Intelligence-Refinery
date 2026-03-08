from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, TypedDict

from openai import OpenAI

from models import BBox, PageIndex, SemanticChunk
from src.agents.indexer import PageIndexBuilder
from src.agents.vector_store import VectorStore

try:
    from langgraph.graph import END, START, StateGraph
except Exception as exc:  # pragma: no cover
    StateGraph = None  # type: ignore[assignment]
    START = "START"  # type: ignore[assignment]
    END = "END"  # type: ignore[assignment]
    _LANGGRAPH_IMPORT_ERROR = exc
else:
    _LANGGRAPH_IMPORT_ERROR = None

try:
    from langchain_core.tools import tool
except Exception as exc:  # pragma: no cover
    tool = None  # type: ignore[assignment]
    _LANGCHAIN_IMPORT_ERROR = exc
else:
    _LANGCHAIN_IMPORT_ERROR = None


class GraphState(TypedDict):
    user_query: str
    mode: Literal["qa", "audit"]
    retrieved_chunks: List[SemanticChunk]
    sql_outputs: List[Dict[str, Any]]
    final_generated_answer: str
    final_response: "QAWithProvenance"
    audit_status: Literal["VERIFIED", "CONTRADICTED", "UNVERIFIABLE"]
    audit_reasoning: str
    audit_contradiction_checked: bool
    chosen_tool: str
    tool_history: List[str]
    tool_outputs: List[str]
    needs_more_retrieval: bool
    route_to: Literal["retriever", "generator"]
    max_tool_calls: int


class BBoxBoundsDict(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float


class ProvenanceItem(TypedDict):
    document_name: str
    content_hash: str
    page_number: int
    section_context: str
    bbox_bounds: BBoxBoundsDict


class QAWithProvenance(TypedDict):
    answer: str
    provenance_chain: List[ProvenanceItem]


class QAGraphAgent:
    def __init__(
        self,
        page_index_builder: PageIndexBuilder,
        vector_store: VectorStore,
        router_model: str = "gpt-4o-mini",
        generator_model: str = "gpt-4o-mini",
    ) -> None:
        if StateGraph is None:
            raise RuntimeError(f"langgraph is required but unavailable: {_LANGGRAPH_IMPORT_ERROR}")
        if tool is None:
            raise RuntimeError(f"langchain-core is required but unavailable: {_LANGCHAIN_IMPORT_ERROR}")

        self.logger = logging.getLogger(__name__)
        self.page_index_builder = page_index_builder
        self.vector_store = vector_store
        self.router_model = router_model
        self.generator_model = generator_model
        self.semantic_search_top_k = max(5, int(os.getenv("QA_SEMANTIC_SEARCH_TOP_K", "20")))
        self.fact_extraction_batch_size = max(10, int(os.getenv("QA_FACT_BATCH_SIZE", "40")))

        self.fact_db_path = Path(".refinery/fact_store.db")
        self.fact_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_fact_db()

        self.llm_client = self._build_llm_client()
        self.tools = self._build_tools()
        self.tool_by_name = {registered_tool.name: registered_tool for registered_tool in self.tools}
        self.graph = self._build_graph()

    def _init_fact_db(self) -> None:
        with sqlite3.connect(self.fact_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_facts (
                    fact_key TEXT,
                    fact_value TEXT,
                    entity TEXT,
                    content_hash TEXT,
                    page_number INTEGER
                )
                """
            )
            conn.commit()

    @staticmethod
    def _build_llm_client() -> OpenAI | None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if groq_api_key:
            return OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
        if openrouter_api_key:
            return OpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1")
        if openai_api_key:
            return OpenAI(api_key=openai_api_key, base_url="https://api.openai.com/v1")
        return None

    def _section_summaries_for_chunks(self, chunks: List[SemanticChunk]) -> List[str]:
        summaries: List[str] = []
        if self.page_index_builder.root is None:
            return summaries

        section_nodes: Dict[str, PageIndex] = {}

        def walk(node: PageIndex) -> None:
            if node.node_type in {"section", "subsection"}:
                section_nodes[node.title] = node
            for child in node.children:
                walk(child)

        walk(self.page_index_builder.root)

        seen: set[str] = set()
        for chunk in chunks:
            key = chunk.section_context.strip().split(">")[0].strip().lstrip("#")
            if not key or key in seen:
                continue
            seen.add(key)
            node = section_nodes.get(key)
            if node is not None and node.summary:
                summaries.append(f"{node.title}: {node.summary}")

        return summaries

    def _document_outline_titles(self) -> List[str]:
        root = self.page_index_builder.root
        if root is None:
            return ["Document Root"]

        lines: List[str] = []

        def walk(node: PageIndex, depth: int) -> None:
            title = node.title or node.node_id or "(untitled)"
            indent = "  " * depth
            lines.append(f"{indent}- {title}")
            for child in node.children:
                walk(child, depth + 1)

        walk(root, 0)
        return lines

    def _document_name(self) -> str:
        root = self.page_index_builder.root
        if root is not None:
            metadata = getattr(root, "metadata", {}) or {}
            source_file = str(metadata.get("source_file", "")).strip()
            if source_file:
                return Path(source_file).name
            title = str(root.title or "").strip()
            if title and title.lower() != "document root":
                return title
        return "Indexed Document"

    def _build_tools(self) -> List[Any]:
        @tool
        def pageindex_navigate(query: str) -> Dict[str, Any]:
            """Navigate high-level section summaries relevant to the query."""
            chunks = self.page_index_builder.traverse_query(query)
            summaries = self._section_summaries_for_chunks(chunks)
            return {
                "tool": "pageindex_navigate",
                "section_summaries": summaries,
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            }

        @tool
        def semantic_search(query: str) -> Dict[str, Any]:
            """Retrieve the top-5 semantically relevant chunks from the vector store."""
            chunks = self.vector_store.semantic_search(query=query, top_k=self.semantic_search_top_k)
            return {
                "tool": "semantic_search",
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            }

        @tool
        def document_outline(query: str = "") -> Dict[str, Any]:
            """Return the full document outline so the agent can reason about structure."""
            _ = query
            outline_lines = self._document_outline_titles()
            return {
                "tool": "document_outline",
                "outline": outline_lines,
                "chunks": [],
            }

        @tool
        def extract_and_store_facts(query: str, chunks_json: str = "") -> Dict[str, Any]:
            """Extract hard facts from chunks and persist them into SQLite document_facts table."""
            _ = query
            try:
                raw_chunks = json.loads(chunks_json) if chunks_json.strip() else []
            except json.JSONDecodeError:
                raw_chunks = []

            chunks = self._chunks_from_payload({"chunks": raw_chunks})
            facts = self._extract_hard_facts(chunks)
            inserted = self._insert_facts(facts)
            return {
                "tool": "extract_and_store_facts",
                "inserted": inserted,
                "facts": facts,
                "chunks": [],
            }

        @tool
        def sql_query_tool(query: str) -> Dict[str, Any]:
            """Run SELECT-only SQL against document_facts and return rows as JSON list."""
            sql, params, generation_error = self._sql_from_user_query(query)
            if generation_error:
                return {
                    "tool": "sql_query_tool",
                    "sql": sql,
                    "rows": [],
                    "error": generation_error,
                    "chunks": [],
                }

            execution = self._run_select_query(sql, params)
            return {
                "tool": "sql_query_tool",
                "sql": sql,
                "rows": execution.get("rows", []),
                "error": execution.get("error", ""),
                "chunks": [],
            }

        return [
            pageindex_navigate,
            semantic_search,
            document_outline,
            extract_and_store_facts,
            sql_query_tool,
        ]

    @staticmethod
    def _is_metric_or_aggregation_query(query: str) -> bool:
        lowered = query.lower()
        signals = [
            "total",
            "sum",
            "compare",
            "list all",
            "revenue",
            "metric",
            "amount",
            "percentage",
        ]
        return any(signal in lowered for signal in signals)

    def _extract_hard_facts(self, chunks: List[SemanticChunk]) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        if self.llm_client is None:
            return self._fallback_fact_extraction(chunks)

        all_facts: List[Dict[str, Any]] = []
        batch_size = max(10, int(self.fact_extraction_batch_size))

        for start_idx in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start_idx : start_idx + batch_size]
            serialized_chunks = [
                {
                    "content": chunk.content,
                    "content_hash": chunk.content_hash,
                    "page_number": int(chunk.page_numbers[0]),
                    "section_context": chunk.section_context,
                }
                for chunk in chunk_batch
            ]

            prompt = (
                "Extract hard facts (dates, currency, percentages, names, specific metrics) from the chunks. "
                "Return strict JSON array where each item has keys: fact_key, fact_value, entity, content_hash, page_number."
                " If none, return [].\n\n"
                f"Chunks:\n{json.dumps(serialized_chunks, ensure_ascii=True)}"
            )

            try:
                response = self.llm_client.chat.completions.create(
                    model=self.generator_model,
                    messages=[
                        {"role": "system", "content": "You extract structured facts from text evidence."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=800,
                )
                raw = (response.choices[0].message.content or "[]").strip()
                start = raw.find("[")
                end = raw.rfind("]")
                if start >= 0 and end >= start:
                    raw = raw[start : end + 1]
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError("Fact extractor did not return a JSON list.")

                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    all_facts.append(
                        {
                            "fact_key": str(item.get("fact_key", "")).strip(),
                            "fact_value": str(item.get("fact_value", "")).strip(),
                            "entity": str(item.get("entity", "")).strip(),
                            "content_hash": str(item.get("content_hash", "")).strip(),
                            "page_number": int(item.get("page_number", 0) or 0),
                        }
                    )
            except Exception:
                all_facts.extend(self._fallback_fact_extraction(chunk_batch))

        deduped: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str, str, int]] = set()
        for fact in all_facts:
            fact_key = str(fact.get("fact_key", "")).strip()
            fact_value = str(fact.get("fact_value", "")).strip()
            entity = str(fact.get("entity", "")).strip()
            content_hash = str(fact.get("content_hash", "")).strip()
            page_number = int(fact.get("page_number", 0) or 0)
            if not fact_key or not fact_value or not content_hash:
                continue
            signature = (fact_key, fact_value, entity, content_hash, page_number)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(
                {
                    "fact_key": fact_key,
                    "fact_value": fact_value,
                    "entity": entity,
                    "content_hash": content_hash,
                    "page_number": page_number,
                }
            )

        return deduped

    @staticmethod
    def _fallback_fact_extraction(chunks: List[SemanticChunk]) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        money_re = re.compile(r"(?:\$|USD\s?)\d[\d,]*(?:\.\d+)?", flags=re.IGNORECASE)
        percent_re = re.compile(r"\b\d+(?:\.\d+)?%")
        date_re = re.compile(r"\b(?:19|20)\d{2}\b")

        for chunk in chunks:
            page_number = int(chunk.page_numbers[0])
            for value in money_re.findall(chunk.content):
                facts.append(
                    {
                        "fact_key": "currency_amount",
                        "fact_value": value,
                        "entity": chunk.section_context,
                        "content_hash": chunk.content_hash,
                        "page_number": page_number,
                    }
                )
            for value in percent_re.findall(chunk.content):
                facts.append(
                    {
                        "fact_key": "percentage",
                        "fact_value": value,
                        "entity": chunk.section_context,
                        "content_hash": chunk.content_hash,
                        "page_number": page_number,
                    }
                )
            for value in date_re.findall(chunk.content):
                facts.append(
                    {
                        "fact_key": "year",
                        "fact_value": value,
                        "entity": chunk.section_context,
                        "content_hash": chunk.content_hash,
                        "page_number": page_number,
                    }
                )
        return facts

    def _insert_facts(self, facts: List[Dict[str, Any]]) -> int:
        if not facts:
            return 0

        inserted = 0
        try:
            with sqlite3.connect(self.fact_db_path) as conn:
                for fact in facts:
                    conn.execute(
                        """
                        INSERT INTO document_facts (fact_key, fact_value, entity, content_hash, page_number)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            str(fact.get("fact_key", "")),
                            str(fact.get("fact_value", "")),
                            str(fact.get("entity", "")),
                            str(fact.get("content_hash", "")),
                            int(fact.get("page_number", 0) or 0),
                        ),
                    )
                    inserted += 1
                conn.commit()
        except sqlite3.Error as exc:
            self.logger.warning("Failed inserting extracted facts into SQLite: %s", exc)
            return inserted
        return inserted

    @staticmethod
    def _validate_select_only_sql(sql: str) -> tuple[bool, str]:
        stripped = sql.strip()
        lowered = stripped.lower()
        if not lowered.startswith("select"):
            return False, "Only SELECT statements are allowed for sql_query_tool."

        blocked = ["drop", "delete", "update", "insert", "alter", "create", "replace", "truncate", "pragma"]
        tokenized = re.findall(r"[a-z_]+", lowered)
        if any(keyword in tokenized for keyword in blocked):
            return False, "Unsafe SQL detected (mutation/DDL keywords are not allowed)."

        if ";" in stripped[:-1]:
            return False, "Multiple statements are not allowed."

        if "from document_facts" not in lowered:
            return False, "Queries must target document_facts."

        return True, ""

    def _sql_from_user_query(self, query: str) -> tuple[str, tuple[Any, ...], str]:
        lowered = query.lower()
        if self.llm_client is not None:
            schema_prompt = (
                "You generate SQL query plans for SQLite table document_facts. "
                "Schema: fact_key TEXT, fact_value TEXT, entity TEXT, content_hash TEXT, page_number INTEGER. "
                "Return STRICT JSON object with keys: select_mode, fact_key_like, entity_like, fact_value_like, "
                "content_hash_eq, page_number_eq, limit. "
                "select_mode must be one of: rows, count. Never emit SQL or mutation commands."
            )
            user_prompt = f"User query: {query}"
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.router_model,
                    messages=[
                        {"role": "system", "content": schema_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=180,
                )
                raw = (response.choices[0].message.content or "{}").strip()
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end >= start:
                    raw = raw[start : end + 1]
                plan = json.loads(raw)
                if isinstance(plan, dict):
                    sql, params = self._sql_from_plan(plan)
                    ok, err = self._validate_select_only_sql(sql)
                    if not ok:
                        return sql, params, err
                    return sql, params, ""
            except Exception:
                pass

        # Deterministic fallback query planner.
        if "revenue" in lowered and ("total" in lowered or "sum" in lowered):
            sql = (
                "SELECT fact_key, fact_value, entity, content_hash, page_number "
                "FROM document_facts WHERE lower(fact_key) LIKE ? OR lower(entity) LIKE ?"
            )
            params = ("%revenue%", "%revenue%")
        elif "list all" in lowered or "all facts" in lowered:
            sql = "SELECT fact_key, fact_value, entity, content_hash, page_number FROM document_facts"
            params = ()
        else:
            sql = (
                "SELECT fact_key, fact_value, entity, content_hash, page_number "
                "FROM document_facts WHERE lower(entity) LIKE ? OR lower(fact_key) LIKE ?"
            )
            safe_term = f"%{re.sub(r'[^a-z0-9\\s]', '', lowered).strip()}%"
            params = (safe_term, safe_term)

        ok, err = self._validate_select_only_sql(sql)
        if not ok:
            return sql, params, err
        return sql, params, ""

    @staticmethod
    def _sql_from_plan(plan: Dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
        mode = str(plan.get("select_mode", "rows")).strip().lower()
        select_clause = "COUNT(*) AS total_rows" if mode == "count" else "fact_key, fact_value, entity, content_hash, page_number"

        sql = f"SELECT {select_clause} FROM document_facts"
        conditions: List[str] = []
        params: List[Any] = []

        def add_like(column: str, key: str) -> None:
            value = str(plan.get(key, "")).strip().lower()
            if value:
                conditions.append(f"lower({column}) LIKE ?")
                params.append(f"%{value}%")

        add_like("fact_key", "fact_key_like")
        add_like("entity", "entity_like")
        add_like("fact_value", "fact_value_like")

        content_hash_eq = str(plan.get("content_hash_eq", "")).strip()
        if content_hash_eq:
            conditions.append("content_hash = ?")
            params.append(content_hash_eq)

        page_number_eq = plan.get("page_number_eq", None)
        if page_number_eq is not None and str(page_number_eq).strip() != "":
            try:
                page_number = int(page_number_eq)
                conditions.append("page_number = ?")
                params.append(page_number)
            except (TypeError, ValueError):
                pass

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        if mode != "count":
            try:
                limit = int(plan.get("limit", 25) or 25)
            except (TypeError, ValueError):
                limit = 25
            limit = max(1, min(limit, 100))
            sql += " LIMIT ?"
            params.append(limit)

        return sql, tuple(params)

    def _run_select_query(self, sql: str, params: Sequence[Any] | None = None) -> Dict[str, Any]:
        ok, err = self._validate_select_only_sql(sql)
        if not ok:
            return {"rows": [], "error": err}

        safe_params: Sequence[Any] = params or ()
        try:
            with sqlite3.connect(self.fact_db_path) as conn:
                cursor = conn.execute(sql, tuple(safe_params))
                columns = [description[0] for description in (cursor.description or [])]
                rows = cursor.fetchall()

            output: List[Dict[str, Any]] = []
            for row in rows:
                output.append({columns[idx]: row[idx] for idx in range(len(columns))})
            return {"rows": output, "error": ""}
        except sqlite3.Error as exc:
            return {"rows": [], "error": f"SQL execution error: {exc}"}

    @staticmethod
    def _parse_router_json(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end >= start:
            cleaned = cleaned[start : end + 1]
        return json.loads(cleaned)

    @staticmethod
    def _heuristic_router(query: str, used_tools: List[str]) -> Dict[str, Any]:
        lowered = query.lower()

        if any(token in lowered for token in ["outline", "structure", "table of contents"]):
            primary = "document_outline"
        elif any(token in lowered for token in ["section", "chapter", "summar", "overview"]):
            primary = "pageindex_navigate"
        else:
            primary = "semantic_search"

        if primary in used_tools:
            for fallback in ["semantic_search", "pageindex_navigate", "document_outline"]:
                if fallback not in used_tools:
                    primary = fallback
                    break

        return {
            "tool": primary,
            "need_more": False,
            "next_tool": "",
            "reason": "Heuristic fallback routing",
        }

    def _audit_router(self, state: GraphState) -> Dict[str, Any]:
        used_tools = state["tool_history"]

        if "semantic_search" not in used_tools:
            return {
                "tool": "semantic_search",
                "need_more": True,
                "next_tool": "pageindex_navigate",
                "reason": "Audit mode first gathers closest semantic matches.",
            }

        if "pageindex_navigate" not in used_tools:
            return {
                "tool": "pageindex_navigate",
                "need_more": True,
                "next_tool": "semantic_search",
                "reason": "Audit mode also gathers section-level contextual evidence.",
            }

        if not state.get("audit_contradiction_checked", False):
            return {
                "tool": "semantic_search",
                "need_more": False,
                "next_tool": "",
                "reason": "Audit mode performs contradiction-focused retrieval before verdict.",
            }

        return {
            "tool": "semantic_search",
            "need_more": False,
            "next_tool": "",
            "reason": "Audit evidence collection completed.",
        }

    def _router_node(self, state: GraphState) -> GraphState:
        query = state["user_query"].strip()
        used_tools = list(state["tool_history"])

        if state.get("mode", "qa") == "audit":
            router_decision = self._audit_router(state)
            state["chosen_tool"] = str(router_decision.get("tool", "semantic_search"))
            state["needs_more_retrieval"] = bool(router_decision.get("need_more", False))
            state["route_to"] = "retriever"
            return state

        if self._is_metric_or_aggregation_query(query):
            if not state["retrieved_chunks"] and "semantic_search" not in used_tools:
                state["chosen_tool"] = "semantic_search"
                state["needs_more_retrieval"] = True
                state["route_to"] = "retriever"
                return state
            if state["retrieved_chunks"] and "extract_and_store_facts" not in used_tools:
                state["chosen_tool"] = "extract_and_store_facts"
                state["needs_more_retrieval"] = True
                state["route_to"] = "retriever"
                return state
            state["chosen_tool"] = "sql_query_tool"
            state["needs_more_retrieval"] = False
            state["route_to"] = "retriever"
            return state

        if not query:
            state["chosen_tool"] = "document_outline"
            state["needs_more_retrieval"] = False
            state["route_to"] = "retriever"
            return state

        router_decision: Dict[str, Any]
        if self.llm_client is None:
            router_decision = self._heuristic_router(query, used_tools)
        else:
            try:
                prompt = (
                    "Decide which retrieval tool should run next for answering the user query. "
                    "Available tools: pageindex_navigate, semantic_search, document_outline, "
                    "extract_and_store_facts, sql_query_tool. "
                    "Return strict JSON with keys: tool, need_more, next_tool, reason. "
                    "Tool must be one of the available tools. next_tool may be empty.\n\n"
                    f"User query: {query}\n"
                    f"Already used tools: {used_tools}\n"
                    f"Retrieved chunk count: {len(state['retrieved_chunks'])}\n"
                    f"SQL output rows count: {len(state['sql_outputs'])}\n"
                )
                response = self.llm_client.chat.completions.create(
                    model=self.router_model,
                    messages=[
                        {"role": "system", "content": "You are a routing controller for a multi-tool QA graph."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=120,
                )
                text = (response.choices[0].message.content or "").strip()
                router_decision = self._parse_router_json(text)
            except Exception:
                router_decision = self._heuristic_router(query, used_tools)

        chosen_tool = str(router_decision.get("tool", "semantic_search")).strip()
        if chosen_tool not in self.tool_by_name:
            chosen_tool = "semantic_search"

        state["chosen_tool"] = chosen_tool
        state["needs_more_retrieval"] = bool(router_decision.get("need_more", False))
        state["route_to"] = "retriever"
        return state

    @staticmethod
    def _chunks_from_payload(payload: Dict[str, Any]) -> List[SemanticChunk]:
        raw_chunks = payload.get("chunks", [])
        chunks: List[SemanticChunk] = []
        for raw in raw_chunks:
            if not isinstance(raw, dict):
                continue
            bbox_raw = raw.get("bbox_bounds", {})
            bbox = BBox(
                x1=float(bbox_raw.get("x1", 0.0)),
                y1=float(bbox_raw.get("y1", 0.0)),
                x2=float(bbox_raw.get("x2", 1.0)),
                y2=float(bbox_raw.get("y2", 1.0)),
            )
            chunks.append(
                SemanticChunk(
                    content=str(raw.get("content", "")).strip(),
                    page_numbers=[int(page) for page in raw.get("page_numbers", [1])],
                    bbox_bounds=bbox,
                    section_context=str(raw.get("section_context", "Unknown")),
                    token_count=int(raw.get("token_count", 1) or 1),
                    content_hash=str(raw.get("content_hash", "")).strip(),
                )
            )
        return chunks

    def _retriever_node(self, state: GraphState) -> GraphState:
        tool_name = state["chosen_tool"]
        tool_impl = self.tool_by_name[tool_name]
        tool_query = state["user_query"]

        if state.get("mode", "qa") == "audit" and tool_name == "semantic_search":
            if "semantic_search" in state["tool_history"] and not state.get("audit_contradiction_checked", False):
                tool_query = (
                    "Find evidence that could contradict or negate this claim. "
                    "Focus on statements that indicate the opposite is true: "
                    f"{state['user_query']}"
                )
                state["audit_contradiction_checked"] = True

        if tool_name == "extract_and_store_facts":
            payload = tool_impl.invoke(
                {
                    "query": tool_query,
                    "chunks_json": json.dumps(
                        [chunk.model_dump(mode="json") for chunk in state["retrieved_chunks"]],
                        ensure_ascii=True,
                    ),
                }
            )
        else:
            payload = tool_impl.invoke({"query": tool_query})
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {"tool": tool_name, "raw": payload, "chunks": []}

        new_chunks = self._chunks_from_payload(payload if isinstance(payload, dict) else {"chunks": []})

        if isinstance(payload, dict):
            if payload.get("tool") == "sql_query_tool":
                rows = payload.get("rows", [])
                if isinstance(rows, list):
                    state["sql_outputs"] = [dict(row) for row in rows if isinstance(row, dict)]
            if payload.get("tool") == "extract_and_store_facts":
                facts = payload.get("facts", [])
                if isinstance(facts, list):
                    normalized_facts = [dict(row) for row in facts if isinstance(row, dict)]
                    if normalized_facts:
                        state["sql_outputs"] = normalized_facts

        existing_hashes = {chunk.content_hash for chunk in state["retrieved_chunks"]}
        for chunk in new_chunks:
            if chunk.content_hash not in existing_hashes:
                state["retrieved_chunks"].append(chunk)
                existing_hashes.add(chunk.content_hash)

        state["tool_history"].append(tool_name)
        state["tool_outputs"].append(json.dumps(payload, ensure_ascii=True)[:2000])

        too_many_calls = len(state["tool_history"]) >= max(1, state["max_tool_calls"])
        if state["needs_more_retrieval"] and not too_many_calls:
            state["route_to"] = "retriever"  # set placeholder; conditional edge will send to router
        else:
            state["route_to"] = "generator"
        return state

    @staticmethod
    def _extract_used_hashes(answer: str, chunks: List[SemanticChunk]) -> set[str]:
        used: set[str] = set()
        valid_hashes = {chunk.content_hash for chunk in chunks if chunk.content_hash}

        for content_hash in valid_hashes:
            if content_hash in answer:
                used.add(content_hash)

        # Optional machine-readable hash list emitted by the generator.
        match = re.search(r"USED_HASHES\s*:\s*(\[[^\]]*\]|[^\n]+)", answer, flags=re.IGNORECASE)
        if not match:
            return used

        raw = match.group(1).strip()
        parsed_values: List[str] = []
        if raw.startswith("[") and raw.endswith("]"):
            candidate = raw.replace("'", '"')
            try:
                payload = json.loads(candidate)
                if isinstance(payload, list):
                    parsed_values = [str(item).strip() for item in payload]
            except json.JSONDecodeError:
                parsed_values = [part.strip().strip("'\"") for part in raw[1:-1].split(",") if part.strip()]
        else:
            parsed_values = [part.strip().strip("'\"") for part in raw.split(",") if part.strip()]

        for item in parsed_values:
            if item in valid_hashes:
                used.add(item)
        return used

    @staticmethod
    def _build_provenance_chain(
        chunks: List[SemanticChunk],
        used_hashes: set[str],
        fallback_mode: bool,
        document_name: str,
    ) -> List[ProvenanceItem]:
        if used_hashes:
            selected_chunks = [chunk for chunk in chunks if chunk.content_hash in used_hashes]
        elif fallback_mode:
            selected_chunks = chunks[:5]
        else:
            selected_chunks = []

        provenance_chain: List[ProvenanceItem] = []
        for chunk in selected_chunks:
            provenance_chain.append(
                {
                    "document_name": document_name,
                    "content_hash": chunk.content_hash,
                    "page_number": int(chunk.page_numbers[0]),
                    "section_context": chunk.section_context,
                    "bbox_bounds": {
                        "x1": float(chunk.bbox_bounds.x1),
                        "y1": float(chunk.bbox_bounds.y1),
                        "x2": float(chunk.bbox_bounds.x2),
                        "y2": float(chunk.bbox_bounds.y2),
                    },
                }
            )
        return provenance_chain

    def _analyze_claim_verification(self, claim: str, chunks: List[SemanticChunk]) -> Dict[str, Any]:
        if not chunks:
            return {
                "status": "UNVERIFIABLE",
                "reasoning": "No supporting or contradicting evidence was retrieved from the indexed document.",
                "used_hashes": [],
            }

        context_lines = [
            f"[hash={chunk.content_hash} page={chunk.page_numbers[0]}] {chunk.content.strip()}"
            for chunk in chunks
        ]
        evidence_block = "\n".join(context_lines)

        if self.llm_client is None:
            cited = ", ".join(
                [f"{chunk.content_hash} (page {chunk.page_numbers[0]})" for chunk in chunks[:2]]
            )
            return {
                "status": "UNVERIFIABLE",
                "reasoning": (
                    "Automatic audit requires an LLM analyzer for semantic verification. "
                    f"Retrieved evidence was inconclusive: {cited}."
                ),
                "used_hashes": [chunk.content_hash for chunk in chunks[:2]],
            }

        prompt = (
            "Assess whether the claim is supported, contradicted, or not verifiable from evidence. "
            "Return strict JSON with keys: status, reasoning, used_hashes. "
            "status must be exactly one of VERIFIED, CONTRADICTED, UNVERIFIABLE. "
            "reasoning must reference specific citations in the form content_hash and page number.\n\n"
            f"Claim: {claim}\n\n"
            f"Evidence:\n{evidence_block}"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.generator_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict audit verifier. Prefer CONTRADICTED when evidence directly conflicts with the claim. "
                            "Prefer UNVERIFIABLE when evidence is weak, indirect, or missing."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=260,
            )
            raw = (response.choices[0].message.content or "{}").strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end >= start:
                raw = raw[start : end + 1]
            payload = json.loads(raw)
        except Exception:
            payload = {}

        status = str(payload.get("status", "UNVERIFIABLE")).strip().upper()
        if status not in {"VERIFIED", "CONTRADICTED", "UNVERIFIABLE"}:
            status = "UNVERIFIABLE"

        reasoning = str(payload.get("reasoning", "")).strip()
        if not reasoning:
            reasoning = "Evidence was insufficient to confidently verify or contradict the claim."

        valid_hashes = {chunk.content_hash for chunk in chunks}
        used_hashes_raw = payload.get("used_hashes", [])
        used_hashes: List[str] = []
        if isinstance(used_hashes_raw, list):
            for item in used_hashes_raw:
                candidate = str(item).strip()
                if candidate in valid_hashes and candidate not in used_hashes:
                    used_hashes.append(candidate)

        if not used_hashes:
            parsed_used = self._extract_used_hashes(reasoning, chunks)
            used_hashes = list(parsed_used)

        if not used_hashes:
            used_hashes = [chunk.content_hash for chunk in chunks[:2] if chunk.content_hash]

        return {
            "status": status,
            "reasoning": reasoning,
            "used_hashes": used_hashes,
        }

    def _generator_node(self, state: GraphState) -> GraphState:
        chunks = state["retrieved_chunks"]
        document_name = self._document_name()

        if state.get("mode", "qa") == "audit":
            analysis = self._analyze_claim_verification(state["user_query"], chunks)
            status = str(analysis.get("status", "UNVERIFIABLE"))
            reasoning = str(analysis.get("reasoning", "")).strip()
            used_hashes = {
                str(item).strip() for item in analysis.get("used_hashes", []) if str(item).strip()
            }

            answer = f"Status: {status}\nReasoning: {reasoning}"
            state["audit_status"] = status if status in {"VERIFIED", "CONTRADICTED", "UNVERIFIABLE"} else "UNVERIFIABLE"
            state["audit_reasoning"] = reasoning
            state["final_generated_answer"] = answer
            state["final_response"] = {
                "answer": answer,
                "provenance_chain": self._build_provenance_chain(
                    chunks=chunks,
                    used_hashes=used_hashes,
                    fallback_mode=not bool(used_hashes),
                    document_name=document_name,
                ),
            }
            return state

        if not chunks:
            answer = (
                "I could not find grounded evidence in the current index for this query. "
                "Try rephrasing the question or indexing more documents."
            )
            state["final_generated_answer"] = answer
            state["final_response"] = {"answer": answer, "provenance_chain": []}
            return state

        context_lines = []
        for chunk in chunks:
            context_lines.append(
                f"[hash={chunk.content_hash} page={chunk.page_numbers[0]}] {chunk.content.strip()}"
            )
        context_block = "\n".join(context_lines)

        if self.llm_client is None:
            # Deterministic fallback with strict citations.
            bullets = [
                f"- {chunk.content.strip()} (content_hash={chunk.content_hash}, page={chunk.page_numbers[0]})"
                for chunk in chunks[:5]
            ]
            answer = "\n".join(bullets)
            state["final_generated_answer"] = answer
            state["final_response"] = {
                "answer": answer,
                "provenance_chain": self._build_provenance_chain(
                    chunks=chunks,
                    used_hashes=self._extract_used_hashes(answer, chunks),
                    fallback_mode=True,
                    document_name=document_name,
                ),
            }
            return state

        prompt = (
            "Answer the user question using only the provided evidence chunks. "
            "For every factual claim, cite both content_hash and page_number. "
            "If evidence is insufficient, explicitly say so.\n\n"
            f"User query: {state['user_query']}\n\n"
            f"Evidence:\n{context_block}\n\n"
            f"Structured Facts (SQL):\n{json.dumps(state['sql_outputs'], ensure_ascii=True)}"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.generator_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a grounded QA system. Cite (content_hash=..., page=...) for each factual claim. "
                            "After the answer, append a final machine-readable line: "
                            "USED_HASHES: [\"hash1\", \"hash2\"]."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=450,
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                answer = "No answer generated from evidence."
            state["final_generated_answer"] = answer
            used_hashes = self._extract_used_hashes(answer, chunks)
            state["final_response"] = {
                "answer": answer,
                "provenance_chain": self._build_provenance_chain(
                    chunks=chunks,
                    used_hashes=used_hashes,
                    fallback_mode=False,
                    document_name=document_name,
                ),
            }
        except Exception:
            bullets = [
                f"- {chunk.content.strip()} (content_hash={chunk.content_hash}, page={chunk.page_numbers[0]})"
                for chunk in chunks[:5]
            ]
            answer = "\n".join(bullets)
            state["final_generated_answer"] = answer
            state["final_response"] = {
                "answer": answer,
                "provenance_chain": self._build_provenance_chain(
                    chunks=chunks,
                    used_hashes=self._extract_used_hashes(answer, chunks),
                    fallback_mode=True,
                    document_name=document_name,
                ),
            }

        return state

    def _build_graph(self) -> Any:
        graph = StateGraph(GraphState)
        graph.add_node("router", self._router_node)
        graph.add_node("retriever", self._retriever_node)
        graph.add_node("generator", self._generator_node)

        graph.add_edge(START, "router")

        def route_after_router(state: GraphState) -> str:
            return "retriever" if state.get("route_to") == "retriever" else "generator"

        graph.add_conditional_edges("router", route_after_router, {"retriever": "retriever", "generator": "generator"})

        def route_after_retriever(state: GraphState) -> str:
            if state.get("needs_more_retrieval", False) and len(state.get("tool_history", [])) < max(
                1, int(state.get("max_tool_calls", 2))
            ):
                return "router"
            return "generator"

        graph.add_conditional_edges("retriever", route_after_retriever, {"router": "router", "generator": "generator"})
        graph.add_edge("generator", END)
        return graph.compile()

    def ask(self, query: str) -> QAWithProvenance:
        initial_state: GraphState = {
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
        final_state = self.graph.invoke(initial_state)
        final_response = final_state.get("final_response")
        if isinstance(final_response, dict):
            answer = str(final_response.get("answer", "")).strip()
            chain = final_response.get("provenance_chain", [])
            if not isinstance(chain, list):
                chain = []
            return {"answer": answer, "provenance_chain": chain}

        fallback_answer = str(final_state.get("final_generated_answer", "")).strip()
        return {"answer": fallback_answer, "provenance_chain": []}

    def audit_claim(self, claim: str) -> Dict[str, Any]:
        initial_state: GraphState = {
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

        final_state = self.graph.invoke(initial_state)
        final_response = final_state.get("final_response", {"answer": "", "provenance_chain": []})
        provenance_chain = final_response.get("provenance_chain", []) if isinstance(final_response, dict) else []
        if not isinstance(provenance_chain, list):
            provenance_chain = []

        status = str(final_state.get("audit_status", "UNVERIFIABLE")).strip().upper()
        if status not in {"VERIFIED", "CONTRADICTED", "UNVERIFIABLE"}:
            status = "UNVERIFIABLE"

        reasoning = str(final_state.get("audit_reasoning", "")).strip()
        if not reasoning and isinstance(final_response, dict):
            answer = str(final_response.get("answer", "")).strip()
            match = re.search(r"Reasoning:\s*(.+)", answer, flags=re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip()

        return {
            "claim": claim,
            "status": status,
            "reasoning": reasoning,
            "provenance_chain": provenance_chain,
        }


def _chunks_from_vector_store(vector_store: VectorStore) -> List[SemanticChunk]:
    results = vector_store.collection.get(include=["documents", "metadatas"])
    documents = results.get("documents", []) or []
    metadatas = results.get("metadatas", []) or []

    chunks: List[SemanticChunk] = []
    for document, metadata in zip(documents, metadatas):
        if document is None or metadata is None:
            continue

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

        chunks.append(
            SemanticChunk(
                content=str(document),
                page_numbers=page_numbers,
                bbox_bounds=bbox,
                section_context=str(metadata.get("section_context", "Unknown")).strip() or "Unknown",
                token_count=int(metadata.get("token_count", max(1, len(str(document).split())))),
                content_hash=str(metadata.get("content_hash", "")).strip(),
            )
        )

    return chunks


def run_qa(query: str) -> QAWithProvenance:
    vector_store = VectorStore(persist_directory=".refinery/vector_db", collection_name="semantic_chunks")
    indexed_chunks = _chunks_from_vector_store(vector_store)

    page_index_builder = PageIndexBuilder(persistence_path=".refinery/page_index.json")
    if indexed_chunks:
        page_index_builder.build_tree(indexed_chunks, persist=True)

    agent = QAGraphAgent(page_index_builder=page_index_builder, vector_store=vector_store)
    return agent.ask(query)
