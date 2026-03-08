from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from src.agents.indexer import PageIndexBuilder
from src.agents.qa_agent import QAGraphAgent, _chunks_from_vector_store
from src.agents.vector_store import VectorStore


def _normalize_provenance(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


@st.cache_resource(show_spinner=False)
def load_agent() -> QAGraphAgent:
    vector_store = VectorStore(persist_directory=".refinery/vector_db", collection_name="semantic_chunks")
    indexed_chunks = _chunks_from_vector_store(vector_store)

    page_index_builder = PageIndexBuilder(persistence_path=".refinery/page_index.json")
    if indexed_chunks:
        page_index_builder.build_tree(indexed_chunks, persist=True)

    return QAGraphAgent(page_index_builder=page_index_builder, vector_store=vector_store)


def render_provenance_entries(provenance_chain: List[Dict[str, Any]], where: Any) -> None:
    if not provenance_chain:
        where.caption("No provenance citations returned.")
        return

    for idx, item in enumerate(provenance_chain, start=1):
        document_name = str(item.get("document_name", "Unknown document")).strip() or "Unknown document"
        content_hash = str(item.get("content_hash", "")).strip()
        section_context = str(item.get("section_context", "Unknown")).strip() or "Unknown"
        page_number = int(item.get("page_number", 0) or 0)

        bbox = item.get("bbox_bounds", {}) if isinstance(item.get("bbox_bounds", {}), dict) else {}
        x1 = float(bbox.get("x1", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        x2 = float(bbox.get("x2", 0.0))
        y2 = float(bbox.get("y2", 0.0))

        where.markdown(
            f"**{idx}. {document_name}**  \n"
            f"Page: `{page_number}`  \n"
            f"Section: `{section_context}`  \n"
            f"Hash: `{content_hash}`  \n"
            f"BBox: `({x1:.3f}, {y1:.3f}) -> ({x2:.3f}, {y2:.3f})`"
        )


def run_standard_query(agent: QAGraphAgent, query: str) -> Dict[str, Any]:
    result = agent.ask(query)
    return {
        "mode": "qa",
        "answer": str(result.get("answer", "")).strip(),
        "status": "",
        "reasoning": "",
        "provenance_chain": _normalize_provenance(result.get("provenance_chain", [])),
    }


def run_claim_verification(agent: QAGraphAgent, claim: str) -> Dict[str, Any]:
    result = agent.audit_claim(claim)
    status = str(result.get("status", "UNVERIFIABLE")).strip().upper()
    reasoning = str(result.get("reasoning", "")).strip()

    answer_lines = [f"Status: {status}"]
    if reasoning:
        answer_lines.append(f"Reasoning: {reasoning}")

    return {
        "mode": "audit",
        "answer": "\n".join(answer_lines),
        "status": status,
        "reasoning": reasoning,
        "provenance_chain": _normalize_provenance(result.get("provenance_chain", [])),
    }


def ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_provenance" not in st.session_state:
        st.session_state.last_provenance = []


def render_chat_history() -> None:
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        content = str(message.get("content", "")).strip()
        with st.chat_message(role):
            st.markdown(content if content else "(empty)")

            if role == "assistant":
                provenance_chain = _normalize_provenance(message.get("provenance_chain", []))
                with st.expander("Provenance", expanded=False):
                    render_provenance_entries(provenance_chain, st)


def render_sidebar() -> None:
    st.sidebar.header("Provenance")
    last_chain = _normalize_provenance(st.session_state.get("last_provenance", []))
    if not last_chain:
        st.sidebar.caption("Ask a question or verify a claim to view citations.")
        return

    with st.sidebar.expander("Latest response evidence", expanded=True):
        render_provenance_entries(last_chain, st)


def main() -> None:
    st.set_page_config(page_title="Document Intelligence Refinery", layout="wide")
    st.title("Document Intelligence Refinery")
    st.caption("Ask grounded questions and verify claims with provenance-aware evidence.")

    ensure_session_state()

    try:
        agent = load_agent()
    except Exception as exc:
        st.error(f"Failed to initialize QA agent: {exc}")
        st.info("Ensure indexing has been run and dependencies are installed, then refresh.")
        return

    render_sidebar()
    render_chat_history()

    st.subheader("Ask Or Audit")
    query = st.text_area(
        "Question or claim",
        key="query_input",
        placeholder="Example: What does the document say about Q4 revenue guidance?",
        height=90,
    ).strip()

    ask_col, verify_col = st.columns(2)
    ask_clicked = ask_col.button("Ask", use_container_width=True)
    verify_clicked = verify_col.button("Verify Claim", use_container_width=True)

    if not ask_clicked and not verify_clicked:
        return

    if not query:
        st.warning("Enter a question or claim before submitting.")
        return

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Running audit..." if verify_clicked else "Generating answer..."):
        if verify_clicked:
            response = run_claim_verification(agent, query)
        else:
            response = run_standard_query(agent, query)

    assistant_content = response.get("answer", "")
    provenance_chain = _normalize_provenance(response.get("provenance_chain", []))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": str(assistant_content).strip(),
            "provenance_chain": provenance_chain,
        }
    )
    st.session_state.last_provenance = provenance_chain

    st.rerun()


if __name__ == "__main__":
    main()
