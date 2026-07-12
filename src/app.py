"""
app.py
------
Streamlit UI for IITB Insti-Assist (Academic scope).

Implements the required web interface, plus all four bonus/stretch goals:
  - Multi-turn conversational memory   -> st.session_state["history"]
  - Citation highlighting               -> expandable cards with exact chunk text
  - Live PDF upload                     -> sidebar uploader, chunked+embedded on the fly
  - Confidence / grounded indicator     -> colored badge from RagAnswer.confidence
"""
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingest import load_pdf_and_chunk
from embed_store import VectorStore
from rag_pipeline import RagPipeline

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(HERE, "index_store")

st.set_page_config(page_title="IITB Insti-Assist — Academic", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Loading knowledge base index...")
def load_store():
    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        st.error(
            "No index found. Run `python build_index.py` from the project "
            "root first to build the FAISS index from the documents in `data/`."
        )
        st.stop()
    return VectorStore.load(INDEX_DIR)


def get_pipeline(store, provider, api_key):
    return RagPipeline(store, provider=provider, api_key=api_key)


CONFIDENCE_BADGE = {
    "grounded": ("✅ Grounded", "green"),
    "low_confidence": ("⚠️ Low confidence", "orange"),
    "not_grounded": ("❌ Not grounded / not in knowledge base", "red"),
}

st.title("🎓 IITB Insti-Assist")
st.caption("Scope: **Academic Assistant** — course registration, grading policy, "
           "academic calendar, exam rules, probation, branch change/minors/honours.")

# ---------------- sidebar: provider + API key + live PDF upload -----------
with st.sidebar:
    st.header("Settings")
    provider_label = st.radio(
        "LLM provider",
        options=["Claude (Anthropic)", "Gemini (Google)"],
        help="Retrieval is identical either way (local embeddings + FAISS); "
             "this only picks which model generates the final answer.",
    )
    provider = "anthropic" if provider_label.startswith("Claude") else "gemini"
    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "GEMINI_API_KEY"
    api_key_input = st.text_input(
        f"{provider_label} API key",
        type="password",
        value=os.environ.get(env_var, ""),
        help=f"Reads from the {env_var} environment variable by default. "
             "You can override it here for this session only (not stored).",
    )
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=2, max_value=8, value=4)

    st.divider()
    st.subheader("📄 Add a document (bonus)")
    st.caption(
        "Upload an extra IITB-related PDF (e.g. a specific department's "
        "grading circular) to extend the knowledge base for this session only."
    )
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf is not None:
        if st.session_state.get("last_uploaded_name") != uploaded_pdf.name:
            with st.spinner(f"Chunking & embedding {uploaded_pdf.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.read())
                    tmp_path = tmp.name
                new_chunks = load_pdf_and_chunk(tmp_path)
                st.session_state["store"].add(new_chunks)
                st.session_state["last_uploaded_name"] = uploaded_pdf.name
            st.success(f"Added {len(new_chunks)} chunks from {uploaded_pdf.name} "
                       f"to this session's index.")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state["history"] = []
        st.rerun()

# ---------------- session state init --------------------------------------
if "store" not in st.session_state:
    st.session_state["store"] = load_store()
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of {"role", "content"} for the LLM
if "display_log" not in st.session_state:
    st.session_state["display_log"] = []  # list of dicts for rendering (incl. citations)

pipeline = get_pipeline(st.session_state["store"], provider, api_key_input or None)

# ---------------- render past turns ----------------------------------------
for turn in st.session_state["display_log"]:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        label, color = CONFIDENCE_BADGE[turn["confidence"]]
        st.markdown(f":{color}[{label}]  (top similarity: {turn['top_score']:.2f})")
        st.markdown(turn["answer"])
        if turn["used_chunks"]:
            with st.expander(f"📚 Sources used ({len(turn['used_chunks'])})"):
                for chunk, score in turn["used_chunks"]:
                    st.markdown(
                        f"**{chunk.doc_title} → {chunk.section}** "
                        f"(similarity: {score:.2f})"
                    )
                    st.code(chunk.text, language=None)

# ---------------- chat input -------------------------------------------
question = st.chat_input("Ask about IITB academics — grading, registration, exams, calendar...")
if question:
    if not (api_key_input or os.environ.get(env_var)):
        st.error(f"Please enter a {provider_label} API key in the sidebar first.")
    else:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant policy chunks and generating a grounded answer..."):
                result = pipeline.answer(
                    question,
                    history=st.session_state["history"],
                    top_k=top_k,
                )
            label, color = CONFIDENCE_BADGE[result.confidence]
            st.markdown(f":{color}[{label}]  (top similarity: {result.top_score:.2f})")
            st.markdown(result.answer)
            if result.used_chunks:
                with st.expander(f"📚 Sources used ({len(result.used_chunks)})"):
                    for chunk, score in result.used_chunks:
                        st.markdown(
                            f"**{chunk.doc_title} → {chunk.section}** "
                            f"(similarity: {score:.2f})"
                        )
                        st.code(chunk.text, language=None)

        # persist for multi-turn memory + re-render on next run
        st.session_state["history"].append({"role": "user", "content": question})
        st.session_state["history"].append({"role": "assistant", "content": result.answer})
        st.session_state["display_log"].append({
            "question": question,
            "answer": result.answer,
            "confidence": result.confidence,
            "top_score": result.top_score,
            "used_chunks": result.used_chunks,
        })
