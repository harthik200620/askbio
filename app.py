"""
AskBio - Streamlit front end. Takes a question, runs retrieve() then
generate_answer(), and shows the grounded answer with its PubMed citations.

    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

# On Streamlit Cloud there's no .env file -- mirror any secrets into the
# environment so config.py (which reads os.getenv) sees QDRANT_URL, GEMINI_API_KEY,
# etc. Locally there's no secrets file, so this is a harmless no-op.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

import config

st.set_page_config(page_title="AskBio - biomedical RAG", page_icon="🧬", layout="centered")


# cache_resource keeps the models/indexes loaded once instead of rebuilding them
# on every rerun. Imports are inside so a load error surfaces in the app.
@st.cache_resource(show_spinner=False)
def _pipeline():
    import retrieve
    import generate
    return retrieve.retrieve, generate.generate_answer


st.title("🧬 AskBio")
st.caption(
    "Answers biomedical questions from real PubMed research - with citations - "
    "and refuses to guess when the literature doesn't cover the question."
)

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "1. **Hybrid retrieval** - meaning search (embeddings) + keyword search (BM25)\n"
        "2. **Reranking** - a cross-encoder re-sorts the best passages to the top\n"
        "3. **Grounded answer** - the LLM answers *only* from those passages and cites PMIDs\n"
        "4. **Guardrail** - if the evidence is weak, it abstains instead of hallucinating"
    )
    st.divider()
    st.caption(
        f"embeddings: `{config.EMBED_BACKEND}`  |  llm: `{config.LLM_BACKEND}`  |  "
        f"corpus target: {config.CORPUS_SUBSET_SIZE:,}"
    )


EXAMPLE = "What is the link between the gut microbiome and immunity?"
# Using a form so the input and submit arrive together (Enter submits too).
with st.form("ask_form"):
    question = st.text_input("Ask a biomedical question:", placeholder=EXAMPLE)
    ask = st.form_submit_button("Ask AskBio", type="primary")

if ask and question.strip():
    try:
        retrieve_fn, generate_fn = _pipeline()
    except Exception as exc:  # noqa: BLE001 - surface load errors to the user
        st.error(f"Pipeline failed to load: {exc}")
        st.stop()

    with st.spinner("Searching PubMed passages..."):
        passages = retrieve_fn(question)
    with st.spinner("Reading the evidence and writing a grounded answer..."):
        result = generate_fn(question, passages)

    st.divider()
    if result["abstained"]:
        st.warning(result["answer"])  # abstained -> show it as a warning
    else:
        st.markdown(result["answer"])

    citations = result.get("citations", [])
    if citations:
        st.subheader(f"Sources ({len(citations)})")
        for c in citations:
            label = f"PMID {c['pmid']} - {c['title'] or 'PubMed article'}"
            with st.expander(label):
                st.markdown(f"[Open on PubMed]({c['url']})")

    with st.expander(f"Show all {len(passages)} retrieved passages"):
        for i, p in enumerate(passages, 1):
            preview = p["text"][:400] + ("…" if len(p["text"]) > 400 else "")
            st.markdown(
                f"**[{i}] PMID {p['pmid']}** · score {p['score']:.3f}  \n{preview}"
            )

elif ask:
    st.info("Type a question first.")
