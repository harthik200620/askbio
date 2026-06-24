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


@st.cache_resource(show_spinner=False)
def _pipeline():
    import retrieve
    import generate
    return retrieve.retrieve, generate.generate_answer


st.title("🧬 AskBio")
st.caption(
    "Answers biomedical questions from real PubMed research — with citations — "
    "and refuses to guess when the literature doesn't cover the question."
)
st.info(
    "This demo indexes PubMed literature on **cardiovascular disease**, **type 2 diabetes**, "
    "**common pain and fever medications** (aspirin, ibuprofen, paracetamol), and "
    "**bacterial infections and antibiotics** — try the examples below or ask your own.",
    icon="📚",
)

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "1. **Hybrid retrieval** — meaning search (embeddings) + keyword search (BM25)\n"
        "2. **Reranking** — a cross-encoder re-sorts the best passages to the top\n"
        "3. **Grounded answer** — the LLM answers *only* from those passages and cites PMIDs\n"
        "4. **Guardrail** — if the evidence is weak, it abstains instead of hallucinating"
    )
    st.divider()

    with st.expander("📊 Evaluation scores", expanded=False):
        st.markdown("Measured on 50 PubMedQA expert questions (ragas + accuracy):")
        col1, col2 = st.columns(2)
        col1.metric("Faithfulness", "0.82", help="Every claim in the answer is supported by the retrieved passages. Measured by ragas: it extracts each claim and checks if the passages entail it.")
        col2.metric("Answer relevancy", "0.78", help="The answer actually addresses the question asked. Measured by ragas: it generates reverse questions from the answer and checks similarity to the original.")
        col1.metric("Context precision", "0.85", help="Useful passages were ranked near the top of the retrieved list. Measured by ragas: checks what fraction of the top-k passages are actually relevant.")
        col2.metric("Accuracy", "0.60", help="Yes/no/maybe label accuracy on PubMedQA. Abstentions are excluded from the denominator — a calibrated 'I don't know' is better than a wrong guess.")
        st.caption("Abstention rate ~40% — the system refuses to answer when evidence is unclear.")

    st.divider()
    if config.CORPUS_TOPICS:
        _total = len(config.CORPUS_TOPICS) * config.CORPUS_PER_TOPIC_TARGET
        _corpus_label = f"~{_total:,} snippets · {len(config.CORPUS_TOPICS)} topics"
    else:
        _corpus_label = f"{config.CORPUS_SUBSET_SIZE:,} snippets"
    st.caption(
        f"embeddings: `{config.EMBED_BACKEND}`  |  llm: `{config.LLM_BACKEND}`  |  "
        f"corpus: {_corpus_label}"
    )


# --- Example questions ---
EXAMPLES = [
    "How does aspirin affect the risk of heart attack?",
    "What are first-line treatments for type 2 diabetes?",
    "Does ibuprofen reduce fever in children?",
    "How do beta-blockers lower blood pressure?",
    "What antibiotics are used to treat bacterial pneumonia?",
]

if "question" not in st.session_state:
    st.session_state.question = ""

st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state.question = example

# --- Question form ---
with st.form("ask_form"):
    question = st.text_input(
        "Or ask your own question:",
        value=st.session_state.question,
        placeholder="e.g. How does aspirin affect the risk of heart attack?",
    )
    ask = st.form_submit_button("Ask AskBio", type="primary")

if ask and question.strip():
    st.session_state.question = question  # preserve across reruns

    try:
        retrieve_fn, generate_fn = _pipeline()
    except Exception as exc:
        st.error(f"Pipeline failed to load: {exc}")
        st.stop()

    with st.spinner("Searching PubMed passages..."):
        passages = retrieve_fn(question)
    with st.spinner("Reading the evidence and writing a grounded answer..."):
        result = generate_fn(question, passages)

    st.divider()
    if result["abstained"]:
        st.warning(result["answer"])
    else:
        st.markdown(result["answer"])

    citations = result.get("citations", [])
    if citations:
        st.subheader(f"Sources ({len(citations)})")
        for c in citations:
            label = f"PMID {c['pmid']} — {c['title'] or 'PubMed article'}"
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
