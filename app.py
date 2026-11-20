

import os

import streamlit as st

import config
from src.cache import QueryCache, answer_with_cache, guard_setup, kb_version
from src.classifier import ScopeClassifier
from src.db import get_connection
from src.retrieval import DenseRetriever

st.set_page_config(page_title="Web Dev Knowledge Assistant", page_icon="📚")
st.title("📚 Web Dev Knowledge Assistant")
st.caption("Answers come only from 20 MDN pages (HTML, CSS, JavaScript basics).")


@st.cache_resource
def load_retriever():
    return DenseRetriever(config.RAG_CHUNK_SIZE)


@st.cache_resource
def load_classifier():

    if os.path.exists(config.CLASSIFIER_PATH):
        return ScopeClassifier()
    return None


@st.cache_resource
def load_cache():
    conn = get_connection(check_same_thread=False)
    vocab, guards = guard_setup(conn, config.CACHE_GUARD_MODE)
    return QueryCache(conn, vocab=vocab, guards=guards, version=kb_version(conn))


cache = load_cache()
st.sidebar.write(f"Cache hits: exact {cache.stats['exact']}, similar {cache.stats['semantic']}, "
                 f"misses {cache.stats['miss']}")
if st.sidebar.button("Clear cache"):
    cache.clear()
    st.sidebar.success("Cache cleared.")

classifier = load_classifier()
if classifier is None:
    st.sidebar.info("Scope classifier not trained yet: using the similarity check only.")
else:
    st.sidebar.success("Scope classifier loaded.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources, retrieved, best_score):
    if sources:
        with st.expander("📎 Sources cited"):
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['source_url']})")
    if retrieved:
        with st.expander(f"🔍 All retrieved chunks (best similarity {best_score:.2f})"):
            for s in retrieved:
                st.markdown(f"**{s['title']}** — score {s['score']:.2f}")
                st.write(s["text"][:300] + "...")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            show_sources(message["sources"], message["retrieved"], message["best_score"])

question = st.chat_input("Ask a question about HTML, CSS or JavaScript...")

if question:
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer_with_cache(question, load_retriever(), cache, classifier=classifier)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        if result["refused"]:
            st.warning(result["answer"])
        else:
            st.write(result["answer"])
        if result["warning"]:
            st.caption("⚠️ " + result["warning"])
        if result["cache"]["hit"]:
            if result["cache"]["kind"] == "exact":
                st.caption("⚡ Served from cache (same question asked before)")
            else:
                st.caption(f"⚡ Served from cache (similar question, similarity {result['cache']['similarity']:.2f})")
        if result["scope_probability"] is not None:
            st.caption(f"Scope classifier: {result['scope_probability']:.2f} probability the question is in scope")
        show_sources(result["sources"], result["retrieved"], result["best_score"])

    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"], "sources": result["sources"],
        "retrieved": result["retrieved"], "best_score": result["best_score"],
    })
