import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb

from src.db import get_connection, create_tables, save_chunk
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

TEXTS = [
    "flexbox lets you arrange items in a row or column inside a flex container",
    "the margin property adds space outside an element border",
    "the map method creates a new array by calling a function on every item",
    "the const keyword declares a variable that cannot be reassigned",
]


def make_chunks():
    return [{"chunk_id": f"id{i}", "source_url": f"http://x/{i}", "title": f"T{i}",
             "scraped_at": "2026-01-01", "chunk_size": 200, "chunk_index": 0, "text": t}
            for i, t in enumerate(TEXTS)]


def test_bm25_finds_the_right_chunk_first():
    conn = get_connection(":memory:")
    create_tables(conn)
    for c in make_chunks():
        save_chunk(conn, c)

    retriever = BM25Retriever(200, conn=conn)
    results = retriever.search("how does the map method work on an array", k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "id2"


def fake_embed(texts):
    words = ["flex", "margin", "map", "const"]
    return [[float(w in t.lower()) + 0.01 for w in words] for t in texts]


def test_dense_retriever_returns_k_results_with_scores():
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("chunks_200", metadata={"hnsw:space": "cosine"})
    chunks = make_chunks()
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=fake_embed([c["text"] for c in chunks]),
        documents=[c["text"] for c in chunks],
        metadatas=[{"source_url": c["source_url"], "title": c["title"]} for c in chunks],
    )

    retriever = DenseRetriever(200, embed_function=fake_embed, client=client)
    results = retriever.search("margin", k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "id1"
    assert results[0]["score"] >= results[1]["score"]


class FakeList:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query, k=5):
        return [{"chunk_id": i, "source_url": "u", "title": "t", "text": i, "score": 1.0}
                for i in self.ids][:k]


def test_hybrid_prefers_chunks_both_retrievers_like():
    bm25 = FakeList(["a", "b", "c"])
    dense = FakeList(["b", "d", "a"])
    hybrid = HybridRetriever(200, bm25=bm25, dense=dense)
    results = hybrid.search("anything", k=3)
    ids = [r["chunk_id"] for r in results]
    assert set(ids[:2]) == {"a", "b"}
    assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]


def test_hybrid_has_no_duplicate_chunks():
    hybrid = HybridRetriever(200, bm25=FakeList(["a", "b"]), dense=FakeList(["a", "b"]))
    ids = [r["chunk_id"] for r in hybrid.search("q", k=5)]
    assert len(ids) == len(set(ids)) == 2
