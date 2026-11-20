


import re

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

import config
from src.db import get_connection
from src.embedding import create_embedding


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, chunk_size, conn=None):
        conn = conn or get_connection()
        rows = conn.execute(
            "SELECT chunk_id, source_url, title, text FROM chunks WHERE chunk_size = ?",
            (chunk_size,),
        ).fetchall()
        self.chunks = [
            {"chunk_id": r[0], "source_url": r[1], "title": r[2], "text": r[3]}
            for r in rows
        ]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def search(self, query, k=5):
        scores = self.bm25.get_scores(tokenize(query))
        best = np.argsort(scores)[::-1][:k]

        results = []
        for i in best:
            chunk = dict(self.chunks[i])
            chunk["score"] = float(scores[i])
            results.append(chunk)
        return results


class HybridRetriever:

    def __init__(self, chunk_size, bm25=None, dense=None):
        self.bm25 = bm25 or BM25Retriever(chunk_size)
        self.dense = dense or DenseRetriever(chunk_size)

    def search(self, query, k=5):
        pool = 20
        points = {}
        chunks = {}
        for results in (self.bm25.search(query, pool), self.dense.search(query, pool)):
            for rank, chunk in enumerate(results):
                cid = chunk["chunk_id"]
                # reciprocal rank fusion
                points[cid] = points.get(cid, 0) + 1 / (60 + rank + 1)
                chunks[cid] = chunk

        best_ids = sorted(points, key=points.get, reverse=True)[:k]
        results = []
        for cid in best_ids:
            chunk = dict(chunks[cid])
            chunk["score"] = points[cid]
            results.append(chunk)
        return results


class DenseRetriever:
    def __init__(self, chunk_size, embed_function=create_embedding, client=None):
        client = client or chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.collection = client.get_collection(f"chunks_{chunk_size}")
        self.embed_function = embed_function

    def search(self, query, k=5):
        query_vector = self.embed_function([query])[0]
        found = self.collection.query(query_embeddings=[query_vector], n_results=k)

        results = []
        for i in range(len(found["ids"][0])):
            meta = found["metadatas"][0][i]
            results.append({
                "chunk_id": found["ids"][0][i],
                "source_url": meta["source_url"],
                "title": meta["title"],
                "text": found["documents"][0][i],

                "score": 1 - found["distances"][0][i],
            })
        return results
