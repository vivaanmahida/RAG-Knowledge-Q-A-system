


import hashlib
import json
import re
from datetime import datetime

import numpy as np

import config
from src.embedding import create_embedding
from src.llm import generate
from src.rag import answer_question


def normalize(question):
    text = re.sub(r"[^\w\s]", " ", question.lower())
    return " ".join(text.split())


def tokens(text):
    return re.findall(r"\w+", text.lower())


def extract_numbers(question):
    return set(re.findall(r"\d+(?:\.\d+)?", question))


NEGATION_WORDS = {"not", "no", "never", "without", "cannot", "none", "cant", "dont", "doesnt", "isnt", "wont"}


def has_negation(question):
    text = question.lower().replace("’", "'")
    for word in re.findall(r"[a-z']+", text):
        if word in NEGATION_WORDS or word.endswith("n't"):
            return True
    return False


def build_entity_vocab(conn, chunk_size=None, max_fraction=0.15):
    rows = conn.execute("SELECT text FROM chunks WHERE chunk_size = ?",
                        (chunk_size or config.RAG_CHUNK_SIZE,)).fetchall()
    doc_freq = {}
    for (text,) in rows:
        for word in set(tokens(text)):
            doc_freq[word] = doc_freq.get(word, 0) + 1
    limit = max_fraction * len(rows)
    return {word for word, count in doc_freq.items() if count <= limit}


def key_terms(question, vocab):
    return {w for w in tokens(question) if w in vocab and not w.isdigit()}


def is_safe_match(question_a, question_b, vocab, guards=("numbers", "negation", "terms")):
    if "numbers" in guards and extract_numbers(question_a) != extract_numbers(question_b):
        return False
    if "negation" in guards and has_negation(question_a) != has_negation(question_b):
        return False
    if "terms" in guards and key_terms(question_a, vocab) != key_terms(question_b, vocab):
        return False
    return True


TITLE_STOPWORDS = {"the", "element", "html", "css", "javascript", "a", "an", "and", "of", "array",
                   "prototype", "data", "statement", "statements", "global", "object", "objects",
                   "reference", "property"}


def build_topic_vocab(conn):
    titles = conn.execute("SELECT DISTINCT title FROM chunks").fetchall()
    vocab = set()
    for (title,) in titles:
        for word in tokens(title):
            if word not in TITLE_STOPWORDS and len(word) >= 2 and not word.isdigit():
                vocab.add(word)
    return vocab


GUARD_MODES = ["none", "numbers+negation", "topic", "rare"]


def guard_setup(conn, mode):
    if mode == "none":
        return set(), ()
    if mode == "numbers+negation":
        return set(), ("numbers", "negation")
    if mode == "topic":
        return build_topic_vocab(conn), ("numbers", "negation", "terms")
    if mode == "rare":
        return build_entity_vocab(conn), ("numbers", "negation", "terms")
    raise ValueError(f"unknown guard mode: {mode}")


def cosine(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def kb_version(conn, chunk_size=None):
    ids = conn.execute("SELECT chunk_id FROM chunks WHERE chunk_size = ? ORDER BY chunk_id",
                       (chunk_size or config.RAG_CHUNK_SIZE,)).fetchall()
    return hashlib.sha256("".join(i[0] for i in ids).encode()).hexdigest()[:12]


class QueryCache:
    def __init__(self, conn, embed_function=create_embedding, threshold=None,
                 vocab=None, version="v1", guards=None):
        self.conn = conn
        self.embed_function = embed_function
        self.threshold = config.CACHE_THRESHOLD if threshold is None else threshold
        self.vocab = vocab or set()
        self.version = version
        self.guards = ("numbers", "negation", "terms") if guards is None else tuple(guards)
        self.stats = {"exact": 0, "semantic": 0, "miss": 0}
        self._last_vector = None

        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                cache_key    TEXT,
                kb_version   TEXT,
                question     TEXT,
                embedding    TEXT,
                result_json  TEXT,
                created_at   TEXT,
                PRIMARY KEY (cache_key, kb_version)
            )""")
        conn.commit()

    def get(self, question):
        key = normalize(question)


        row = self.conn.execute(
            "SELECT result_json FROM query_cache WHERE cache_key = ? AND kb_version = ?",
            (key, self.version)).fetchone()
        if row:
            self.stats["exact"] += 1
            return json.loads(row[0]), "exact", 1.0


        rows = self.conn.execute(
            "SELECT question, embedding, result_json FROM query_cache WHERE kb_version = ?",
            (self.version,)).fetchall()
        vector = self.embed_function([question])[0]
        self._last_vector = (key, vector)

        candidates = sorted(
            ((cosine(vector, json.loads(emb)), q, res) for q, emb, res in rows), reverse=True)
        best_similarity = candidates[0][0] if candidates else 0.0

        for similarity, cached_question, result_json in candidates:
            if similarity < self.threshold:
                break
            if self.guards and not is_safe_match(question, cached_question, self.vocab, self.guards):
                continue
            self.stats["semantic"] += 1
            return json.loads(result_json), "semantic", similarity

        self.stats["miss"] += 1
        return None, None, best_similarity

    def put(self, question, result):
        if result["refused"]:
            return False
        key = normalize(question)
        if self._last_vector and self._last_vector[0] == key:
            vector = self._last_vector[1]
        else:
            vector = self.embed_function([question])[0]
        self.conn.execute(
            "INSERT OR REPLACE INTO query_cache VALUES (?, ?, ?, ?, ?, ?)",
            (key, self.version, question, json.dumps(list(map(float, vector))),
             json.dumps(result), datetime.now().isoformat()))
        self.conn.commit()
        return True

    def clear(self):
        self.conn.execute("DELETE FROM query_cache")
        self.conn.commit()


def answer_with_cache(question, retriever, cache, classifier=None, llm=generate):
    if question.strip():
        cached, kind, similarity = cache.get(question)
        if cached is not None:
            cached["cache"] = {"hit": True, "kind": kind, "similarity": round(similarity, 3)}
            return cached
    else:
        similarity = 0.0

    result = answer_question(question, retriever, llm=llm, classifier=classifier)
    if question.strip():
        cache.put(question, result)
    result["cache"] = {"hit": False, "kind": None, "similarity": round(similarity, 3)}
    return result
