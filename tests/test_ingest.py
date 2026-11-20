import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.chunking import chunk_text, make_chunk_id
from src.db import get_connection, create_tables, save_chunk, count_chunks
from src.ingest import build_chunks


def test_chunk_size_is_respected():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert all(len(c.split()) <= 200 for c in chunks)


def test_overlap_between_chunks():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=200, overlap=40)

    assert chunks[0].split()[-40:] == chunks[1].split()[:40]


def test_short_text_gives_one_chunk():
    assert len(chunk_text("just a few words", 200, 40)) == 1


def test_same_input_gives_same_id():
    assert make_chunk_id("http://a.com", "hello") == make_chunk_id("http://a.com", "hello")


def test_different_text_gives_different_id():
    assert make_chunk_id("http://a.com", "hello") != make_chunk_id("http://a.com", "bye")


def test_saving_same_chunk_twice_gives_no_duplicate():
    conn = get_connection(":memory:")
    create_tables(conn)
    pages = [{"url": "http://a.com", "title": "A", "scraped_at": "2026-01-01",
              "text": " ".join(f"w{i}" for i in range(300))}]
    chunks = build_chunks(pages, 200)

    for _ in range(2):
        for c in chunks:
            save_chunk(conn, c)

    assert count_chunks(conn, 200) == len(chunks)
