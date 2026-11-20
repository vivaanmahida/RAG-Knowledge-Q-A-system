


import sqlite3

import config


def get_connection(path=None, check_same_thread=True):

    return sqlite3.connect(path or config.SQLITE_PATH, check_same_thread=check_same_thread)


def create_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT PRIMARY KEY,
            source_url  TEXT,
            title       TEXT,
            scraped_at  TEXT,
            chunk_size  INTEGER,
            chunk_index INTEGER,
            text        TEXT
        )
    """)
    conn.commit()


def chunk_exists(conn, chunk_id):
    row = conn.execute(
        "SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    return row is not None


def save_chunk(conn, chunk):

    conn.execute(
        """INSERT OR IGNORE INTO chunks
           (chunk_id, source_url, title, scraped_at, chunk_size, chunk_index, text)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (chunk["chunk_id"], chunk["source_url"], chunk["title"],
         chunk["scraped_at"], chunk["chunk_size"], chunk["chunk_index"],
         chunk["text"]),
    )
    conn.commit()


def count_chunks(conn, chunk_size):
    return conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE chunk_size = ?", (chunk_size,)
    ).fetchone()[0]
