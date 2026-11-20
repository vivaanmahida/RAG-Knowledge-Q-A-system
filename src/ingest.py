


import json
import logging
import os

import chromadb

import config
from src.chunking import chunk_text, make_chunk_id
from src.db import get_connection, create_tables, chunk_exists, save_chunk, count_chunks
from src.embedding import create_embedding

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_pages():
    pages = []
    for file_name in sorted(os.listdir(config.RAW_FOLDER)):
        if file_name.endswith(".json"):
            with open(os.path.join(config.RAW_FOLDER, file_name)) as f:
                pages.append(json.load(f))
    return pages


def build_chunks(pages, chunk_size):
    all_chunks = []
    for page in pages:
        pieces = chunk_text(page["text"], chunk_size, config.CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": make_chunk_id(page["url"], piece),
                "source_url": page["url"],
                "title": page["title"],
                "scraped_at": page["scraped_at"],
                "chunk_size": chunk_size,
                "chunk_index": i,
                "text": piece,
            })
    return all_chunks


def ingest(chunk_size, conn, collection, embed_function=create_embedding):
    pages = load_pages()
    chunks = build_chunks(pages, chunk_size)


    new_chunks = [c for c in chunks if not chunk_exists(conn, c["chunk_id"])]
    skipped = len(chunks) - len(new_chunks)


    for start in range(0, len(new_chunks), config.EMBED_BATCH_SIZE):
        batch = new_chunks[start:start + config.EMBED_BATCH_SIZE]
        try:
            vectors = embed_function([c["text"] for c in batch])
        except Exception as e:
            logging.error(f"Embedding failed for batch at {start}: {e}")
            continue


        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[{"source_url": c["source_url"], "title": c["title"]} for c in batch],
        )
        for c in batch:
            save_chunk(conn, c)

    logging.info(
        f"chunk_size={chunk_size}: {len(new_chunks)} new, {skipped} skipped (already stored)"
    )


def main():
    os.makedirs("data", exist_ok=True)
    conn = get_connection()
    create_tables(conn)
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    for size in config.CHUNK_SIZES:

        collection = client.get_or_create_collection(
            name=f"chunks_{size}", metadata={"hnsw:space": "cosine"}
        )
        ingest(size, conn, collection)
        logging.info(f"Total in SQLite for size {size}: {count_chunks(conn, size)}")


if __name__ == "__main__":
    main()
