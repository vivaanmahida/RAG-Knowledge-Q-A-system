import hashlib


def chunk_text(text, chunk_size=200, overlap=40):
    # split by words, each chunk repeats the last few words of the previous one
    words = text.split()
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break

    return chunks


def make_chunk_id(url, text):
    # same url + text gives the same id, so re-running adds no duplicates
    return hashlib.sha256((url + text).encode("utf-8")).hexdigest()
