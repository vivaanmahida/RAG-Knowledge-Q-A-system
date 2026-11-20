import requests

import config


def create_embedding(text_list):
    r = requests.post(
        config.OLLAMA_URL + "/api/embed",
        json={"model": config.EMBED_MODEL, "input": text_list},
        timeout=300,
    )
    data = r.json()

    if "embeddings" in data:
        return data["embeddings"]
    raise ValueError(f"Unexpected response from Ollama: {data}")
