import requests

import config


def generate(prompt, json_mode=False, model=None):
    body = {"model": model or config.LLM_MODEL, "prompt": prompt, "stream": False}
    if json_mode:
        body["format"] = "json"

    r = requests.post(config.OLLAMA_URL + "/api/generate", json=body, timeout=600)
    data = r.json()

    if "response" in data:
        return data["response"]
    raise ValueError(f"Ollama error: {data}")
