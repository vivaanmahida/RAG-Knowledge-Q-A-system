


import json
import os
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

import config

BASE = "https://developer.mozilla.org/en-US/docs/Web/"


PAGES = [
    "HTML/Element/a", "HTML/Element/div", "HTML/Element/img",
    "HTML/Element/form", "HTML/Element/table", "HTML/Element/ul",
    "HTML/Element/input", "HTML/Element/button",
    "CSS/color", "CSS/display", "CSS/margin", "CSS/padding",
    "CSS/position", "CSS/font-size", "CSS/flex",
    "JavaScript/Reference/Statements/let",
    "JavaScript/Reference/Statements/const",
    "JavaScript/Reference/Statements/function",
    "JavaScript/Reference/Global_Objects/Array/map",
    "JavaScript/Reference/Global_Objects/Array/filter",
]


def fetch_page(url):
    headers = {"User-Agent": "Mozilla/5.0 (student project)"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return response.text


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else "No title"


    main = soup.find("main") or soup
    for tag in main(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()

    text = main.get_text(separator=" ")
    text = " ".join(text.split())
    return title, text


def scrape_all():
    os.makedirs(config.RAW_FOLDER, exist_ok=True)

    for page in PAGES:
        url = BASE + page
        try:
            html = fetch_page(url)
            title, text = clean_html(html)
        except Exception as e:
            print("FAILED:", url, e)
            continue

        data = {
            "url": url,
            "title": title,
            "scraped_at": str(date.today()),
            "text": text,
        }
        file_name = page.replace("/", "_") + ".json"
        with open(os.path.join(config.RAW_FOLDER, file_name), "w") as f:
            json.dump(data, f)

        print("saved", file_name, "-", len(text.split()), "words")
        time.sleep(1)


if __name__ == "__main__":
    scrape_all()
