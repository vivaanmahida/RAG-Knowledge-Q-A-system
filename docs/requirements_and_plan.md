# Requirements and Execution Plan

Project: **Knowledge Q&A (RAG) MVP** over public web documentation (Problem Statement 1).
Knowledge base: 20 pages of the MDN Web Docs about HTML, CSS and JavaScript basics.

## 1. Problem understanding

The task is to build a question-answering system that answers **only** from a knowledge base that I collect myself, and that behaves safely when the knowledge base cannot answer. A plain chatbot can invent answers. A RAG (Retrieval-Augmented Generation) system reduces this by first *retrieving* relevant passages and then asking a language model to answer *using only those passages* and to cite them.

The system must therefore do five things well: collect and clean data, retrieve the right passage, answer with citations, **refuse** questions it cannot support, and be **measured** honestly (experiments, evaluation, error analysis).

I chose a technical domain I already know (web development basics) so that I can write correct evaluation questions and judge the answers.

## 2. Requirement analysis

### 2.1 Functional requirements

| ID | Requirement |
|---|---|
| FR1 | Collect pages from public web sources and store title, URL and collection date with the text. |
| FR2 | Clean and split text into chunks; running the pipeline again must **not** create duplicates. |
| FR3 | Store chunk text and metadata in a database and chunk embeddings in a vector database. |
| FR4 | Retrieve the most relevant chunks for a question (keyword, dense and hybrid methods compared). |
| FR5 | Generate an answer using only the retrieved chunks, with numbered source citations. |
| FR6 | Refuse (clearly, without inventing) when the knowledge base cannot support an answer. |
| FR7 | Provide a simple user interface to ask questions and inspect the sources. |
| FR8 | Train and evaluate a model: an in-scope / out-of-scope question classifier. |
| FR9 | Cache answers (exact and semantic) without returning wrong answers for different questions. |
| FR10 | Evaluate retrieval and answer quality on a labelled dataset (at least 30 questions) and analyse failures. |

### 2.2 Non-functional requirements

- **Local and free:** all models run locally through Ollama; no API keys or paid services.
- **Reproducible:** fixed random seed, a single `config.py`, scripted data preparation and evaluation.
- **Modular and testable:** one responsibility per file; models can be replaced by fakes in unit tests.
- **Safe failure:** if a layer is unsure, the system refuses instead of guessing.
- **Understandable:** simple code that I can explain line by line.

### 2.3 Assumptions

1. The knowledge base is small (20 pages), so exact search over thousands of chunks is fast enough.
2. Users ask in English about HTML, CSS and JavaScript basics.
3. A single user at a time (a demo, not a production service).
4. Ollama is installed and running on the laptop with `bge-m3` and `llama3.2` pulled.
5. MDN pages are publicly available and may be used for a learning project with attribution.
6. A small local model (3B parameters) is acceptable; its weaknesses will be measured and reported.

### 2.4 Dependencies and constraints

- 20 working hours in total, so scope must be limited (see section 3).
- Laptop hardware: CPU-class inference, so answers take seconds, and large models are not an option.
- Dependency on the Ollama service: if it is not running, embedding and answering both fail.
- Website structure can change; scraped pages are saved as snapshots so later steps do not depend on the live site.
- Small evaluation sets (dozens of questions) mean results are indicative, not statistically strong.

### 2.5 Measurable success criteria (targets set for the MVP)

| Area | Target |
|---|---|
| Ingestion | Running ingestion a second time adds **0** new chunks. |
| Retrieval | Chosen configuration reaches **Recall@5 of at least 0.85** on the evaluation set. |
| Citations | At least **90%** of answered questions contain a citation; correctness is measured separately. |
| Refusal | Clearly unrelated questions are refused; **false refusals on in-scope questions at most 10%**. |
| Classifier | **F1 of at least 0.90** on a held-out test set; also reported on a harder hand-checked set. |
| Cache | **No wrong reuse** of an answer on the labelled pair set. |
| Quality | Unit tests pass; every reported number comes from a script in the repository. |

Targets are compared with actual results in the one-page experiment report.

## 3. Scope and prioritisation

| Priority | Feature | Status |
|---|---|---|
| **P0 (deal-breaker)** | Scraping and cleaning, chunking, duplicate-safe ingestion | Built |
| **P0** | Vector database plus metadata database | Built (Chroma, SQLite) |
| **P0** | Retrieval with keyword and dense methods, and the 4-row experiment table | Built |
| **P0** | RAG answer with citations and refusal when unsupported | Built |
| **P0** | Streamlit interface, README and unit tests | Built |
| **P1** | In-scope / out-of-scope classifier (train, validate, test) | Built |
| **P1** | Labelled evaluation set (30+ questions), answer-quality metrics, LLM judge versus human labels | Built |
| **P1** | Exact and semantic caching with wrong-hit protection | Built |
| **P1** | Error analysis of the five worst answers | Built |
| **P2 (nice to have)** | Hybrid retrieval (BM25 + dense) | Built as two extra experiment rows because it was cheap |
| **P2** | Cross-encoder reranker, FastAPI service, Docker, Elasticsearch, multi-source ingestion | Not built (future work) |

**Deal-breakers for the MVP:** re-runnable ingestion without duplicates, retrieval that works, answers with citations, refusal of unsupported questions, and honest measurement. Everything marked P1 or P2 is built only after these work.

## 4. Technical planning (summary)

Full details, with the reasons for each choice and the alternatives, are in `technical_documentation.md`.

- **Architecture:** an offline pipeline (scrape, chunk, deduplicate, embed, store) and an online pipeline (cache, retrieve, three safety layers, generate). See `diagrams/architecture.png`.
- **Models and libraries:** `bge-m3` for embeddings and `llama3.2` for answers, both through Ollama; `rank_bm25` for keyword search; scikit-learn Logistic Regression for the classifier; Streamlit for the interface.
- **Databases:** ChromaDB for vectors, SQLite for chunk metadata and the cache.
- **Data pipeline:** `scrape.py` then `ingest.py`; chunk ID is a SHA-256 hash of URL and text so identical chunks are never stored twice.
- **Model training and evaluation:** retrieval experiments over 2 chunk sizes and 3 retrievers; classifier trained on 70% of labelled questions, tuned on 15% and tested once on 15%, then checked with cross-validation and a hand-checked hard set; answer quality scored by metrics defined in advance and compared with human labels.
- **Testing:** pytest unit tests with fake embedders, fake language models and in-memory databases, so tests run without Ollama.
- **Integration approach:** all settings in `config.py`; each stage reads the files written by the previous stage; the app wires the retriever, classifier, cache and language model together.

**Alternatives considered:** FAISS instead of Chroma; sentence-transformers instead of Ollama embeddings; fine-tuning a cross-encoder instead of a logistic-regression classifier; FastAPI instead of Streamlit; PostgreSQL instead of SQLite. Reasons are in the technical documentation.

## 5. Time management

Indicative allocation for the 20 hours. **The "Actual" column must be filled in with the real time spent.**

| Activity | Planned | Actual |
|---|---|---|
| Requirement analysis and planning | 2 h | ___ |
| Data collection and preprocessing | 2 h | ___ |
| Architecture and database design | 2 h | ___ |
| Core implementation (retrieval, RAG, UI, cache) | 7 h | ___ |
| Model training and experiments (classifier, retrieval, evaluation) | 3 h | ___ |
| Testing and debugging | 2 h | ___ |
| Documentation and demo preparation | 2 h | ___ |
| **Total** | **20 h** | ___ |

## 6. Risks and mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Hardware limits (CPU only) | Slow generation, long evaluation runs | Small models (`llama3.2` 3B), batch embeddings, caching, a quick-test option (`evaluate_answers 5`). |
| Model inference latency | Slow demo | Cache for repeated questions; rehearse the demo; prepared backup screenshots. |
| Dataset availability and quality | Noisy text (for example encoding errors in some characters) | Save snapshots, clean HTML, document the known issue; evidence-based validation of generated questions. |
| Model compatibility | Ollama or model versions behave differently | Pin model names in `config.py`; record versions in the README. |
| Retrieval quality | Wrong chunks lead to wrong answers | Compare chunk sizes and retrievers; threshold and classifier refuse weak matches; diagnose misses in error analysis. |
| Integration dependencies | Ollama not running breaks every step | Clear error messages, README check (`ollama list`), tests that need no Ollama. |
| Evaluation limitations | Small sets, LLM-written questions, weak 3B judge | Human review of all datasets, human labels to check the judge, report limitations and agreement rather than only scores. |
| Time overrun | Incomplete MVP | Strict P0/P1/P2 priorities; small working system before extra features. |

## 7. Demo strategy

The demo shows: architecture, data ingestion (including a second run with no duplicates), the retrieval workflow, a successful question with citations, an unsuccessful or edge-case question, the evaluation results, and the known limitations with proposed improvements. The detailed script, timings and likely questions are in `demo_and_presentation.md`.
