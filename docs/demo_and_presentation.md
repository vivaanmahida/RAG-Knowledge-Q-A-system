# Demo and Presentation Guide

Total: 20 to 25 minutes of presentation, then panel questions.

## 1. Presentation outline (matches the 8 required topics)

| # | Topic | Minutes | What to say / show |
|---|---|---|---|
| 1 | Problem understanding and assumptions | 2 | RAG answers only from my own knowledge base; why refusal and measurement matter; domain = MDN web basics; assumptions list from the plan. |
| 2 | Requirements and prioritisation | 2 | P0 / P1 / P2 table; deal-breakers; what I deliberately left out. |
| 3 | Architecture and technology selection | 3 | `architecture.png`; offline vs online; three refusal layers; one sentence per technology on *why* and *alternative*. |
| 4 | Implementation walk-through | 4 | Ingestion (hash ID), retrieval (BM25 / dense / hybrid), `rag.py` prompt and refusal, cache guards. Show a few real lines of code. |
| 5 | Model training and evaluation | 3 | Classifier: data, 70/15/15 split, choosing C on validation, baseline. Answer-quality metrics and how the judge was checked against my labels. |
| 6 | Experiment results and metrics | 3 | Retrieval table, classifier results (including the harder hand-checked set), refusal funnel, answer-quality table, cache table. |
| 7 | Error analysis and limitations | 3 | Two or three of the worst-five answers; retrieval miss versus generation problem; the limitations list. |
| 8 | Improvements with another week | 2 | Reranker, retrieval-aware classifier, larger labelled set, stronger judge, encoding fix, API and Docker. |

Keep every number consistent with the files in `results/`. Say clearly which results are weak and why.

## 2. Live demo script (about 6 to 8 minutes inside topic 4 or after topic 6)

**Before you start:** Ollama running, `ollama list` shows both models, app tested once, browser tab open, terminal open in the project folder, clear the cache with the "Clear cache" button.

1. **Show the structure:** `dir` / the file tree; open `config.py` for one second.
2. **Ingestion without duplicates:** run `python -m src.ingest` and point at "0 new, N skipped".
3. **Start the app:** `streamlit run app.py`.
4. **Successful use case:** ask *"What is the main purpose of the alt attribute?"* Show the answer, the `[n]` citation, the *Sources cited* panel and the similarity score.
5. **Obvious refusal:** ask *"What is the capital of Gujarat?"* Show "I don't have enough information…" and that the best similarity was low, so the language model was never called.
6. **Edge case:** a near-topic question such as *"What does the CSS opacity property do?"* Explain which layer stops it, or what the system does if it slips through. (Use the result of `refusal_checks` to choose the question and be honest about the outcome. TODO: pick and rehearse.)
7. **Cache:** ask the first question again (exact hit), then a paraphrase (semantic hit), then *"What does margin do?"* followed by *"What does padding do?"* to show the guard preventing a wrong reuse.
8. **Failure case:** open `results/worst5.md`, read one entry, explain the cause and the fix.
9. **Tests:** run `python -m pytest tests` and show all passing.

**Backup plan if Ollama or the laptop fails:** keep screenshots of steps 4 to 7, the `results/` files open in an editor, and the tests (they need no Ollama).

## 3. Questions the panel may ask, with short answers

1. **Why RAG instead of just asking the model?** The model alone can invent facts; retrieval grounds the answer in my documents and allows citations and refusal.
2. **How do you avoid duplicates on re-run?** Chunk ID is the SHA-256 of URL plus text; SQLite ignores existing IDs and Chroma upserts.
3. **Why Chroma and SQLite both?** Chroma is built for vector search; SQLite holds the text, metadata and cache and feeds BM25.
4. **Why 200-word chunks?** Recall was within about one question of the best setting, five 200-word chunks fit the model's context, and the dense score is usable for refusal. State honestly that 500-word chunks are not a perfectly fair comparison.
5. **Explain cosine similarity.** The angle between two vectors: close to 1 means similar meaning.
6. **Explain Recall@5, MRR, nDCG.** Found in top five; average of 1 / rank of the first hit; hit position weighted by `1 / log2(rank + 1)`.
7. **How does hybrid (RRF) work?** Each retriever's ranks give `1 / (60 + rank)` points; chunks both like get the most.
8. **Why Logistic Regression for the classifier?** Simple, fast, works well on embeddings, interpretable probabilities, little overfitting with the `C` setting.
9. **Is perfect test F1 believable?** No: the test set is small and LLM-written. That is why I added cross-validation and a hand-checked hard set, whose score was lower.
10. **Why three refusal layers?** The similarity score cannot separate near-topic questions; the classifier cannot know which topics the pages cover; the language model is the last check. The funnel check measures what leaks through.
11. **How do you stop wrong cache answers?** Only answers are cached; a semantic hit needs high similarity plus matching numbers, negation and topic words; the cache is invalidated if the knowledge base changes. The first guard design was too strict (it blocked almost every valid paraphrase), so I compared four guard modes on labelled pairs.
12. **How reliable is your LLM judge?** I compared it with my own labels on a sample: report agreement and kappa, and say it is a 3B model judging its own answers.
13. **What would you change first?** Add a reranker and more out-of-scope training data about topics adjacent to the knowledge base.
14. **Can you change X live?** Be ready to change a threshold in `config.py`, a prompt line in `rag.py`, or the chunk size, and re-run a test. Practise this once.

## 4. Preparation checklist

- [ ] Read every file in `src/` once and be able to say what it does in one sentence.
- [ ] Be able to draw the architecture from memory.
- [ ] Know every number in the report and where it came from.
- [ ] Have your plan's "Actual hours" column filled in.
- [ ] Be ready to say exactly how AI tools were used (see the README disclosure).
- [ ] Rehearse the demo twice with the timing above.
