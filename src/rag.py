

import re

import config
from src.llm import generate

NO_ANSWER = "I don't have enough information in the knowledge base to answer that."

PROMPT = """You are a helpful assistant for web development questions.
Answer the question using ONLY the sources below.
- After each fact, write the source number in square brackets, like [1] or [2].
- Do not use any outside knowledge.
- If the sources do not contain the answer, reply exactly: {no_answer}

Sources:
{sources}

Question: {question}

Answer:"""


def build_context(chunks):
    lines = []
    for i, chunk in enumerate(chunks):
        lines.append(f"[{i + 1}] ({chunk['title']}) {chunk['text']}")
    return "\n\n".join(lines)


def find_cited_numbers(answer, number_of_sources):
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in numbers if 1 <= n <= number_of_sources)


def llm_said_no(answer):
    text = answer.lower().replace("’", "'")
    return "don't have enough information" in text


def make_result(answer, refused, reason, best_score, sources=None, retrieved=None,
                warning=None, scope_probability=None):
    return {
        "answer": answer,
        "refused": refused,
        "reason": reason,
        "best_score": best_score,
        "scope_probability": scope_probability,
        "sources": sources or [],
        "retrieved": retrieved or [],
        "warning": warning,
    }


def answer_question(question, retriever, llm=generate,
                    threshold=config.REFUSE_THRESHOLD, k=config.TOP_K, classifier=None):
    question = question.strip()
    if not question:
        return make_result("Please type a question.", True, "empty_question", 0.0)


    chunks = retriever.search(question, k)
    best_score = chunks[0]["score"] if chunks else 0.0


    # weak match: refuse without calling the llm
    if best_score < threshold:
        return make_result(NO_ANSWER, True, "low_similarity", best_score, retrieved=chunks)


    scope_probability = None
    # second check: scope classifier
    if classifier is not None:
        in_scope, scope_probability = classifier.is_in_scope(question)
        if not in_scope:
            return make_result(NO_ANSWER, True, "out_of_scope_classifier", best_score,
                               retrieved=chunks, scope_probability=scope_probability)


    prompt = PROMPT.format(no_answer=NO_ANSWER, sources=build_context(chunks), question=question)
    answer = llm(prompt).strip()


    # the model can refuse on its own
    if llm_said_no(answer):
        return make_result(NO_ANSWER, True, "llm_refused", best_score, retrieved=chunks,
                           scope_probability=scope_probability)


    cited = find_cited_numbers(answer, len(chunks))
    cited_chunks = [chunks[n - 1] for n in cited]
    warning = None
    if not cited_chunks:
        warning = "The answer has no source citations, so treat it with care."

    return make_result(answer, False, "ok", best_score, cited_chunks, chunks, warning,
                       scope_probability)
