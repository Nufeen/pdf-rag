"""RAG evaluation scoring functions."""

import re

import numpy as np
import requests
from ollama import Client

from pdf_rag.llm import load_prompt


def factual_score(
    question: str,
    answer: str,
    ground_truth: str,
    client: Client,
    judge_model: str,
) -> float:
    """
    Score factual correctness of answer vs ground truth using LLM judge.

    Returns a float between 0.0 and 1.0.
    Falls back to 0.0 on parse failure.
    """
    prompt = load_prompt(
        "evaluate_answer", question=question, ground_truth=ground_truth, answer=answer
    )

    try:
        response = client.chat(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        content = response["message"]["content"].strip()
    except Exception:
        return 0.0

    # Parse first float 0-1 from response
    match = re.search(r"0?\.[0-9]+|1\.0|[01]", content)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 0.0
    return 0.0


def semantic_score(
    answer: str,
    ground_truth: str,
    embed_model: str,
    base_url: str,
) -> float:
    """
    Compute cosine similarity between answer and ground_truth embeddings.

    Returns a float between 0.0 and 1.0.
    """
    try:
        resp = requests.post(
            f"{base_url}/api/embed",
            json={"model": embed_model, "input": [answer, ground_truth]},
            timeout=60,
        )
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
        answer_emb = np.array(embeddings[0])
        gt_emb = np.array(embeddings[1])

        # Cosine similarity
        norm_a = np.linalg.norm(answer_emb)
        norm_b = np.linalg.norm(gt_emb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(answer_emb, gt_emb) / (norm_a * norm_b))
    except Exception:
        return 0.0


def score(
    question: str,
    answer: str,
    ground_truth: str,
    client: Client,
    judge_model: str,
    embed_model: str,
    base_url: str,
    factual_weight: float = 0.75,
) -> float:
    """
    Compute answer_correctness as weighted average of factual and semantic scores.

    Default weight: 0.75 factual, 0.25 semantic (per RAGAS).
    """
    f_score = factual_score(question, answer, ground_truth, client, judge_model)
    s_score = semantic_score(answer, ground_truth, embed_model, base_url)
    return factual_weight * f_score + (1 - factual_weight) * s_score
