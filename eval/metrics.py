"""
NextGen Tutor AI – Evaluation Metrics
=======================================
Lightweight ROUGE-L and BLEU scoring for proxy evaluation of RAG
answer quality, plus a hallucination-risk heuristic.
"""

import math
from typing import List


# ── ROUGE-L ──────────────────────────────────────────────────────────────────

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of the Longest Common Subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


_MAX_TOKENS = 500  # Safety cap to prevent O(n×m) blowup in LCS


def compute_rouge_l(answer: str, reference: str) -> dict:
    """Compute ROUGE-L precision, recall, and F1.

    Both inputs are lowercased and tokenised by whitespace.
    Token lists are capped at _MAX_TOKENS to bound LCS runtime.
    Returns a dict with keys: precision, recall, f1.
    """
    ans_tokens = answer.lower().split()[:_MAX_TOKENS]
    ref_tokens = reference.lower().split()[:_MAX_TOKENS]

    if not ans_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(ans_tokens, ref_tokens)
    precision = lcs / len(ans_tokens) if ans_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ── BLEU (simplified unigram) ───────────────────────────────────────────────

def compute_bleu(answer: str, reference: str) -> float:
    """Compute a simplified unigram BLEU score.

    Uses unigram precision with a brevity penalty.
    Token lists are capped at _MAX_TOKENS for consistency.
    """
    ans_tokens = answer.lower().split()[:_MAX_TOKENS]
    ref_tokens = reference.lower().split()[:_MAX_TOKENS]

    if not ans_tokens or not ref_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    clipped = 0
    for t in ans_tokens:
        if ref_counts.get(t, 0) > 0:
            clipped += 1
            ref_counts[t] -= 1

    precision = clipped / len(ans_tokens)

    # Brevity penalty
    bp = 1.0
    if len(ans_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(ans_tokens))

    return round(bp * precision, 4)


# ── Hallucination Risk ───────────────────────────────────────────────────────

def flag_hallucination_risk(answer: str, retrieved_context: str, threshold: float = 0.15) -> dict:
    """Heuristic hallucination check.

    Computes token overlap between the answer and retrieved context.
    Low overlap → higher hallucination risk.

    Returns:
        risk_flag (bool): True if overlap is below threshold.
        overlap_ratio (float): fraction of answer tokens found in context.
    """
    ans_tokens = set(answer.lower().split())
    ctx_tokens = set(retrieved_context.lower().split())

    if not ans_tokens:
        return {"risk_flag": True, "overlap_ratio": 0.0}

    overlap = ans_tokens & ctx_tokens
    ratio = len(overlap) / len(ans_tokens)

    return {
        "risk_flag": ratio < threshold,
        "overlap_ratio": round(ratio, 4),
    }
