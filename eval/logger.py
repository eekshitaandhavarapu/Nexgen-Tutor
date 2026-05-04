"""
NextGen Tutor AI – Interaction & Quiz Logger
==============================================
Persists interaction data and quiz results to JSONL files for the
Insights dashboard and CSV/JSON export.
"""

import csv
import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.config import LOG_DIR


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _interaction_path() -> str:
    return os.path.join(LOG_DIR, "interactions.jsonl")


def _quiz_path() -> str:
    return os.path.join(LOG_DIR, "quiz_results.jsonl")


# ── Interaction logging ──────────────────────────────────────────────────────

def log_interaction(
    query: str,
    answer: str,
    retrieved_sources: List[str],
    avg_similarity: float,
    rouge_l: Optional[Dict] = None,
    bleu: Optional[float] = None,
    hallucination_flag: Optional[bool] = None,
    learner_name: str = "Learner",
    course: str = "General",
) -> None:
    """Append a single Q&A interaction record to the log file."""
    _ensure_log_dir()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "learner_name": learner_name,
        "course": course,
        "query": query,
        "answer_length": len(answer),
        "retrieved_sources": retrieved_sources,
        "avg_similarity": avg_similarity,
        "rouge_l": rouge_l,
        "bleu": bleu,
        "hallucination_flag": hallucination_flag,
    }
    with open(_interaction_path(), "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Quiz result logging ─────────────────────────────────────────────────────

def log_quiz_result(
    learner_name: str,
    topic: str,
    scores: List[Dict[str, Any]],
    overall_score: float,
) -> None:
    """Append a quiz attempt record."""
    _ensure_log_dir()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "learner_name": learner_name,
        "topic": topic,
        "overall_score": overall_score,
        "details": scores,
    }
    with open(_quiz_path(), "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Reading logs ─────────────────────────────────────────────────────────────

def read_interactions() -> List[Dict]:
    """Read all logged interactions."""
    path = _interaction_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_quiz_results() -> List[Dict]:
    """Read all logged quiz results."""
    path = _quiz_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Export ───────────────────────────────────────────────────────────────────

def export_logs(fmt: str = "json") -> str:
    """Export combined interaction + quiz logs.

    Parameters
    ----------
    fmt : str
        ``"json"`` or ``"csv"``.

    Returns
    -------
    str
        Serialised log data ready for download.
    """
    interactions = read_interactions()
    quizzes = read_quiz_results()

    combined = {
        "interactions": interactions,
        "quiz_results": quizzes,
    }

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        # Interactions
        writer.writerow(["--- Interactions ---"])
        if interactions:
            writer.writerow(interactions[0].keys())
            for row in interactions:
                writer.writerow([str(v) for v in row.values()])
        writer.writerow([])
        # Quizzes
        writer.writerow(["--- Quiz Results ---"])
        if quizzes:
            writer.writerow(quizzes[0].keys())
            for row in quizzes:
                writer.writerow([str(v) for v in row.values()])
        return output.getvalue()

    return json.dumps(combined, indent=2)
