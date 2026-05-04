"""
Code Arena – Data Models & Persistence
========================================
JSON-based persistence for problems, submissions, contests, and ratings.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "arena_data")


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def gen_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ── Problem persistence ─────────────────────────────────────────────────────

def _problems_path() -> str:
    return os.path.join(DATA_DIR, "problems.json")


def save_problem(problem: Dict) -> None:
    _ensure_dir()
    problems = get_problems()
    problems.append(problem)
    with open(_problems_path(), "w") as f:
        json.dump(problems, f, indent=2)


def get_problems() -> List[Dict]:
    path = _problems_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def get_problem_by_id(problem_id: str) -> Optional[Dict]:
    for p in get_problems():
        if p.get("id") == problem_id:
            return p
    return None


# ── Submission persistence ───────────────────────────────────────────────────

def _submissions_path() -> str:
    return os.path.join(DATA_DIR, "submissions.json")


def save_submission(submission: Dict) -> None:
    _ensure_dir()
    subs = get_submissions()
    subs.append(submission)
    with open(_submissions_path(), "w") as f:
        json.dump(subs, f, indent=2)


def get_submissions(user: Optional[str] = None) -> List[Dict]:
    path = _submissions_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        subs = json.load(f)
    if user:
        return [s for s in subs if s.get("user") == user]
    return subs


# ── Contest persistence ──────────────────────────────────────────────────────

def _contests_path() -> str:
    return os.path.join(DATA_DIR, "contests.json")


def save_contest(contest: Dict) -> None:
    _ensure_dir()
    contests = get_contests()
    contests.append(contest)
    with open(_contests_path(), "w") as f:
        json.dump(contests, f, indent=2)


def get_contests() -> List[Dict]:
    path = _contests_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


# ── Rating & Badges ─────────────────────────────────────────────────────────

def _ratings_path() -> str:
    return os.path.join(DATA_DIR, "ratings.json")


def get_rating(user: str) -> Dict:
    _ensure_dir()
    path = _ratings_path()
    if not os.path.exists(path):
        return {"user": user, "rating": 1200, "problems_solved": 0, "badges": []}
    with open(path) as f:
        ratings = json.load(f)
    for r in ratings:
        if r.get("user") == user:
            return r
    return {"user": user, "rating": 1200, "problems_solved": 0, "badges": []}


def update_rating(user: str, delta: int, badge: Optional[str] = None) -> Dict:
    _ensure_dir()
    path = _ratings_path()
    ratings = []
    if os.path.exists(path):
        with open(path) as f:
            ratings = json.load(f)

    found = False
    for r in ratings:
        if r.get("user") == user:
            r["rating"] = max(0, r["rating"] + delta)
            r["problems_solved"] = r.get("problems_solved", 0) + (1 if delta > 0 else 0)
            if badge and badge not in r.get("badges", []):
                r.setdefault("badges", []).append(badge)
            found = True
            user_rating = r
            break

    if not found:
        user_rating = {
            "user": user,
            "rating": max(0, 1200 + delta),
            "problems_solved": 1 if delta > 0 else 0,
            "badges": [badge] if badge else [],
        }
        ratings.append(user_rating)

    with open(path, "w") as f:
        json.dump(ratings, f, indent=2)

    return user_rating


def get_leaderboard() -> List[Dict]:
    _ensure_dir()
    path = _ratings_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        ratings = json.load(f)
    return sorted(ratings, key=lambda r: r.get("rating", 0), reverse=True)
