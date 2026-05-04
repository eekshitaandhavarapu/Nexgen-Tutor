"""
Code Arena – Contest Manager
==============================
Manages timed coding competitions with leaderboard tracking.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from arena.models import gen_id, save_contest, get_contests, update_rating


def create_contest(
    problems: List[Dict],
    duration_minutes: int = 30,
    user: str = "Learner",
) -> Dict[str, Any]:
    """Create a new timed contest.

    Returns a contest dict with id, problems, start_time, duration, status.
    """
    contest = {
        "id": gen_id("contest"),
        "user": user,
        "problems": [p["id"] for p in problems],
        "problem_details": problems,
        "duration_minutes": duration_minutes,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "status": "active",
        "submissions": {},  # problem_id -> {code, results, timestamp}
        "score": 0,
        "created_at": datetime.now().isoformat(),
    }
    save_contest(contest)
    return contest


def get_remaining_time(contest: Dict) -> int:
    """Get remaining time in seconds for a contest.

    Returns 0 if contest is expired.
    """
    start = datetime.fromisoformat(contest["start_time"])
    elapsed = (datetime.now() - start).total_seconds()
    remaining = (contest["duration_minutes"] * 60) - elapsed
    return max(0, int(remaining))


def is_contest_active(contest: Dict) -> bool:
    """Check if the contest is still within its time limit."""
    return get_remaining_time(contest) > 0 and contest.get("status") == "active"


def submit_solution(
    contest: Dict,
    problem_id: str,
    code: str,
    test_results: Dict,
) -> Dict:
    """Record a solution submission for a contest problem.

    Returns updated contest dict.
    """
    contest["submissions"][problem_id] = {
        "code": code,
        "results": test_results,
        "timestamp": datetime.now().isoformat(),
        "passed": test_results.get("all_passed", False),
    }

    # Recalculate score
    score = 0
    for pid, sub in contest["submissions"].items():
        if sub.get("passed"):
            # Find problem difficulty for scoring
            prob = next(
                (p for p in contest.get("problem_details", []) if p["id"] == pid),
                None,
            )
            if prob:
                diff_scores = {"Easy": 100, "Medium": 200, "Hard": 300}
                score += diff_scores.get(prob.get("difficulty", "Easy"), 100)

    contest["score"] = score
    return contest


def end_contest(contest: Dict, user: str) -> Dict:
    """End a contest and update ratings."""
    contest["status"] = "completed"
    contest["end_time"] = datetime.now().isoformat()

    # Calculate rating change based on performance
    solved = sum(1 for s in contest["submissions"].values() if s.get("passed"))
    total = len(contest.get("problems", []))

    if total > 0:
        ratio = solved / total
        if ratio >= 0.8:
            delta = 50
            badge = "🏆 Contest Champion" if ratio == 1.0 else None
        elif ratio >= 0.5:
            delta = 25
            badge = None
        elif ratio > 0:
            delta = 10
            badge = None
        else:
            delta = -10
            badge = None

        update_rating(user, delta, badge)

    return contest


def format_time(seconds: int) -> str:
    """Format seconds as MM:SS."""
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"
