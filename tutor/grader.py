"""
NextGen Tutor AI – Grader (Groq)
==================================
Uses Groq's LLM to perform rubric-based grading of quiz answers with
partial credit, feedback, and hints.
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from utils.config import DEFAULT_MODEL, GRADING_TEMPLATE


def grade_answers(
    questions: List[Dict[str, Any]],
    user_answers: Dict[int, str],
    groq_api_key: str,
    temperature: float = 0.2,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """Grade learner answers against reference answers using Groq LLM.

    Parameters
    ----------
    questions : list[dict]
        The quiz questions (with correct_answer field).
    user_answers : dict[int, str]
        Mapping of question id → learner's answer string.
    groq_api_key : str
        User-supplied Groq API key.
    temperature : float
        LLM temperature (low for consistency).
    model : str
        Groq model to use.

    Returns
    -------
    list[dict]
        Each dict has: id, is_correct, score, feedback, hint,
        retry_recommended.
    """
    # Build the JSON strings for the prompt
    q_for_prompt = []
    for q in questions:
        q_for_prompt.append({
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "correct_answer": q["correct_answer"],
        })

    a_for_prompt = []
    for q in questions:
        qid = q["id"]
        a_for_prompt.append({
            "id": qid,
            "user_answer": user_answers.get(qid, "(no answer)"),
        })

    prompt = GRADING_TEMPLATE.format(
        questions_json=json.dumps(q_for_prompt, indent=2),
        answers_json=json.dumps(a_for_prompt, indent=2),
    )

    llm = ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=groq_api_key,
    )

    response = llm.invoke([
        SystemMessage(
            content="You are a fair grader. Return ONLY valid JSON."
        ),
        HumanMessage(content=prompt),
    ])

    raw = response.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        grades = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            grades = json.loads(match.group(0))
        else:
            grades = []

    return grades
