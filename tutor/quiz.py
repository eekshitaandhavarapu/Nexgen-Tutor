"""
NextGen Tutor AI – Quiz Generator (Groq)
==========================================
Generates micro-quizzes from retrieved context chunks using Groq's LLM.

Educators can customise:
  - num_questions           (default 5)
  - question type mix       (edit QUIZ_GENERATION_TEMPLATE in config.py)
  - difficulty mapping      (edit difficulty param)
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from utils.config import DEFAULT_MODEL, QUIZ_GENERATION_TEMPLATE, QUIZ_HINT_TEMPLATE


def generate_quiz(
    context: str,
    groq_api_key: str,
    difficulty: str = "medium",
    num_questions: int = 5,
    temperature: float = 0.4,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """Generate a quiz from the given context using Groq LLM.

    Parameters
    ----------
    context : str
        The retrieved context string (already formatted).
    groq_api_key : str
        User-supplied Groq API key.
    difficulty : str
        One of "easy", "medium", "hard".
    num_questions : int
        Number of questions to generate (3–5 recommended).
    temperature : float
        LLM temperature for creativity control.
    model : str
        Groq model to use (e.g. "llama3-70b-8192").

    Returns
    -------
    list[dict]
        Each dict has: id, type, question, options, correct_answer,
        difficulty, explanation.
    """
    prompt = QUIZ_GENERATION_TEMPLATE.format(
        context=context,
        num_questions=num_questions,
        difficulty=difficulty,
    )

    llm = ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=groq_api_key,
    )

    response = llm.invoke([
        SystemMessage(
            content="You are a quiz generator. Return ONLY valid JSON."
        ),
        HumanMessage(content=prompt),
    ])

    raw = response.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        questions = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: attempt to extract the first JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            questions = json.loads(match.group(0))
        else:
            questions = []

    return questions


def generate_hint(
    question_text: str,
    context: str,
    groq_api_key: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate a conceptual hint for a quiz question without revealing the answer.

    Parameters
    ----------
    question_text : str
        The quiz question the learner is stuck on.
    context : str
        The original context the quiz was generated from.
    groq_api_key : str
        User-supplied Groq API key.
    model : str
        Groq model to use.

    Returns
    -------
    str
        A short hint guiding the learner toward the answer.
    """
    prompt = QUIZ_HINT_TEMPLATE.format(
        question=question_text,
        context=context[:3000],
    )

    llm = ChatGroq(
        model=model,
        temperature=0.4,
        groq_api_key=groq_api_key,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        return "💡 Think about the key concepts in this topic and how they relate to each other."
