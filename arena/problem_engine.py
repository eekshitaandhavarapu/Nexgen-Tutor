"""
Code Arena – Problem Generation Engine (Groq)
================================================
Generates coding problems from course PDFs using RAG + Groq.
"""

import json
import re
import time
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from utils.config import DEFAULT_MODEL
from rag.retriever import retrieve_with_scores, format_context
from arena.models import gen_id, save_problem


# ── Supported Languages ─────────────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    "Python": {
        "extension": ".py",
        "comment": "#",
        "starter_hint": "def solution(args):\n    # Your code here\n    pass",
    },
    "JavaScript": {
        "extension": ".js",
        "comment": "//",
        "starter_hint": "function solution(args) {\n    // Your code here\n}",
    },
}


PROBLEM_GENERATION_PROMPT = """You are a competitive programming problem author. Using ONLY the course material below, create a {difficulty} coding problem on the topic "{topic}".

TARGET LANGUAGE: {language}

COURSE MATERIAL:
{rag_context}

DIFFICULTY GUIDE:
- Easy: Single concept, straightforward, 1-2 visible examples
- Medium: Combines 2 concepts, edge cases matter, 2-3 visible examples
- Hard: Multi-step algorithm, optimal solution non-obvious, tricky edge cases

REQUIREMENTS:
1. The problem MUST be solvable in {language}
2. Input is read from function arguments (not stdin)
3. Output is the return value of the function
4. Include 2-3 visible examples with input, output, and explanation
5. Include 4-6 hidden test cases covering edge cases
6. Starter code with a single function named "solution"
7. Reference solution that is correct and efficient

Return ONLY valid JSON (no markdown fences) with this structure:
{{
    "title": "Problem Title",
    "description": "Full problem description with constraints",
    "examples": [
        {{"input": "nums = [1,3,5], target = 3", "output": "1", "explanation": "3 is at index 1"}}
    ],
    "hidden_tests": [
        {{"input": "nums = [1], target = 1", "expected": "0"}}
    ],
    "starter_code": "<starter code in {language}>",
    "reference_solution": "<complete solution in {language}>",
    "constraints": "1 <= len(nums) <= 10^4",
    "topic_tags": ["Arrays", "Search"]
}}"""


def generate_problem(
    topic: str,
    difficulty: str,
    vectorstore,
    api_key: str,
    language: str = "Python",
    model: str = DEFAULT_MODEL,
) -> Optional[dict]:
    """Generate a coding problem using RAG context from course PDFs."""
    docs = retrieve_with_scores(
        query=f"{topic} algorithms data structures programming concepts",
        vectorstore=vectorstore,
        top_k=5,
    )

    if not docs:
        return None

    rag_context = format_context(docs)

    prompt = PROBLEM_GENERATION_PROMPT.format(
        difficulty=difficulty,
        topic=topic,
        language=language,
        rag_context=rag_context,
    )

    llm = ChatGroq(
        model=model,
        temperature=0.7,
        groq_api_key=api_key,
    )

    response = None
    for attempt in range(3):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(5 * (attempt + 1))
            else:
                raise

    if response is None:
        return None

    raw = response.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        problem_data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                problem_data = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None

    problem = {
        "id": gen_id("prob"),
        "title": problem_data.get("title", f"{topic} Problem"),
        "difficulty": difficulty,
        "language": language,
        "topic_tags": problem_data.get("topic_tags", [topic]),
        "description": problem_data.get("description", ""),
        "examples": problem_data.get("examples", []),
        "hidden_tests": problem_data.get("hidden_tests", []),
        "starter_code": problem_data.get("starter_code", ""),
        "reference_solution": problem_data.get("reference_solution", ""),
        "constraints": problem_data.get("constraints", ""),
        "source_chunks": [
            f"{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}"
            for d, _ in docs[:3]
        ],
        "created_at": datetime.now().isoformat(),
        "times_solved": 0,
        "times_attempted": 0,
    }

    save_problem(problem)
    return problem


def get_available_topics() -> List[str]:
    """Return default CS topics for problem generation."""
    return [
        "Arrays", "Strings", "Linked Lists", "Stacks & Queues",
        "Trees", "Graphs", "Sorting", "Searching",
        "Dynamic Programming", "Recursion", "Hash Tables",
        "Greedy Algorithms", "Bit Manipulation", "Math",
    ]
