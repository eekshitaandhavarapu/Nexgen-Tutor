"""
Code Arena – AI Coach (Groq)
==============================
Provides hints, code review, optimization suggestions, and explanations.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from utils.config import DEFAULT_MODEL


COACH_SYSTEM = """You are an expert coding coach. Be concise, encouraging, and educational.
When giving hints, don't reveal the full solution — guide the learner step by step.
When reviewing code, point out bugs, inefficiencies, and suggest improvements."""


def get_hint(
    problem_description: str,
    current_code: str,
    groq_api_key: str,
    hint_level: int = 1,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate a hint for the learner.

    hint_level:
      1 = gentle nudge (conceptual direction)
      2 = moderate help (approach suggestion)
      3 = strong hint (near-solution guidance)
    """
    levels = {
        1: "Give a GENTLE conceptual nudge. Don't mention specific code. Just point them in the right direction with a question or analogy.",
        2: "Give a MODERATE hint. Suggest the algorithmic approach and key data structures, but don't write any code.",
        3: "Give a STRONG hint. Explain the step-by-step approach and mention key operations, but let them write the code.",
    }

    prompt = (
        f"Problem:\n{problem_description[:2000]}\n\n"
        f"Learner's current code:\n```\n{current_code[:1500]}\n```\n\n"
        f"Hint level: {levels.get(hint_level, levels[1])}\n\n"
        f"Provide a hint."
    )

    llm = ChatGroq(model=model, temperature=0.4, groq_api_key=groq_api_key)
    resp = llm.invoke([
        SystemMessage(content=COACH_SYSTEM),
        HumanMessage(content=prompt),
    ])
    return resp.content.strip()


def review_code(
    problem_description: str,
    code: str,
    test_results: dict,
    groq_api_key: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Review submitted code and provide feedback."""
    passed = test_results.get("total_passed", 0)
    total = test_results.get("total_tests", 0)

    prompt = (
        f"Problem:\n{problem_description[:2000]}\n\n"
        f"Submitted code:\n```\n{code[:2000]}\n```\n\n"
        f"Test results: {passed}/{total} passed\n\n"
        f"Provide a brief code review covering:\n"
        f"1. Correctness issues (if any tests failed)\n"
        f"2. Code quality and style\n"
        f"3. Time/space complexity\n"
        f"4. One optimization suggestion"
    )

    llm = ChatGroq(model=model, temperature=0.3, groq_api_key=groq_api_key)
    resp = llm.invoke([
        SystemMessage(content=COACH_SYSTEM),
        HumanMessage(content=prompt),
    ])
    return resp.content.strip()


def get_solution(
    problem_description: str,
    language: str,
    groq_api_key: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate a complete reference solution with step-by-step explanation.

    Parameters
    ----------
    problem_description : str
        The coding problem description.
    language : str
        Programming language for the solution (e.g. "Python").
    groq_api_key : str
        User-supplied Groq API key.
    model : str
        Groq model to use.

    Returns
    -------
    str
        Formatted solution with explanation in markdown.
    """
    prompt = (
        f"Problem:\n{problem_description[:2000]}\n\n"
        f"Provide a complete, well-commented solution in {language}.\n\n"
        f"Format your response as:\n"
        f"## Approach\n(Brief explanation of the algorithm/strategy)\n\n"
        f"## Solution\n```{language.lower()}\n(complete code)\n```\n\n"
        f"## Complexity\n- Time: O(...)\n- Space: O(...)\n\n"
        f"## Key Insights\n(2-3 bullet points about why this approach works)"
    )

    llm = ChatGroq(model=model, temperature=0.2, groq_api_key=groq_api_key)
    try:
        resp = llm.invoke([
            SystemMessage(content="You are an expert programmer. Provide clean, correct solutions with clear explanations."),
            HumanMessage(content=prompt),
        ])
        return resp.content.strip()
    except Exception:
        return "Could not generate solution. Please try again."
