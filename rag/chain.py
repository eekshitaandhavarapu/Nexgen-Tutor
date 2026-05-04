"""
NextGen Tutor AI – QA Chain (Groq)
====================================
Builds a LangChain chain that combines retrieval context + learner profile
into a structured, cited answer using Groq's LLM API.

Includes low-confidence ("I don't know") handling.
"""

from typing import Any, Dict, List, Tuple
import time

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from rag.retriever import format_context, retrieve_with_scores
from utils.config import (
    DEFAULT_MODEL,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    FOLLOWUP_TEMPLATE,
    RAG_ANSWER_TEMPLATE,
    TUTOR_SYSTEM_PROMPT,
)

# ── Cached LLM factory ──────────────────────────────────────────────────────
_llm_cache: dict[tuple, ChatGroq] = {}


def _get_llm(api_key: str, model: str, temperature: float) -> ChatGroq:
    """Return a cached ChatGroq instance, creating one if needed.

    Avoids ~200ms overhead of re-creating HTTP clients and auth per call.
    """
    key = (api_key, model, temperature)
    if key not in _llm_cache:
        _llm_cache[key] = ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=api_key,
            max_retries=2,
        )
    return _llm_cache[key]


def _build_system_prompt(personalization_instructions: str) -> str:
    return TUTOR_SYSTEM_PROMPT.format(
        personalization_instructions=personalization_instructions
    )


def ask(
    question: str,
    vectorstore,
    groq_api_key: str,
    learner_profile: Dict[str, Any],
    personalization_instructions: str = "",
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    eli12: bool = False,
    verbosity: int = 5,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Run the full RAG pipeline: retrieve → check confidence → answer.

    Returns a dict with keys:
      - answer: str
      - docs: list of retrieved (Document, score) pairs
      - low_confidence: bool
      - avg_score: float
    """
    # 1. Retrieve with scores
    docs_with_scores: List[Tuple[Document, float]] = retrieve_with_scores(
        query=question,
        vectorstore=vectorstore,
        top_k=top_k,
    )

    # 2. Check confidence
    if not docs_with_scores:
        return {
            "answer": (
                "I couldn't find any relevant information in the uploaded materials. "
                "Could you rephrase your question or let me know which topic or "
                "chapter you're studying?"
            ),
            "docs": [],
            "low_confidence": True,
            "avg_score": 0.0,
        }

    scores = [s for _, s in docs_with_scores]
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    low_confidence = max_score < DEFAULT_SCORE_THRESHOLD

    if low_confidence:
        return {
            "answer": (
                "I'm not confident I have the right materials to answer this well. "
                "The closest content I found isn't a strong match.\n\n"
                "**Could you:**\n"
                "- Rephrase or add more detail to your question?\n"
                "- Let me know which chapter or topic you're studying?\n\n"
                "I want to make sure I give you accurate information!"
            ),
            "docs": docs_with_scores,
            "low_confidence": True,
            "avg_score": avg_score,
        }

    # 3. Build context string
    context = format_context(docs_with_scores)

    # 4. Construct the prompt
    extra = ""
    if eli12:
        extra += "\nExplain as if the learner is 12 years old — use simple words, fun analogies, and short sentences."
    if verbosity <= 3:
        extra += "\nBe very concise — short paragraphs, bullet points only."
    elif verbosity >= 8:
        extra += "\nBe very detailed — include background, derivations, and extended examples."

    system_prompt = _build_system_prompt(personalization_instructions + extra)

    user_prompt = RAG_ANSWER_TEMPLATE.format(
        context=context,
        learner_name=learner_profile.get("name", "Learner"),
        course=learner_profile.get("course", "General"),
        skill_level=learner_profile.get("skill_level", "Intermediate"),
        goals=learner_profile.get("goals", "Understand the material"),
        question=question,
    )

    # 5. Call the Groq LLM (with retry for rate limits)
    llm = _get_llm(groq_api_key, model, temperature)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = None
    for attempt in range(3):
        try:
            response = llm.invoke(messages)
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = 2 * (attempt + 1)
                time.sleep(wait)
                if attempt == 2:
                    raise RuntimeError(
                        "Groq API rate limit exceeded after retries. "
                        "Please wait a minute and try again."
                    )
            else:
                raise

    if response is None:
        raise RuntimeError("Groq API failed to return a response.")

    return {
        "answer": response.content,
        "docs": docs_with_scores,
        "low_confidence": False,
        "avg_score": avg_score,
    }


def generate_followups(
    question: str,
    answer: str,
    groq_api_key: str,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """Generate 3 unique follow-up questions using Groq LLM.

    Returns a list of 3 question strings, or fallback static questions on error.
    """
    import json
    import re

    prompt = FOLLOWUP_TEMPLATE.format(
        question=question[:500],
        answer=answer[:1000],
    )

    llm = _get_llm(groq_api_key, model, 0.6)

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        questions = json.loads(raw)
        if isinstance(questions, list) and len(questions) >= 3:
            return [str(q) for q in questions[:3]]
    except Exception:
        pass

    # Fallback: static suggestions
    topic = question[:60]
    return [
        f"Can you explain more about '{topic}'?",
        f"What are real-world examples of this?",
        f"What are common misconceptions about this topic?",
    ]

