"""
NextGen Tutor AI – AI Debate Mode (Groq)
==========================================
Provides a devil's advocate response that challenges the student's reasoning
to deepen understanding. Uses the fast model to minimize API load.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from utils.config import FAST_MODEL, DEBATE_SYSTEM_PROMPT


def generate_debate_response(
    topic: str,
    student_argument: str,
    context: str,
    groq_api_key: str,
    model: str = FAST_MODEL,
) -> str:
    """Generate a counter-argument that challenges the student's reasoning.

    Parameters
    ----------
    topic : str
        The topic being debated.
    student_argument : str
        The student's latest argument or claim.
    context : str
        Retrieved course context for factual grounding.
    groq_api_key : str
        User-supplied Groq API key.
    model : str
        Groq model to use (defaults to fast model).

    Returns
    -------
    str
        A counter-argument with educational value.
    """
    prompt = (
        f"Topic: {topic}\n\n"
        f"Course Context:\n{context[:3000]}\n\n"
        f"Student's argument:\n{student_argument}\n\n"
        "Challenge their reasoning with a thoughtful counter-argument. "
        "Stay factual using the course context. End by asking them to defend or refine their position."
    )

    llm = ChatGroq(model=model, temperature=0.6, groq_api_key=groq_api_key)
    try:
        resp = llm.invoke([
            SystemMessage(content=DEBATE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return resp.content.strip()
    except Exception:
        return "Hmm, I couldn't formulate a counter-argument right now. Could you rephrase your point?"
