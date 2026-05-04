"""
NextGen Tutor AI – Podcast Script Generator (Groq)
=====================================================
Converts chat session history into an engaging two-host podcast script.
"""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from utils.config import DEFAULT_MODEL, PODCAST_TEMPLATE


def generate_podcast_script(
    chat_messages: List[Dict[str, str]],
    groq_api_key: str,
    topic_focus: str = "",
    episode_length: str = "Medium (8 min)",
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """Generate a podcast script from chat session history.

    Parameters
    ----------
    chat_messages : list[dict]
        List of ``{"role": "user"|"assistant", "content": "..."}`` dicts.
    groq_api_key : str
        User-supplied Groq API key.
    topic_focus : str
        Optional topic emphasis for the podcast.
    episode_length : str
        One of "Short (3 min)", "Medium (8 min)", "Full (15 min)".
    model : str
        Groq model to use.

    Returns
    -------
    str or None
        The podcast script in markdown dialogue format, or None on failure.
    """
    if not chat_messages:
        return None

    # Build a condensed transcript from chat history
    transcript_lines = []
    for msg in chat_messages:
        role = "Learner" if msg["role"] == "user" else "Tutor"
        content = msg.get("content", "")[:1500]
        transcript_lines.append(f"{role}: {content}")

    transcript = "\n".join(transcript_lines)[:6000]

    length_map = {
        "Short (3 min)": "short (approximately 500 words, 3-4 discussion points)",
        "Medium (8 min)": "medium-length (approximately 1200 words, 5-7 discussion points)",
        "Full (15 min)": "full-length (approximately 2500 words, 8-12 discussion points)",
    }
    length_instruction = length_map.get(episode_length, length_map["Medium (8 min)"])

    focus_instruction = ""
    if topic_focus:
        focus_instruction = f"\nPay special attention to and emphasize this aspect: {topic_focus}"

    prompt = PODCAST_TEMPLATE.format(
        transcript=transcript,
        length_instruction=length_instruction,
        focus_instruction=focus_instruction,
    )

    llm = ChatGroq(
        model=model,
        temperature=0.7,
        groq_api_key=groq_api_key,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        return None


def generate_episode_notes(
    podcast_script: str,
    groq_api_key: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate concise show notes from a podcast script.

    Returns markdown-formatted episode notes with key topics,
    timestamps-style markers, and takeaways.
    """
    llm = ChatGroq(
        model=model,
        temperature=0.3,
        groq_api_key=groq_api_key,
    )

    prompt = (
        "Create concise podcast show notes from this script.\n\n"
        "Include:\n"
        "- Episode summary (2-3 sentences)\n"
        "- Key topics discussed (bulleted list)\n"
        "- Main takeaways (numbered list)\n"
        "- Key terms mentioned\n\n"
        f"Script:\n{podcast_script[:5000]}\n\n"
        "Format as clean markdown."
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        return "Could not generate episode notes."
