"""
NextGen Tutor AI – Timeline Generator (Groq)
===============================================
Generates visual timelines from RAG content for historical/sequential topics.
"""

import json
import re
import time
from typing import Optional, List, Dict, Any

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from utils.config import DEFAULT_MODEL
from rag.retriever import retrieve_with_scores, format_context

TIMELINE_PROMPT = """From the following course material, extract a chronological timeline for the topic "{topic}".
For each event provide:
- order (integer starting from 1)
- title (short event title)
- description (1-2 sentences)
- date_or_period (date, year, or period label — use "Step N" if no dates)
- importance (high, medium, low)

Output as a strict JSON array. No markdown fences, no extra text.

Context:
{context}"""


def generate_timeline(
    topic: str,
    vectorstore,
    groq_api_key: str,
    model: str = DEFAULT_MODEL,
) -> Optional[List[Dict[str, Any]]]:
    """Generate timeline events from RAG content."""
    docs = retrieve_with_scores(
        f"{topic} history timeline events chronological", vectorstore, top_k=5
    )
    if not docs:
        return None
    context = format_context(docs)

    prompt = TIMELINE_PROMPT.format(topic=topic, context=context[:5000])
    llm = ChatGroq(model=model, temperature=0.3, groq_api_key=groq_api_key)

    for attempt in range(3):
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(5 * (attempt + 1))
            else:
                raise
    else:
        return None

    raw = resp.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]

    try:
        events = json.loads(raw)
        return events if isinstance(events, list) else None
    except json.JSONDecodeError:
        return None


def render_timeline(events: List[Dict]):
    """Render a timeline using Streamlit components."""
    import streamlit as st

    importance_colors = {
        "high": "#e53e3e",
        "medium": "#dd6b20",
        "low": "#38a169",
    }

    for i, event in enumerate(events):
        importance = event.get("importance", "medium")
        color = importance_colors.get(importance, "#667eea")
        title = event.get("title", f"Event {i+1}")
        desc = event.get("description", "")
        date = event.get("date_or_period", "")

        # Visual timeline node
        side = "left" if i % 2 == 0 else "right"
        st.markdown(
            f'<div style="border-left: 4px solid {color}; padding: 12px 16px; '
            f'margin: 8px 0; border-radius: 0 8px 8px 0; '
            f'background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);">'
            f'<div style="display: flex; justify-content: space-between; align-items: center;">'
            f'<strong style="color: {color};">{title}</strong>'
            f'<span style="font-size: 0.8em; color: #718096; background: #e2e8f0; '
            f'padding: 2px 8px; border-radius: 12px;">{date}</span>'
            f'</div>'
            f'<p style="margin: 4px 0 0 0; color: #4a5568; font-size: 0.9em;">{desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
