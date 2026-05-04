"""
NextGen Tutor AI – Flashcard Generator (Groq)
================================================
Generates interactive study flashcards from document content using Groq LLM.
Each flashcard has a term, definition, example, and category.
"""

import json
import re
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from utils.config import DEFAULT_MODEL, FLASHCARD_TEMPLATE


def generate_flashcards(
    context: str,
    groq_api_key: str,
    num_cards: int = 10,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """Generate flashcards from retrieved context.

    Returns a list of dicts, each with keys:
      id, term, definition, example, category
    """
    llm = ChatGroq(
        model=model,
        temperature=0.4,
        groq_api_key=groq_api_key,
    )

    prompt = FLASHCARD_TEMPLATE.format(
        context=context[:6000],
        num_cards=num_cards,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        cards = json.loads(raw)
        if isinstance(cards, list):
            return cards
    except json.JSONDecodeError:
        pass

    # Fallback: try to find JSON array in the response
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        try:
            cards = json.loads(raw[start:end + 1])
            if isinstance(cards, list):
                return cards
        except json.JSONDecodeError:
            pass

    return []


def flashcards_to_pdf_bytes(cards: List[Dict[str, Any]]) -> bytes:
    """Convert flashcards to a simple PDF. Falls back to plain text."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        import io

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        y = height - 40

        c.setFont("Helvetica-Bold", 18)
        c.drawString(30, y, "NextGen Tutor AI - Flashcards")
        y -= 30

        for card in cards:
            if y < 100:
                c.showPage()
                y = height - 40

            c.setFont("Helvetica-Bold", 12)
            c.drawString(30, y, f"#{card.get('id', '?')}  {card.get('term', 'N/A')}")
            y -= 18

            c.setFont("Helvetica", 10)
            defn = card.get("definition", "")
            for line in _wrap_text(defn, 90):
                c.drawString(40, y, line)
                y -= 14

            example = card.get("example", "")
            if example:
                c.setFont("Helvetica-Oblique", 10)
                c.drawString(40, y, f"Example: {example[:120]}")
                y -= 14

            y -= 10

        c.save()
        return buf.getvalue()

    except ImportError:
        lines = ["NEXTGEN TUTOR AI - FLASHCARDS\n", "=" * 40 + "\n\n"]
        for card in cards:
            lines.append(f"#{card.get('id', '?')}  {card.get('term', 'N/A')}\n")
            lines.append(f"   Definition: {card.get('definition', '')}\n")
            if card.get("example"):
                lines.append(f"   Example: {card['example']}\n")
            lines.append(f"   Category: {card.get('category', '')}\n")
            lines.append("-" * 40 + "\n\n")
        return "".join(lines).encode("utf-8")


def _wrap_text(text: str, max_chars: int = 90) -> list:
    """Simple word-wrap helper."""
    words = text.split()
    lines = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 > max_chars:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        lines.append(current)
    return lines
