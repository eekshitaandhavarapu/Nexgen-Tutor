"""
NextGen Tutor AI – Mind Map Generator (Groq)
===============================================
Generates a hierarchical mind map from document content using Groq LLM.
Outputs both JSON (for interactive rendering) and Mermaid syntax (for diagrams).
"""

import json
import re
from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from utils.config import DEFAULT_MODEL, MINDMAP_TEMPLATE


def generate_mindmap(
    context: str,
    groq_api_key: str,
    model: str = DEFAULT_MODEL,
) -> Optional[Dict[str, Any]]:
    """Generate a mind map hierarchy from content.

    Returns a nested dict with keys: title, children
    """
    llm = ChatGroq(
        model=model,
        temperature=0.4,
        groq_api_key=groq_api_key,
    )

    prompt = MINDMAP_TEMPLATE.format(context=context[:6000])
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "title" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: find JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(raw[start:end + 1])
            if isinstance(data, dict) and "title" in data:
                return data
        except json.JSONDecodeError:
            pass

    return None


def mindmap_to_mermaid(node: Dict[str, Any], direction: str = "TD") -> str:
    """Convert a mind map JSON hierarchy to Mermaid mindmap syntax."""
    lines = ["mindmap"]
    _build_mermaid_tree(node, lines, depth=1)
    return "\n".join(lines)


def _build_mermaid_tree(node: Dict[str, Any], lines: list, depth: int):
    """Recursively build Mermaid mindmap lines."""
    indent = "  " * depth
    title = _sanitize_label(node.get("title", ""))

    if depth == 1:
        lines.append(f"{indent}root(({title}))")
    else:
        lines.append(f"{indent}{title}")

    for child in node.get("children", []):
        _build_mermaid_tree(child, lines, depth + 1)


def _sanitize_label(text: str) -> str:
    """Remove characters that break Mermaid syntax."""
    for ch in ['(', ')', '[', ']', '{', '}', '"', "'", "<", ">", "`"]:
        text = text.replace(ch, "")
    return text.strip()


def mindmap_to_markdown(node: Dict[str, Any], depth: int = 0) -> str:
    """Convert mind map to indented markdown for text display."""
    prefix = "  " * depth
    if depth == 0:
        bullet = "# "
    elif depth == 1:
        bullet = "- **"
    else:
        bullet = "- "

    title = node.get("title", "")
    if depth == 1:
        line = f"{prefix}{bullet}{title}**\n"
    else:
        line = f"{prefix}{bullet}{title}\n"

    for child in node.get("children", []):
        line += mindmap_to_markdown(child, depth + 1)

    return line
