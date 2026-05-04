"""
NextGen Tutor AI – Flowchart Generator (Groq)
================================================
Extracts process steps from RAG content and renders Graphviz flowcharts.
"""

import json
import re
import time
from typing import Optional, List, Dict, Any

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from utils.config import DEFAULT_MODEL
from rag.retriever import retrieve_with_scores, format_context

FLOWCHART_PROMPT = """From the following course material extract all steps of the process or algorithm for the topic "{topic}".
For each step provide:
- step_number (integer starting from 1)
- step_title (short title)
- description (1-2 sentences)
- step_type (one of: action, decision, input_output, start, end)
- decision_branches (object with "yes" and "no" as next step numbers, null if not a decision step)
- complexity_notes (null if none)

Output as a strict JSON array. No markdown fences, no extra text.

Context:
{context}"""


def generate_flowchart(
    topic: str,
    vectorstore,
    groq_api_key: str,
    model: str = DEFAULT_MODEL,
) -> Optional[List[Dict[str, Any]]]:
    """Generate flowchart steps from RAG content."""
    docs = retrieve_with_scores(
        f"{topic} process steps algorithm workflow", vectorstore, top_k=5
    )
    if not docs:
        return None
    context = format_context(docs)

    prompt = FLOWCHART_PROMPT.format(topic=topic, context=context[:5000])
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
        steps = json.loads(raw)
        return steps if isinstance(steps, list) else None
    except json.JSONDecodeError:
        return None


def render_flowchart(steps: List[Dict], current_step: int = -1):
    """Render flowchart using Graphviz in Streamlit."""
    import streamlit as st

    type_shapes = {
        "start": ("Mrecord", "#4285F4", "white"),
        "end": ("Mrecord", "#4285F4", "white"),
        "action": ("box", "#34A853", "white"),
        "decision": ("diamond", "#FBBC04", "black"),
        "input_output": ("parallelogram", "#FF6D01", "white"),
    }

    dot = 'digraph G {\n  rankdir=TB;\n  node [fontname="Arial", fontsize=11, style=filled];\n  edge [fontname="Arial", fontsize=9];\n\n'

    for step in steps:
        sn = step.get("step_number", 0)
        title = step.get("step_title", f"Step {sn}")
        stype = step.get("step_type", "action")
        shape, color, fontcolor = type_shapes.get(stype, type_shapes["action"])

        penwidth = "3" if sn == current_step else "1"
        border = "#FF0000" if sn == current_step else color

        label = title.replace('"', '\\"')
        dot += f'  step{sn} [label="{label}", shape={shape}, fillcolor="{color}", fontcolor="{fontcolor}", penwidth={penwidth}, color="{border}"];\n'

    # Edges
    for step in steps:
        sn = step.get("step_number", 0)
        branches = step.get("decision_branches")
        if branches and isinstance(branches, dict):
            yes_next = branches.get("yes")
            no_next = branches.get("no")
            if yes_next:
                dot += f'  step{sn} -> step{yes_next} [label="Yes", color="#34A853"];\n'
            if no_next:
                dot += f'  step{sn} -> step{no_next} [label="No", color="#EA4335"];\n'
        else:
            next_steps = [s for s in steps if s.get("step_number", 0) == sn + 1]
            if next_steps:
                dot += f'  step{sn} -> step{sn + 1};\n'

    dot += "}\n"

    try:
        st.graphviz_chart(dot)
    except Exception:
        st.code(dot, language="dot")

    # Show current step description
    if current_step >= 0:
        current = next((s for s in steps if s.get("step_number") == current_step), None)
        if current:
            st.info(
                f"**Step {current_step}: {current.get('step_title', '')}**\n\n"
                f"{current.get('description', '')}"
            )
