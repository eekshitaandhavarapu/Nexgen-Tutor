"""
NextGen Tutor AI – Centralised Configuration
==============================================
All tuneable parameters live here. Values are read from environment variables
(via .env) when available, otherwise sensible defaults are used.

The Groq API key is expected to be supplied at runtime through the Streamlit
UI and stored **only** in session state (never written to disk).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # optional .env support

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_store")
LOG_DIR = str(PROJECT_ROOT / "logs")

# ── Chunking defaults (educators can customise these) ────────────────────────
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))      # characters
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # characters

# ── Retriever defaults ───────────────────────────────────────────────────────
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
DEFAULT_SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))

# ── LLM defaults (Groq) ─────────────────────────────────────────────────────
# Configurable: "llama3-70b-8192" for depth, "llama3-8b-8192" for speed
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# ── Embedding model (local HuggingFace — no API key required) ────────────────
# Groq does not offer an embedding endpoint, so we use a local model.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Collection name for ChromaDB ─────────────────────────────────────────────
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "nexgen_tutor")

# ── Available Groq models (shown in UI dropdown) ─────────────────────────────
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

# ── Prompt Templates ─────────────────────────────────────────────────────────

TUTOR_SYSTEM_PROMPT = """You are **NextGen Tutor**, a patient, encouraging, and knowledgeable AI tutor.

RULES YOU MUST FOLLOW:
1. Base your answers **only** on the retrieved context provided below.
2. **Cite your sources** — always mention the PDF name, page number, and a short snippet.
3. If the context does not contain enough information to answer confidently, say:
   "I'm not sure about this based on the materials I have. Could you rephrase your question or tell me which topic you're studying?"
4. Never fabricate facts. If you are uncertain, say so.
5. Structure your response with clear headings.
6. Check for understanding at the end of your answer by asking a reflective question.

{personalization_instructions}
"""

RAG_ANSWER_TEMPLATE = """Use the retrieved context below to answer the learner's question.

### Retrieved Context
{context}

### Learner Profile
- Name: {learner_name}
- Course: {course}
- Skill Level: {skill_level}
- Goals: {goals}

### Learner's Question
{question}

---
Respond with the following clearly-labelled sections (use markdown headings):
## Answer
(step-by-step, clear explanation in simple language)

## Key Concepts
(bulleted list of the core concepts involved)

## Example
(a concrete example that illustrates the answer)

## Common Mistakes
(pitfalls learners often fall into on this topic)

## Citations
(list each source: PDF name · page number · relevant snippet)

## What to Learn Next
(3 bullet suggestions for deeper study)
"""

QUIZ_GENERATION_TEMPLATE = """Based **strictly** on the following context, generate {num_questions} quiz questions.

### Context
{context}

### Difficulty Level: {difficulty}

Return your response as a valid JSON array. Each element must have:
- "id": integer (1-based)
- "type": "mcq" or "short_answer"
- "question": string
- "options": list of 4 strings (only for mcq, null for short_answer)
- "correct_answer": string
- "difficulty": "easy" | "medium" | "hard"
- "explanation": string (brief explanation of the correct answer)

Mix question types: roughly 60% MCQ, 40% short-answer.
Return ONLY the JSON array — no markdown fences, no extra text.
"""

GRADING_TEMPLATE = """You are a fair grader. Evaluate the learner's answers against the reference answers.

### Questions & Reference Answers
{questions_json}

### Learner's Answers
{answers_json}

For each question, return a JSON array where each element has:
- "id": question id
- "is_correct": boolean
- "score": float 0-1 (partial credit allowed for short_answer)
- "feedback": string (specific, helpful feedback)
- "hint": string (a hint for improvement if incorrect)
- "retry_recommended": boolean

Return ONLY the JSON array — no markdown fences, no extra text.
"""

# ── Flashcard Template ───────────────────────────────────────────────────────

FLASHCARD_TEMPLATE = """Based on the following course material, generate {num_cards} study flashcards.

### Context
{context}

Return a valid JSON array where each element has:
- "id": integer (1-based)
- "term": string (the concept or keyword)
- "definition": string (clear, concise definition)
- "example": string (a concrete example)
- "category": string (topic category)

Return ONLY the JSON array — no markdown fences, no extra text.
"""

# ── Mind Map Template ────────────────────────────────────────────────────────

MINDMAP_TEMPLATE = """From the following course material, create a hierarchical mind map.

### Context
{context}

Return a valid JSON object with this recursive structure:
{{
    "title": "Main Topic",
    "children": [
        {{
            "title": "Subtopic 1",
            "children": [
                {{"title": "Detail A", "children": []}},
                {{"title": "Detail B", "children": []}}
            ]
        }}
    ]
}}

Create 3-5 main branches with 2-4 sub-items each. Maximum depth: 3.
Return ONLY the JSON object — no markdown fences, no extra text.
"""

# ── Follow-Up Question Template ──────────────────────────────────────────────

FOLLOWUP_TEMPLATE = """Based on the following Q&A exchange, generate 3 unique follow-up questions
that would deepen the learner's understanding. Each question should explore a different angle.

### Previous Question
{question}

### Previous Answer
{answer}

Return exactly 3 follow-up questions as a JSON array of strings.
Make them specific, thought-provoking, and directly related to the topic.
Return ONLY the JSON array — no markdown fences, no extra text.
"""

# ── Podcast Script Template ──────────────────────────────────────────────────

PODCAST_TEMPLATE = """Convert the following tutoring session transcript into an engaging podcast episode
between two hosts: **Alex** (the explainer) and **Sam** (the curious learner).

### Tutoring Session Transcript
{transcript}

### Episode Guidelines
- Create a {length_instruction} episode
- Start with a catchy intro where Sam sets up the topic
- Alex explains concepts clearly using the session content
- Sam asks clarifying questions, shares relatable analogies
- Include moments of humor and "aha!" reactions
- End with a quick recap and one actionable takeaway
{focus_instruction}

### Format Rules
- Use **Alex:** and **Sam:** speaker labels on separate lines
- Use *italics* for tone/stage directions (e.g., *laughs*, *excited*)
- Add section markers: **[INTRO]**, **[MAIN DISCUSSION]**, **[RECAP]**, **[OUTRO]**
- Keep the tone conversational, energetic, and educational

Write the complete podcast script now.
"""

# ── Quiz Hint Template ───────────────────────────────────────────────────────

QUIZ_HINT_TEMPLATE = """You are a helpful tutor. A learner is stuck on a quiz question.
Give a short, conceptual hint that guides them toward the answer WITHOUT revealing it.

### Question
{question}

### Context the question was based on
{context}

Rules:
- Do NOT state the answer directly
- Point the learner to the relevant concept or reasoning approach
- Keep it to 1-3 sentences
- Be encouraging

Respond with ONLY the hint text — no JSON, no fences.
"""

# ── Debate System Prompt ─────────────────────────────────────────────────────

DEBATE_SYSTEM_PROMPT = """You are an AI Devil's Advocate for an educational platform.
Your goal is to respectfully challenge the student's reasoning to deepen their understanding of the topic.
Do not be overly aggressive, but maintain a firm opposing or critical stance to make them think.
"""
