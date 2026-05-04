"""
NextGen Tutor AI – URL Content Ingest (Groq)
===============================================
Fetches and extracts text from YouTube videos and blog/web URLs.
- YouTube: uses youtube-transcript-api for transcripts
- Blogs: uses trafilatura for clean text extraction
- Summarization via Groq LLM in multiple styles including Podcast Script
"""

import re
from typing import Optional, Dict, Any

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from utils.config import DEFAULT_MODEL


# ── YouTube helpers ──────────────────────────────────────────────────────────

_YT_REGEX = re.compile(
    r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})'
)


def is_youtube_url(url: str) -> bool:
    return bool(_YT_REGEX.search(url))


def extract_youtube_id(url: str) -> Optional[str]:
    m = _YT_REGEX.search(url)
    return m.group(1) if m else None


def fetch_youtube_transcript(url: str) -> Optional[str]:
    """Fetch the transcript of a YouTube video."""
    video_id = extract_youtube_id(url)
    if not video_id:
        return None

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        return " ".join([snippet.text for snippet in transcript])
    except Exception:
        return None


# ── Blog/Web helpers ─────────────────────────────────────────────────────────

def fetch_blog_content(url: str) -> Optional[str]:
    """Extract clean readable text from a blog or web page."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            return text
    except Exception:
        pass
    return None


# ── Unified URL processor ───────────────────────────────────────────────────

def fetch_url_content(url: str) -> Dict[str, Any]:
    """Fetch content from a URL (YouTube or web page).

    Returns dict with keys: type, content, error
    """
    url = url.strip()

    if is_youtube_url(url):
        text = fetch_youtube_transcript(url)
        if text:
            return {"type": "youtube", "content": text, "error": None}
        else:
            return {"type": "youtube", "content": None,
                    "error": "Could not fetch YouTube transcript. The video may not have captions."}

    # Try as web page
    text = fetch_blog_content(url)
    if text:
        return {"type": "blog", "content": text, "error": None}
    else:
        return {"type": "blog", "content": None,
                "error": "Could not extract content from URL. It may be behind a paywall or use heavy JavaScript."}


# ── Summarization styles ────────────────────────────────────────────────────

SUMMARY_STYLES = {
    "Concise": "Create a brief, to-the-point summary in 5-8 bullet points.",
    "Detailed": "Create a comprehensive, detailed summary with sections, key ideas, and examples.",
    "Child-friendly": "Explain this in simple terms a 12-year-old would understand. Use fun analogies.",
    "Technical": "Create a technical summary with precise terminology, key findings, and methodology.",
    "Podcast Script": (
        "Convert this into an engaging podcast script between two hosts (Host A and Host B). "
        "They should discuss the main ideas in a conversational, educational tone. "
        "Include an intro, main discussion points, and a closing summary. "
        "Format as a dialogue with speaker labels."
    ),
    "Timeline": "Create a chronological timeline of the key points discussed.",
    "Key Ideas": "Extract the 5-10 most important ideas or takeaways as numbered points.",
}


def summarize_content(
    content: str,
    groq_api_key: str,
    style: str = "Concise",
    content_type: str = "article",
    model: str = DEFAULT_MODEL,
) -> str:
    """Summarize content using Groq LLM with a given style."""
    llm = ChatGroq(
        model=model,
        temperature=0.3,
        groq_api_key=groq_api_key,
    )

    style_instruction = SUMMARY_STYLES.get(style, SUMMARY_STYLES["Concise"])

    prompt = (
        f"Summarize the following {content_type} content.\n\n"
        f"**Style:** {style_instruction}\n\n"
        f"**Content:**\n{content[:8000]}\n\n"
        f"Format your response with clear markdown headings and structure."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
