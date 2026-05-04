"""
NextGen Tutor AI – Translator (Groq)
======================================
Uses Groq LLM for multilingual translation of platform content.
"""

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from utils.config import DEFAULT_MODEL

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Russian": "ru",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Bengali": "bn",
    "Marathi": "mr",
}


def translate_text(
    text: str,
    target_language: str,
    groq_api_key: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Translate text to the target language using Groq.

    Args:
        text: text to translate
        target_language: target language name (e.g., "Hindi", "Spanish")
        groq_api_key: Groq API key

    Returns:
        Translated text
    """
    if target_language == "English":
        return text

    llm = ChatGroq(
        model=model,
        temperature=0.2,
        groq_api_key=groq_api_key,
    )

    prompt = (
        f"Translate the following text to {target_language}. "
        f"Preserve all formatting (markdown, bullet points, headings). "
        f"Do NOT add any commentary or explanation — return ONLY the translated text.\n\n"
        f"{text[:8000]}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()
