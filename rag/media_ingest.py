"""
NextGen Tutor AI – Media Ingest
=================================
Handles non-PDF file uploads: PPT, images, text files.
Extracts text content for use in the knowledge base and chat tutor.
"""

from typing import Optional

# ── Supported file types ─────────────────────────────────────────────────────

SUPPORTED_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "bmp", "webp"]
SUPPORTED_TEXT_EXTENSIONS = ["txt", "md", "csv", "log"]
SUPPORTED_PPT_EXTENSIONS = ["pptx"]


def get_mime_type(filename: str) -> str:
    """Get MIME type from filename extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "bmp": "image/bmp",
        "webp": "image/webp",
        "txt": "text/plain",
        "md": "text/markdown",
        "csv": "text/csv",
        "log": "text/plain",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    return mime_map.get(ext, "application/octet-stream")


def extract_text_from_text_file(file_bytes: bytes) -> str:
    """Extract text content from a text-based file."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="replace")


def extract_text_from_pptx(file_bytes: bytes) -> Optional[str]:
    """Extract text from a PowerPoint (.pptx) file.

    Extracts text from all slides including titles, body text,
    and notes. Returns concatenated text or None on failure.
    """
    try:
        from pptx import Presentation
        import io

        prs = Presentation(io.BytesIO(file_bytes))
        all_text = []

        for slide_num, slide in enumerate(prs.slides, 1):
            slide_texts = [f"--- Slide {slide_num} ---"]

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())

            # Also extract notes if present
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    slide_texts.append(f"[Notes: {notes}]")

            all_text.append("\n".join(slide_texts))

        return "\n\n".join(all_text) if all_text else None

    except ImportError:
        return None
    except Exception:
        return None


def extract_text_from_image(file_bytes: bytes, groq_api_key: str = "", mime_type: str = "image/png") -> Optional[str]:
    """Attempt OCR on an image.

    Note: Groq does not support vision/multimodal models for OCR.
    This is a stub that returns a helpful message instead.
    For production use, integrate with a dedicated OCR service like
    Tesseract, Google Vision API, or AWS Textract.
    """
    # Groq doesn't support multimodal/vision, so we can't do LLM-based OCR
    # Return a stub message
    return (
        "[Image uploaded — OCR not available with Groq API. "
        "For image text extraction, consider using Tesseract OCR locally: "
        "pip install pytesseract, or use a dedicated OCR service.]"
    )
