"""
NextGen Tutor: Adaptive Learning Chatbot (Groq Edition) — Expanded
====================================================================
Main Streamlit application with 8 tabs.  Run with:

    streamlit run app.py

Tabs:
  1. Setup           – API key, model, learner profile, language, PDF/PPT upload
  2. Chat Tutor      – RAG Q&A with voice input, translations, dynamic follow-ups
  3. Practice Quiz   – Micro-quizzes with grading & feedback
  4. Flashcards      – Auto-generated flip-cards with export
  5. Visual Learn    – Mind maps, flowcharts, timelines
  6. Summarize URL   – YouTube/blog extraction, podcast-style summaries
  7. Code Arena      – Coding problems, editor, tests, timed contests
  8. Insights        – Metrics, quiz trends, log export
"""

# ── Streamlit Cloud: ChromaDB requires a newer SQLite than the cloud provides ──
# This swap MUST happen before any chromadb import.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # Not on Streamlit Cloud (or pysqlite3 not installed) — no swap needed

import json
import os
import time as _time
import streamlit as st
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ── Module imports ───────────────────────────────────────────────────────────
from rag.ingest import (
    build_vector_store, chunk_documents, clear_vector_store,
    get_existing_hashes, load_existing_vector_store, load_pdfs,
)
from rag.retriever import format_context, retrieve_with_scores
from rag.chain import ask, generate_followups
from rag.url_ingest import fetch_url_content, summarize_content, SUMMARY_STYLES
from rag.media_ingest import (
    extract_text_from_pptx, extract_text_from_text_file,
    extract_text_from_image, SUPPORTED_PPT_EXTENSIONS,
    SUPPORTED_TEXT_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS,
)
from tutor.personalize import (
    LearnerProfile, SessionMemory, get_personalization_instructions,
)
from tutor.quiz import generate_quiz, generate_hint
from tutor.grader import grade_answers
from tutor.flashcards import generate_flashcards, flashcards_to_pdf_bytes
from tutor.mindmap import generate_mindmap, mindmap_to_mermaid, mindmap_to_markdown
from tutor.podcast import generate_podcast_script, generate_episode_notes
from tutor.debate import generate_debate_response
from visual.flowchart_generator import generate_flowchart, render_flowchart
from visual.timeline_generator import generate_timeline, render_timeline
from arena.problem_engine import generate_problem, get_available_topics, SUPPORTED_LANGUAGES
from arena.executor import run_all_tests
from arena.contest import create_contest, get_remaining_time, is_contest_active, submit_solution, end_contest, format_time
from arena.coach import get_hint, review_code, get_solution
from arena.models import get_leaderboard, get_rating, update_rating
from eval.metrics import compute_rouge_l, compute_bleu, flag_hallucination_risk
from eval.logger import (
    log_interaction, log_quiz_result, read_interactions,
    read_quiz_results, export_logs,
)
from utils.config import (
    AVAILABLE_MODELS, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL, FAST_MODEL, DEFAULT_SCORE_THRESHOLD, DEFAULT_TEMPERATURE, DEFAULT_TOP_K,
)
from utils.translator import translate_text, SUPPORTED_LANGUAGES as SUPPORTED_LANGS


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NextGen Tutor · Adaptive Learning Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
# Preload font (non-blocking) instead of @import (render-blocking)
st.markdown('<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">',
            unsafe_allow_html=True)
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 14px; font-weight: 500; font-size: 0.85rem; }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px; padding: 1.2rem; text-align: center; margin-bottom: 0.8rem;
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; color: #4a5568; }
    .metric-card p { margin: 0.2rem 0 0 0; color: #718096; font-size: 0.85rem; }
    .correct-answer {
        background: #f0fff4; border-left: 4px solid #38a169;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .wrong-answer {
        background: #fff5f5; border-left: 4px solid #e53e3e;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .flashcard {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px; padding: 2rem; color: white; min-height: 180px;
        display: flex; align-items: center; justify-content: center;
        margin: 0.5rem 0; text-align: center; font-size: 1.1rem;
        cursor: pointer; transition: transform 0.3s ease;
    }
    .flashcard:hover { transform: scale(1.02); }
    .flashcard-back {
        background: linear-gradient(135deg, #f5f7fa 0%, #e2e8f0 100%);
        border-radius: 16px; padding: 2rem; color: #2d3748; min-height: 180px;
        margin: 0.5rem 0; border: 2px solid #667eea;
    }
    .stButton > button { transition: all 0.2s ease; border-radius: 8px; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    /* Login page styles */
    .login-container {
        max-width: 480px; margin: 0 auto; padding: 2rem;
    }
    .login-header {
        text-align: center; margin-bottom: 1rem;
        color: #667eea; font-size: 1rem; opacity: 0.9;
    }
    .login-title {
        font-size: 2rem; font-weight: 700; margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        # Authentication
        "logged_in": False,
        "current_user": "",
        "login_mode": "login",  # "login" or "signup"
        "registered_users": {"admin": "admin123", "student": "student123"},
        # App state
        "groq_api_key": st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")),
        "selected_model": DEFAULT_MODEL,
        "vectorstore": None,
        "learner_profile": LearnerProfile(),
        "session_memory": SessionMemory(),
        "chat_messages": [],
        "quiz_questions": [], "quiz_answers": {}, "quiz_grades": [],
        "quiz_submitted": False,
        "last_context": "", "last_docs_with_scores": [],
        "file_hashes": {}, "kb_built": False,
        "chunk_size": DEFAULT_CHUNK_SIZE, "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "top_k": DEFAULT_TOP_K, "temperature": DEFAULT_TEMPERATURE,
        "eli12": False, "verbosity": 5, "pending_followup": None,
        # New features
        "native_language": "English",
        "flashcards": [], "flashcard_index": 0, "flashcard_flipped": False,
        "mindmap_data": None, "flowchart_steps": None, "timeline_events": None,
        "arena_problem": None, "arena_code": "", "arena_results": None,
        "arena_attempted": False,
        "contest": None,
        "media_extracted_text": "",
        # Podcast
        "podcast_script": None, "podcast_notes": None,
        # Quiz hints
        "quiz_hints": {}, "quiz_hints_used": 0,
        # Spaced repetition
        "spaced_review_queue": [],
        # Debate mode
        "debate_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


def _get_vectorstore():
    """Get vectorstore from session state, loading once if needed."""
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = load_existing_vector_store()
    return st.session_state.vectorstore


# ── Login / Sign-Up Page ─────────────────────────────────────────────────────
def show_login_page():
    """Render the login / sign-up page and return True if user is authenticated."""
    st.markdown('<p class="login-header">Adaptive Learning Chatbot · Personalized Tutoring</p>',
                unsafe_allow_html=True)

    # Login / Sign Up toggle buttons
    col_login_btn, col_signup_btn = st.columns(2)
    with col_login_btn:
        if st.button("🔑 Login", use_container_width=True,
                     type="primary" if st.session_state.login_mode == "login" else "secondary"):
            st.session_state.login_mode = "login"
            st.rerun()
    with col_signup_btn:
        if st.button("📝 Sign Up", use_container_width=True,
                     type="primary" if st.session_state.login_mode == "signup" else "secondary"):
            st.session_state.login_mode = "signup"
            st.rerun()

    st.divider()

    if st.session_state.login_mode == "login":
        st.markdown('<h1 class="login-title">Welcome Back!</h1>', unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter your username", key="login_user")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")

        if st.button("🚀 Login", use_container_width=True, type="primary"):
            users = st.session_state.registered_users
            if username and password:
                if username in users and users[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.success(f"Welcome back, **{username}**! 🎉")
                    _time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")
            else:
                st.warning("Please enter both username and password.")

    else:  # signup mode
        st.markdown('<h1 class="login-title">Create Account</h1>', unsafe_allow_html=True)
        new_user = st.text_input("Choose a Username", placeholder="Enter a username", key="signup_user")
        new_pass = st.text_input("Choose a Password", type="password", placeholder="Enter a password", key="signup_pass")
        confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="signup_confirm")

        if st.button("📝 Sign Up", use_container_width=True, type="primary", key="signup_submit"):
            if not new_user or not new_pass:
                st.warning("Please fill in all fields.")
            elif new_pass != confirm_pass:
                st.error("❌ Passwords do not match.")
            elif new_user in st.session_state.registered_users:
                st.error("❌ Username already taken. Choose another.")
            else:
                st.session_state.registered_users[new_user] = new_pass
                st.success(f"Account created for **{new_user}**! Please log in. ✅")
                st.session_state.login_mode = "login"
                _time.sleep(0.5)
                st.rerun()


# ── Gate: show login or main app ─────────────────────────────────────────────
if not st.session_state.logged_in:
    show_login_page()
    st.stop()

# ── Header (only shown after login) ─────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 NextGen Tutor</h1>
    <p>Adaptive Learning Chatbot · Personalized Tutoring · Powered by Groq + RAG</p>
</div>
""", unsafe_allow_html=True)

# Logout button in sidebar
with st.sidebar:
    st.markdown(f"👤 Logged in as **{st.session_state.current_user}**")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        st.rerun()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_setup, tab_chat, tab_quiz, tab_flash, tab_visual, tab_url, tab_pdfsummary, tab_podcast, tab_arena, tab_insights = st.tabs([
    "⚙️ Setup", "💬 Chat Tutor", "📝 Practice Quiz", "🃏 Flashcards",
    "🎨 Visual Learn", "📺 Summarize URL", "📄 PDF Summary", "🎙️ Podcast", "🏟️ Code Arena", "📊 Insights",
])


# ━━━━━━━━━━━━━━━━━━━━ TAB 1 – SETUP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_setup:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🔑 API Configuration")
        api_key = st.text_input(
            "Groq API Key", type="password", value=st.session_state.groq_api_key,
            help="Get your key from https://console.groq.com",
        )
        if api_key:
            st.session_state.groq_api_key = api_key

        st.session_state.selected_model = st.selectbox(
            "Groq Model", AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.selected_model)
            if st.session_state.selected_model in AVAILABLE_MODELS else 0,
        )

        # Language selector
        st.session_state.native_language = st.selectbox(
            "🌐 Response Language",
            list(SUPPORTED_LANGS.keys()),
            index=list(SUPPORTED_LANGS.keys()).index(st.session_state.native_language)
            if st.session_state.native_language in SUPPORTED_LANGS else 0,
            help="Chat Tutor will respond in this language.",
        )

        st.divider()
        st.subheader("👤 Learner Profile")
        name = st.text_input("Your Name", value=st.session_state.learner_profile.name)
        course = st.text_input("Course / Subject", value=st.session_state.learner_profile.course)
        skill = st.selectbox("Skill Level", ["Beginner", "Intermediate", "Advanced"],
            index=["Beginner", "Intermediate", "Advanced"].index(st.session_state.learner_profile.skill_level))
        goals = st.text_area("Learning Goals", value=st.session_state.learner_profile.goals, height=80)
        st.session_state.learner_profile = LearnerProfile(name=name, course=course, skill_level=skill, goals=goals)

    with col_right:
        st.subheader("📄 Knowledge Base")

        # PDF upload
        uploaded_pdfs = st.file_uploader("Upload course PDFs", type=["pdf"], accept_multiple_files=True)

        # Multimedia upload section
        st.markdown("##### 📎 Additional Media (PPT, Text, Images)")
        uploaded_media = st.file_uploader(
            "Upload PPT, text, or image files",
            type=SUPPORTED_PPT_EXTENSIONS + SUPPORTED_TEXT_EXTENSIONS + SUPPORTED_IMAGE_EXTENSIONS,
            accept_multiple_files=True,
            key="media_upload",
        )

        if uploaded_media:
            extract_btn = st.button("📤 Extract Text from Media", type="secondary")
            if extract_btn:
                all_text = []
                for f in uploaded_media:
                    ext = f.name.rsplit(".", 1)[-1].lower()
                    data = f.read()
                    if ext in SUPPORTED_PPT_EXTENSIONS:
                        txt = extract_text_from_pptx(data)
                        if txt:
                            all_text.append(f"--- {f.name} ---\n{txt}")
                            st.success(f"✅ Extracted text from {f.name}")
                        else:
                            st.warning(f"⚠️ Could not extract text from {f.name}. Install python-pptx.")
                    elif ext in SUPPORTED_TEXT_EXTENSIONS:
                        txt = extract_text_from_text_file(data)
                        all_text.append(f"--- {f.name} ---\n{txt}")
                        st.success(f"✅ Loaded {f.name}")
                    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
                        txt = extract_text_from_image(data)
                        all_text.append(f"--- {f.name} ---\n{txt}")
                        st.info(f"ℹ️ {f.name}: Image uploaded (OCR not available with Groq)")
                st.session_state.media_extracted_text = "\n\n".join(all_text)
                if all_text:
                    with st.expander("📋 Extracted Text Preview"):
                        st.text(st.session_state.media_extracted_text[:2000])

        with st.expander("⚙️ Chunking Configuration", expanded=False):
            st.session_state.chunk_size = st.slider("Chunk size", 200, 3000, st.session_state.chunk_size, 100)
            st.session_state.chunk_overlap = st.slider("Chunk overlap", 0, 500, st.session_state.chunk_overlap, 50)

        col_build, col_clear = st.columns(2)
        with col_build:
            build_btn = st.button("🏗️ Build / Update KB", type="primary",
                use_container_width=True, disabled=not st.session_state.groq_api_key)
        with col_clear:
            clear_btn = st.button("🗑️ Clear Vector Store", use_container_width=True)

        if build_btn:
            if not uploaded_pdfs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.status("Building knowledge base…", expanded=True) as status:
                    st.write("📥 Loading PDFs…")
                    docs, hashes = load_pdfs(uploaded_pdfs,
                        course_id=st.session_state.learner_profile.course,
                        uploaded_by=st.session_state.learner_profile.name)
                    st.write(f"✅ Loaded {len(docs)} pages from {len(uploaded_pdfs)} file(s)")
                    st.write("✂️ Chunking…")
                    chunks = chunk_documents(docs, st.session_state.chunk_size, st.session_state.chunk_overlap)
                    st.write(f"✅ Created {len(chunks)} chunks")

                    existing_hashes = set()
                    existing_vs = load_existing_vector_store()
                    if existing_vs:
                        existing_hashes = get_existing_hashes(existing_vs)

                    st.write("📤 Embedding & storing…")
                    vs = build_vector_store(chunks, existing_hashes=existing_hashes)
                    st.session_state.vectorstore = vs
                    st.session_state.file_hashes.update(hashes)
                    st.session_state.kb_built = True
                    status.update(label="Knowledge base ready! ✅", state="complete")
                st.success(f"KB built with {len(chunks)} chunks from {len(uploaded_pdfs)} PDF(s).")

        if clear_btn:
            clear_vector_store()
            st.session_state.vectorstore = None
            st.session_state.kb_built = False
            st.session_state.file_hashes = {}
            st.success("Vector store cleared.")

        if st.session_state.kb_built:
            st.markdown("🟢 **Knowledge base is active**")
        else:
            st.markdown("🔴 **No knowledge base loaded**")


# ━━━━━━━━━━━━━━━━━━━━ TAB 2 – CHAT TUTOR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_chat:
    if not st.session_state.groq_api_key:
        st.warning("Please enter your Groq API key in the ⚙️ Setup tab.")
    elif not st.session_state.kb_built and not st.session_state.vectorstore:
        st.warning("Please build a knowledge base first in the ⚙️ Setup tab.")
    else:
        with st.sidebar:
            st.markdown("### 🎛️ Tutor Controls")
            st.session_state.top_k = st.slider("Retrieval top-k", 1, 15, st.session_state.top_k)
            st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
            st.session_state.eli12 = st.toggle("Explain like I'm 12 🧒", st.session_state.eli12)
            st.session_state.verbosity = st.slider("Verbosity", 1, 10, st.session_state.verbosity)
            st.markdown("---")
            st.session_state.debate_mode = st.toggle("🎭 Debate Mode", st.session_state.debate_mode,
                help="When ON, the tutor challenges your reasoning instead of just answering.")
            if st.session_state.debate_mode:
                st.caption("⚡ Using fast model for debates")

        # Chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("citations"):
                    with st.expander("📚 View Citations & Sources"):
                        for cit in msg["citations"]:
                            st.info(f"📄 **{cit['source']}** — Page {cit['page']}\n*Relevance: {cit['score']:.2f}*\n\n{cit['snippet'][:300]}…")

                if msg["role"] == "assistant" and msg.get("suggestions"):
                    st.markdown("**💡 What to explore next:**")
                    for i, s in enumerate(msg["suggestions"]):
                        if st.button(f"➤ {s}", key=f"hist_fu_{id(msg)}_{i}"):
                            st.session_state.pending_followup = s
                            st.rerun()

        # Voice input
        voice_text = None
        try:
            from audio_recorder_streamlit import audio_recorder
            st.markdown("##### 🎙️ Tap to Speak")
            audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=16000, key="voice_input")
            if audio_bytes:
                try:
                    import speech_recognition as sr
                    import io, wave, tempfile
                    # Save audio bytes to temp WAV file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(tmp_path) as source:
                        audio_data = recognizer.record(source)
                    voice_text = recognizer.recognize_google(audio_data)
                    st.success(f"🎤 Recognized: *{voice_text}*")
                    os.unlink(tmp_path)
                except Exception as e:
                    st.warning(f"Could not recognize speech: {e}")
        except ImportError:
            pass  # audio_recorder_streamlit not installed

        # Handle pending follow-up or voice input
        pending = st.session_state.pending_followup
        if pending:
            st.session_state.pending_followup = None
            user_input = pending
        elif voice_text:
            user_input = voice_text
        else:
            user_input = st.chat_input("Ask me anything about your course material…")

        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            st.session_state.session_memory.add_message("user", user_input)

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    profile = st.session_state.learner_profile
                    memory = st.session_state.session_memory
                    pers = get_personalization_instructions(profile, memory)

                    vs = _get_vectorstore()

                    # Debate mode: use fast model + debate response
                    if st.session_state.debate_mode:
                        try:
                            from utils.config import FAST_MODEL
                            # Get context for grounding
                            result = ask(
                                question=user_input, vectorstore=vs,
                                groq_api_key=st.session_state.groq_api_key,
                                learner_profile=profile.to_dict(),
                                personalization_instructions=pers,
                                top_k=st.session_state.top_k,
                                temperature=st.session_state.temperature,
                                model=FAST_MODEL,
                            )
                            debate_answer = generate_debate_response(
                                topic=memory.get_recent_topic() or "General",
                                student_argument=user_input,
                                context=result.get("context", ""),
                                groq_api_key=st.session_state.groq_api_key,
                            )
                            answer = f"🎭 **Debate Mode**\n\n{debate_answer}"
                            st.markdown(answer)
                            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                            st.session_state.session_memory.add_message("assistant", answer)
                        except Exception as e:
                            st.error(f"❌ Debate error: {e}")
                        st.stop()

                    try:
                        result = ask(
                            question=user_input, vectorstore=vs,
                            groq_api_key=st.session_state.groq_api_key,
                            learner_profile=profile.to_dict(),
                            personalization_instructions=pers,
                            top_k=st.session_state.top_k,
                            temperature=st.session_state.temperature,
                            eli12=st.session_state.eli12,
                            verbosity=st.session_state.verbosity,
                            model=st.session_state.selected_model,
                        )
                    except Exception as e:
                        err = str(e)
                        if "429" in err or "rate_limit" in err.lower():
                            st.error("⏳ **Rate limit reached.** Please wait 30-60s and try again.")
                        else:
                            st.error(f"❌ Error: {e}")
                        st.stop()

                    answer = result["answer"]

                    # Translate if needed
                    lang = st.session_state.native_language
                    if lang != "English":
                        try:
                            answer = translate_text(answer, lang, st.session_state.groq_api_key,
                                                    model=st.session_state.selected_model)
                        except Exception:
                            pass  # fallback to English

                    st.markdown(answer)

                    # Citations
                    citations = []
                    if result["docs"]:
                        for doc, score in result["docs"]:
                            citations.append({
                                "source": doc.metadata.get("source", "Unknown"),
                                "page": doc.metadata.get("page", "?"),
                                "score": score,
                                "snippet": doc.page_content[:300],
                            })
                        with st.expander("📚 View Citations & Sources"):
                            for cit in citations:
                                st.info(f"📄 **{cit['source']}** — Page {cit['page']}\n*Relevance: {cit['score']:.2f}*\n\n{cit['snippet']}…")

                    # Dynamic follow-ups via fast LLM (8B for speed)
                    try:
                        suggestions = generate_followups(
                            question=user_input,
                            answer=result["answer"][:500],
                            groq_api_key=st.session_state.groq_api_key,
                            model=FAST_MODEL,
                        )
                    except Exception:
                        suggestions = [
                            f"Tell me more about this topic",
                            f"Give me examples related to this",
                            f"What are common mistakes here?",
                        ]

                    st.markdown("**💡 What to explore next:**")
                    for i, s in enumerate(suggestions):
                        if st.button(f"➤ {s}", key=f"fu_{i}"):
                            st.session_state.pending_followup = s
                            st.rerun()

                    # Save context
                    if result["docs"]:
                        st.session_state.last_context = format_context(result["docs"])
                        st.session_state.last_docs_with_scores = result["docs"]

                    memory.record_concept(user_input[:80])
                    memory.add_message("assistant", answer[:500])

                    # Log (cap context to 500 tokens for fast ROUGE/BLEU)
                    if result["docs"] and not result.get("low_confidence"):
                        ctx = format_context(result["docs"])
                        ctx_capped = " ".join(ctx.split()[:500])
                        rouge = compute_rouge_l(answer, ctx_capped)
                        bleu = compute_bleu(answer, ctx_capped)
                        hallu = flag_hallucination_risk(answer, ctx_capped)
                        sources = [d.metadata.get("source", "Unknown") for d, _ in result["docs"]]
                        log_interaction(query=user_input, answer=answer, retrieved_sources=sources,
                            avg_similarity=result["avg_score"], rouge_l=rouge, bleu=bleu,
                            hallucination_flag=hallu["risk_flag"],
                            learner_name=profile.name, course=profile.course)

                    st.session_state.chat_messages.append({
                        "role": "assistant", "content": answer,
                        "citations": citations, "suggestions": suggestions,
                    })


# ━━━━━━━━━━━━━━━━━━━━ TAB 3 – PRACTICE QUIZ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_quiz:
    if not st.session_state.groq_api_key:
        st.warning("Please enter your Groq API key in the ⚙️ Setup tab.")
    else:
        st.subheader("📝 Micro-Quiz — Reinforce Your Learning")
        col_q1, col_q2 = st.columns([2, 1])
        with col_q1:
            num_q = st.slider("Number of questions", 3, 5, 4)
        with col_q2:
            difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)

        gen_btn = st.button("🎲 Generate Quiz", type="primary", disabled=not st.session_state.last_context)
        if not st.session_state.last_context:
            st.info("Chat with the tutor first — the quiz will use the last topic.")

        if gen_btn:
            with st.spinner("Generating quiz…"):
                questions = generate_quiz(context=st.session_state.last_context,
                    groq_api_key=st.session_state.groq_api_key, difficulty=difficulty,
                    num_questions=num_q, model=st.session_state.selected_model)
                if questions:
                    st.session_state.quiz_questions = questions
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_grades = []
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_hints = {}
                    st.session_state.quiz_hints_used = 0
                    st.success(f"Generated {len(questions)} questions!")
                else:
                    st.error("Failed to generate quiz. Try again.")

        if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
            st.markdown("---")
            for q in st.session_state.quiz_questions:
                qid = q["id"]
                st.markdown(f"**Q{qid}.** {q['question']}")
                if q.get("type") == "mcq" and q.get("options"):
                    ans = st.radio(f"Q{qid}", q["options"], key=f"qq_{qid}", label_visibility="collapsed")
                else:
                    ans = st.text_input(f"Q{qid}", key=f"qq_{qid}", label_visibility="collapsed")
                st.session_state.quiz_answers[qid] = ans

                # Per-question hint button (pre-submit)
                if st.button(f"💡 Hint for Q{qid}", key=f"hint_btn_{qid}"):
                    with st.spinner("Generating hint…"):
                        hint = generate_hint(
                            q["question"], st.session_state.last_context,
                            st.session_state.groq_api_key, model=st.session_state.selected_model)
                        st.session_state.quiz_hints[qid] = hint
                        st.session_state.quiz_hints_used += 1
                if qid in st.session_state.quiz_hints:
                    st.info(f"💡 {st.session_state.quiz_hints[qid]}")

            if st.button("✅ Submit Answers", type="primary", use_container_width=True):
                with st.spinner("Grading…"):
                    grades = grade_answers(st.session_state.quiz_questions, st.session_state.quiz_answers,
                        st.session_state.groq_api_key, model=st.session_state.selected_model)
                    st.session_state.quiz_grades = grades
                    st.session_state.quiz_submitted = True
                    if grades:
                        overall = sum(g.get("score", 0) for g in grades) / len(grades)
                        topic = st.session_state.session_memory.get_recent_topic() or "General"
                        log_quiz_result(st.session_state.learner_profile.name, topic, grades, overall)
                        st.session_state.session_memory.record_quiz(topic, overall, datetime.now(timezone.utc).isoformat())
                        # Queue wrong answers for spaced repetition
                        for g in grades:
                            if not g.get("is_correct", True):
                                st.session_state.session_memory.flag_misunderstanding(topic)
                                qid = g.get("id")
                                q_obj = next((q for q in st.session_state.quiz_questions if q["id"] == qid), None)
                                if q_obj and not any(r["question"] == q_obj["question"] for r in st.session_state.spaced_review_queue):
                                    st.session_state.spaced_review_queue.append({
                                        "question": q_obj["question"],
                                        "type": q_obj.get("type", "short_answer"),
                                        "options": q_obj.get("options"),
                                        "correct_answer": q_obj["correct_answer"],
                                        "explanation": q_obj.get("explanation", ""),
                                        "topic": topic,
                                        "attempts": 0,
                                    })
                    st.rerun()

        if st.session_state.quiz_submitted and st.session_state.quiz_grades:
            st.markdown("---")
            grades = st.session_state.quiz_grades
            total_score = sum(g.get("score", 0) for g in grades)
            pct = (total_score / len(grades) * 100) if grades else 0
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(f'<div class="metric-card"><h3>{pct:.0f}%</h3><p>Overall Score</p></div>', unsafe_allow_html=True)
            with col_s2:
                correct = sum(1 for g in grades if g.get("is_correct"))
                st.markdown(f'<div class="metric-card"><h3>{correct}/{len(grades)}</h3><p>Correct</p></div>', unsafe_allow_html=True)

            for g in grades:
                qid = g.get("id", "?")
                q_obj = next((q for q in st.session_state.quiz_questions if q["id"] == qid), {})
                q_text = q_obj.get("question", "")
                css = "correct-answer" if g.get("is_correct") else "wrong-answer"
                icon = "✅" if g.get("is_correct") else "❌"
                st.markdown(
                    f'<div class="{css}"><strong>{icon} Q{qid}</strong> — Score: {g.get("score", 0):.1f}/1.0<br>'
                    f'<em>{q_text}</em><br><br>'
                    f'<strong>Feedback:</strong> {g.get("feedback", "")}<br>'
                    f'{"<strong>💡 Hint:</strong> " + g.get("hint", "") if g.get("hint") else ""}</div>',
                    unsafe_allow_html=True)

                # Solution reveal after grading
                with st.expander(f"📖 Show Solution for Q{qid}"):
                    st.markdown(f"**Correct Answer:** {q_obj.get('correct_answer', 'N/A')}")
                    if q_obj.get("explanation"):
                        st.markdown(f"**Explanation:** {q_obj['explanation']}")
                    if not g.get("is_correct"):
                        st.markdown(f"**Your Answer:** {st.session_state.quiz_answers.get(qid, 'N/A')}")

            if st.button("🔄 Try Another Quiz", use_container_width=True):
                st.session_state.quiz_questions = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_grades = []
                st.session_state.quiz_submitted = False
                st.session_state.quiz_hints = {}
                st.rerun()

        # ── Spaced Repetition Review ─────────────────────────────────────
        if st.session_state.spaced_review_queue:
            st.markdown("---")
            st.markdown(f"### 🔁 Review Queue — {len(st.session_state.spaced_review_queue)} weak questions")
            st.caption("These are questions you got wrong. Practice them to strengthen your understanding.")

            for idx, item in enumerate(st.session_state.spaced_review_queue[:5]):
                with st.expander(f"📋 Q{idx+1}: {item['question'][:80]}…" if len(item['question']) > 80 else f"📋 Q{idx+1}: {item['question']}"):
                    st.markdown(f"**Topic:** {item.get('topic', 'General')}")
                    st.markdown(f"**Question:** {item['question']}")
                    if item.get("type") == "mcq" and item.get("options"):
                        review_ans = st.radio("Your answer:", item["options"], key=f"review_{idx}")
                    else:
                        review_ans = st.text_input("Your answer:", key=f"review_{idx}")

                    if st.button("✅ Check", key=f"review_check_{idx}"):
                        if review_ans and review_ans.strip().lower() == item["correct_answer"].strip().lower():
                            st.success("🎉 Correct! Removed from review queue.")
                            st.session_state.spaced_review_queue.pop(idx)
                            st.rerun()
                        else:
                            item["attempts"] += 1
                            st.error(f"❌ Not quite. Hint: {item.get('explanation', 'Review the topic again.')}")



# ━━━━━━━━━━━━━━━━━━━━ TAB 4 – FLASHCARDS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_flash:
    if not st.session_state.groq_api_key:
        st.warning("Please enter your Groq API key in the ⚙️ Setup tab.")
    else:
        st.subheader("🃏 Study Flashcards")
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            num_cards = st.slider("Number of cards", 5, 15, 10)
        with col_f2:
            gen_flash = st.button("🎲 Generate Flashcards", type="primary",
                disabled=not st.session_state.last_context)

        if not st.session_state.last_context:
            st.info("Chat with the tutor first to generate flashcards from your topic.")

        if gen_flash:
            with st.spinner("Generating flashcards…"):
                cards = generate_flashcards(st.session_state.last_context,
                    st.session_state.groq_api_key, num_cards, st.session_state.selected_model)
                if cards:
                    st.session_state.flashcards = cards
                    st.session_state.flashcard_index = 0
                    st.session_state.flashcard_flipped = False
                    st.success(f"Generated {len(cards)} flashcards!")
                else:
                    st.error("Failed to generate flashcards.")

        if st.session_state.flashcards:
            cards = st.session_state.flashcards
            idx = st.session_state.flashcard_index
            card = cards[idx]

            st.markdown(f"**Card {idx + 1} of {len(cards)}** — Category: *{card.get('category', 'General')}*")

            if not st.session_state.flashcard_flipped:
                st.markdown(
                    f'<div class="flashcard"><h2>{card.get("term", "Term")}</h2></div>',
                    unsafe_allow_html=True)
                if st.button("🔄 Flip Card", use_container_width=True):
                    st.session_state.flashcard_flipped = True
                    st.rerun()
            else:
                st.markdown(
                    f'<div class="flashcard-back">'
                    f'<h3>{card.get("term", "")}</h3>'
                    f'<p><strong>Definition:</strong> {card.get("definition", "")}</p>'
                    f'<p><strong>Example:</strong> {card.get("example", "")}</p>'
                    f'</div>', unsafe_allow_html=True)
                if st.button("🔄 Flip Back", use_container_width=True):
                    st.session_state.flashcard_flipped = False
                    st.rerun()

            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("⬅️ Previous", disabled=idx == 0, use_container_width=True):
                    st.session_state.flashcard_index = max(0, idx - 1)
                    st.session_state.flashcard_flipped = False
                    st.rerun()
            with col_next:
                if st.button("➡️ Next", disabled=idx >= len(cards) - 1, use_container_width=True):
                    st.session_state.flashcard_index = min(len(cards) - 1, idx + 1)
                    st.session_state.flashcard_flipped = False
                    st.rerun()

            # Export
            st.markdown("---")
            pdf_data = flashcards_to_pdf_bytes(cards)
            st.download_button("📥 Export Flashcards", data=pdf_data,
                file_name="nexgen_flashcards.txt", mime="text/plain", use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━ TAB 5 – VISUAL LEARN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_visual:
    if not st.session_state.groq_api_key:
        st.warning("Please enter your Groq API key in the ⚙️ Setup tab.")
    elif not st.session_state.kb_built:
        st.warning("Please build a knowledge base first in the ⚙️ Setup tab.")
    else:
        st.subheader("🎨 Visual Learning Tools")
        vis_topic = st.text_input("Enter topic for visualization", placeholder="e.g., Binary Search, Photosynthesis")

        vis_col1, vis_col2, vis_col3 = st.columns(3)

        with vis_col1:
            gen_mindmap = st.button("🧠 Mind Map", type="primary", use_container_width=True, disabled=not vis_topic)
        with vis_col2:
            gen_flow = st.button("📊 Flowchart", type="primary", use_container_width=True, disabled=not vis_topic)
        with vis_col3:
            gen_time = st.button("📅 Timeline", type="primary", use_container_width=True, disabled=not vis_topic)

        vs = _get_vectorstore()

        if gen_mindmap and vis_topic:
            with st.spinner("Generating mind map…"):
                docs = retrieve_with_scores(vis_topic, vs, top_k=5)
                if docs:
                    ctx = format_context(docs)
                    mm = generate_mindmap(ctx, st.session_state.groq_api_key, st.session_state.selected_model)
                    if mm:
                        st.session_state.mindmap_data = mm
                    else:
                        st.error("Failed to generate mind map.")
                else:
                    st.warning("No relevant content found for this topic.")

        if gen_flow and vis_topic:
            with st.spinner("Generating flowchart…"):
                steps = generate_flowchart(vis_topic, vs, st.session_state.groq_api_key, st.session_state.selected_model)
                if steps:
                    st.session_state.flowchart_steps = steps
                else:
                    st.error("Failed to generate flowchart.")

        if gen_time and vis_topic:
            with st.spinner("Generating timeline…"):
                events = generate_timeline(vis_topic, vs, st.session_state.groq_api_key, st.session_state.selected_model)
                if events:
                    st.session_state.timeline_events = events
                else:
                    st.error("Failed to generate timeline.")

        # Render results
        if st.session_state.mindmap_data:
            st.markdown("### 🧠 Mind Map")
            mermaid_code = mindmap_to_mermaid(st.session_state.mindmap_data)
            st.code(mermaid_code, language="mermaid")
            with st.expander("📝 Markdown View"):
                st.markdown(mindmap_to_markdown(st.session_state.mindmap_data))

        if st.session_state.flowchart_steps:
            st.markdown("### 📊 Process Flowchart")
            render_flowchart(st.session_state.flowchart_steps)

        if st.session_state.timeline_events:
            st.markdown("### 📅 Timeline")
            render_timeline(st.session_state.timeline_events)


# ━━━━━━━━━━━━━━━━━━━━ TAB 6 – SUMMARIZE URL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_url:
    if not st.session_state.groq_api_key:
        st.warning("Please enter your Groq API key in the ⚙️ Setup tab.")
    else:
        st.subheader("📺 URL Extraction & Summarization")
        url_input = st.text_input("Enter YouTube or blog URL", placeholder="https://youtube.com/watch?v=... or https://example.com/article")
        summary_style = st.selectbox("Summary Style", list(SUMMARY_STYLES.keys()))

        if st.button("🔍 Fetch & Summarize", type="primary", disabled=not url_input):
            with st.spinner("Fetching content…"):
                result = fetch_url_content(url_input)

            if result["error"]:
                st.error(f"❌ {result['error']}")
            elif result["content"]:
                st.success(f"✅ Fetched {result['type']} content ({len(result['content'])} chars)")

                with st.expander("📄 Raw Content (first 3000 chars)"):
                    st.text(result["content"][:3000])

                with st.spinner(f"Generating {summary_style} summary…"):
                    summary = summarize_content(
                        result["content"], st.session_state.groq_api_key,
                        style=summary_style, content_type=result["type"],
                        model=st.session_state.selected_model,
                    )

                st.markdown("### 📝 Summary")
                st.markdown(summary)

                st.download_button("📥 Download Summary", data=summary,
                    file_name="nexgen_summary.md", mime="text/markdown", use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━ TAB 7 – PDF SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_pdfsummary:
    if not st.session_state.groq_api_key:
        st.warning("Please enter your Groq API key in the ⚙️ Setup tab.")
    else:
        st.subheader("📄 PDF Summary — Get a Complete Summary of Any PDF")

        pdf_file = st.file_uploader("Upload a PDF to summarize", type=["pdf"], key="pdf_summary_upload")

        summary_style = st.selectbox("Summary Style", [
            "Concise", "Detailed", "Key Ideas", "Child-friendly", "Technical", "Timeline",
        ], index=1, key="pdf_summary_style")

        if pdf_file:
            if st.button("📄 Summarize PDF", type="primary", use_container_width=True):
                with st.spinner("Extracting text from PDF…"):
                    # Extract text using existing loader
                    from langchain_community.document_loaders import PyPDFLoader
                    from pathlib import Path
                    import hashlib

                    raw = pdf_file.read()
                    tmp_path = Path("/tmp") / f"nexgen_summary_{hashlib.md5(raw).hexdigest()}.pdf"
                    tmp_path.write_bytes(raw)
                    pdf_file.seek(0)

                    loader = PyPDFLoader(str(tmp_path))
                    pages = loader.load()
                    full_text = "\n\n".join([p.page_content for p in pages])
                    tmp_path.unlink(missing_ok=True)

                if not full_text.strip():
                    st.error("Could not extract text from this PDF. It may be scanned/image-based.")
                else:
                    st.success(f"✅ Extracted {len(pages)} pages ({len(full_text):,} characters)")

                    with st.expander("📄 Raw Text (first 3000 chars)"):
                        st.text(full_text[:3000])

                    with st.spinner(f"Generating {summary_style} summary…"):
                        summary = summarize_content(
                            full_text, st.session_state.groq_api_key,
                            style=summary_style, content_type="PDF document",
                            model=st.session_state.selected_model,
                        )

                    st.markdown("### 📝 Summary")
                    st.markdown(summary)

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button("📥 Download Summary", data=summary,
                            file_name=f"{pdf_file.name.replace('.pdf', '')}_summary.md",
                            mime="text/markdown", use_container_width=True)
                    with col_dl2:
                        st.download_button("📥 Download Full Text", data=full_text,
                            file_name=f"{pdf_file.name.replace('.pdf', '')}_fulltext.txt",
                            mime="text/plain", use_container_width=True)
        else:
            st.info("📁 Upload a PDF file to get started.")


# ━━━━━━━━━━━━━━━━━━━━ TAB 7 – PODCAST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_podcast:
    if not st.session_state.groq_api_key:
        st.warning("Please enter your Groq API key in the ⚙️ Setup tab.")
    else:
        st.subheader("🎙️ Podcast Studio — Turn Your Sessions into Episodes")

        if not st.session_state.chat_messages:
            st.info("💬 Chat with the tutor first — your conversations become podcast episodes!")
        else:
            st.success(f"📚 {len(st.session_state.chat_messages)} messages available from your chat session.")

            col_p1, col_p2 = st.columns([2, 1])
            with col_p1:
                topic_focus = st.text_input("🎯 Topic Focus (optional)",
                    placeholder="e.g., Focus on the algorithms discussion",
                    help="Leave empty to cover the entire conversation.")
            with col_p2:
                episode_length = st.selectbox("⏱️ Episode Length",
                    ["Short (3 min)", "Medium (8 min)", "Full (15 min)"], index=1)

            if st.button("🎙️ Generate Podcast Episode", type="primary", use_container_width=True):
                with st.spinner("Writing your podcast script… 🎧"):
                    script = generate_podcast_script(
                        st.session_state.chat_messages,
                        st.session_state.groq_api_key,
                        topic_focus=topic_focus,
                        episode_length=episode_length,
                        model=st.session_state.selected_model,
                    )
                    if script:
                        st.session_state.podcast_script = script
                        # Generate show notes
                        notes = generate_episode_notes(
                            script, st.session_state.groq_api_key,
                            model=st.session_state.selected_model,
                        )
                        st.session_state.podcast_notes = notes
                        st.success("🎉 Podcast episode generated!")
                    else:
                        st.error("Failed to generate podcast. Try again.")

            if st.session_state.podcast_script:
                st.markdown("### 🎧 Episode Script")
                st.markdown(st.session_state.podcast_script)

                # 🔊 Audio Playback
                st.markdown("---")
                st.markdown("### 🔊 Listen to Episode")
                if st.button("🎧 Generate Audio", use_container_width=True):
                    with st.spinner("Converting script to audio… this may take a moment 🎵"):
                        try:
                            import re as _re
                            from gtts import gTTS
                            import io

                            # Clean script: remove markdown formatting, speaker labels
                            clean_text = st.session_state.podcast_script
                            clean_text = _re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)  # bold
                            clean_text = _re.sub(r'#{1,4}\s*', '', clean_text)  # headers
                            clean_text = _re.sub(r'---+', '', clean_text)  # horizontal rules
                            clean_text = _re.sub(r'\[.*?\]', '', clean_text)  # brackets
                            clean_text = clean_text.strip()

                            # Generate audio
                            tts = gTTS(text=clean_text, lang='en', slow=False)
                            audio_buffer = io.BytesIO()
                            tts.write_to_fp(audio_buffer)
                            audio_buffer.seek(0)
                            st.session_state["podcast_audio"] = audio_buffer.getvalue()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Audio generation failed: {e}")

                if st.session_state.get("podcast_audio"):
                    st.audio(st.session_state["podcast_audio"], format="audio/mp3")
                    st.download_button("📥 Download MP3", data=st.session_state["podcast_audio"],
                        file_name="nexgen_podcast_episode.mp3", mime="audio/mpeg",
                        use_container_width=True)

                with st.expander("📝 Show Notes"):
                    st.markdown(st.session_state.podcast_notes or "No notes generated.")

                st.markdown("---")
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button("📥 Download Script", data=st.session_state.podcast_script,
                        file_name="nexgen_podcast_episode.md", mime="text/markdown",
                        use_container_width=True)
                with col_dl2:
                    if st.session_state.podcast_notes:
                        st.download_button("📝 Download Show Notes", data=st.session_state.podcast_notes,
                            file_name="nexgen_podcast_notes.md", mime="text/markdown",
                            use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━ TAB 7 – CODE ARENA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_arena:
    if not st.session_state.groq_api_key:
        st.warning("Please enter your Groq API key in the ⚙️ Setup tab.")
    elif not st.session_state.kb_built:
        st.warning("Please build a knowledge base first in the ⚙️ Setup tab.")
    else:
        st.subheader("🏟️ Code Arena — Learn & Compete")

        arena_tab1, arena_tab2, arena_tab3 = st.tabs(["🎯 Practice", "⏱️ Contest", "🏆 Leaderboard"])

        vs = _get_vectorstore()

        # ── Practice Mode ────────────────────────────────────────────────
        with arena_tab1:
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                topic = st.selectbox("Topic", get_available_topics())
            with col_a2:
                diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
            with col_a3:
                lang = st.selectbox("Language", list(SUPPORTED_LANGUAGES.keys()))

            if st.button("🎲 Generate Problem", type="primary"):
                with st.spinner("Generating problem…"):
                    prob = generate_problem(topic, diff, vs,
                        st.session_state.groq_api_key, lang, st.session_state.selected_model)
                    if prob:
                        st.session_state.arena_problem = prob
                        st.session_state.arena_code = prob.get("starter_code", "")
                        st.session_state.arena_results = None
                        st.success(f"Generated: {prob['title']}")
                    else:
                        st.error("Failed to generate problem. Try again.")

            if st.session_state.arena_problem:
                prob = st.session_state.arena_problem
                st.markdown(f"### {prob['title']}")
                st.markdown(f"**Difficulty:** {prob['difficulty']} | **Language:** {prob['language']}")
                st.markdown(prob.get("description", ""))

                if prob.get("examples"):
                    st.markdown("#### Examples")
                    for ex in prob["examples"]:
                        st.code(f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}\nExplanation: {ex.get('explanation', '')}")

                if prob.get("constraints"):
                    st.markdown(f"**Constraints:** {prob['constraints']}")

                # Code editor
                st.markdown("#### ✏️ Your Solution")
                try:
                    from streamlit_ace import st_ace
                    code = st_ace(
                        value=st.session_state.arena_code,
                        language="python" if prob["language"] == "Python" else "javascript",
                        theme="monokai", min_lines=15, key="code_editor",
                    )
                    st.session_state.arena_code = code
                except ImportError:
                    code = st.text_area("Code", value=st.session_state.arena_code, height=300)
                    st.session_state.arena_code = code

                col_run, col_hint = st.columns(2)
                with col_run:
                    if st.button("▶️ Run Tests", type="primary", use_container_width=True):
                        with st.spinner("Running tests…"):
                            results = run_all_tests(
                                st.session_state.arena_code,
                                prob.get("examples", []),
                                prob.get("hidden_tests", []),
                                timeout=5, language=prob["language"],
                            )
                            st.session_state.arena_results = results
                            st.session_state.arena_attempted = True

                with col_hint:
                    hint_level = st.select_slider("Hint level", [1, 2, 3], value=1)
                    if st.button("💡 Get Hint", use_container_width=True):
                        with st.spinner("Generating hint…"):
                            h = get_hint(prob["description"], st.session_state.arena_code,
                                st.session_state.groq_api_key, hint_level, st.session_state.selected_model)
                            st.info(h)

                if st.session_state.arena_results:
                    res = st.session_state.arena_results
                    st.markdown(f"### Results: {res['total_passed']}/{res['total_tests']} passed")
                    for r in res["results"]:
                        icon = "✅" if r["passed"] else "❌"
                        vis = "👁️" if r["test_type"] == "visible" else "🔒"
                        st.markdown(f"{icon} {vis} Test {r['test_num']}: {'Passed' if r['passed'] else 'Failed'}")
                        if not r["passed"] and r["test_type"] == "visible":
                            st.caption(f"Input: {r.get('input', '')} | Expected: {r.get('expected', '')} | Got: {r.get('stdout', '')}")

                    if res["all_passed"]:
                        st.balloons()
                        st.success("🎉 All tests passed!")
                        user = st.session_state.learner_profile.name
                        delta = {"Easy": 10, "Medium": 25, "Hard": 50}.get(prob["difficulty"], 10)
                        update_rating(user, delta)

                    # Code Review button (after running tests)
                    if st.button("🔍 Code Review", use_container_width=True):
                        with st.spinner("Reviewing your code…"):
                            review = review_code(
                                prob.get("description", ""), st.session_state.arena_code,
                                res, st.session_state.groq_api_key, st.session_state.selected_model)
                            st.markdown("### 🔍 Code Review")
                            st.markdown(review)

                # Solution reveal (gated behind at least one attempt)
                if st.session_state.arena_attempted:
                    with st.expander("📖 View Reference Solution"):
                        st.warning("⚠️ Make sure you've tried solving it yourself first!")
                        if st.button("📖 Show Solution", key="show_sol"):
                            with st.spinner("Generating solution…"):
                                sol = get_solution(
                                    prob.get("description", ""), prob.get("language", "Python"),
                                    st.session_state.groq_api_key, st.session_state.selected_model)
                                st.markdown(sol)

        # ── Contest Mode ─────────────────────────────────────────────────
        with arena_tab2:
            st.markdown("#### ⏱️ Timed Contest")
            if not st.session_state.contest:
                contest_dur = st.slider("Duration (minutes)", 10, 60, 30)
                if st.button("🚀 Start Contest", type="primary"):
                    with st.spinner("Generating contest problems…"):
                        problems = []
                        for d in ["Easy", "Medium", "Hard"]:
                            p = generate_problem("Arrays", d, vs,
                                st.session_state.groq_api_key, "Python", st.session_state.selected_model)
                            if p:
                                problems.append(p)
                        if problems:
                            contest = create_contest(problems, contest_dur, st.session_state.learner_profile.name)
                            st.session_state.contest = contest
                            st.rerun()
                        else:
                            st.error("Failed to generate contest problems.")
            else:
                contest = st.session_state.contest
                remaining = get_remaining_time(contest)
                if remaining > 0:
                    st.markdown(f"### ⏰ Time Remaining: **{format_time(remaining)}**")
                    st.progress(remaining / (contest["duration_minutes"] * 60))
                    for p in contest.get("problem_details", []):
                        with st.expander(f"📋 {p['title']} ({p['difficulty']})"):
                            st.markdown(p.get("description", ""))
                    if st.button("🏁 End Contest"):
                        end_contest(contest, st.session_state.learner_profile.name)
                        st.session_state.contest = None
                        st.rerun()
                else:
                    st.warning("⏰ Contest time is up!")
                    end_contest(contest, st.session_state.learner_profile.name)
                    st.session_state.contest = None

        # ── Leaderboard ──────────────────────────────────────────────────
        with arena_tab3:
            st.markdown("#### 🏆 Leaderboard")
            lb = get_leaderboard()
            if lb:
                for i, entry in enumerate(lb[:10]):
                    medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                    badges = " ".join(entry.get("badges", []))
                    st.markdown(
                        f"{medal} **{entry['user']}** — Rating: {entry['rating']} | "
                        f"Solved: {entry.get('problems_solved', 0)} {badges}")
            else:
                st.info("No entries yet. Solve some problems to get on the leaderboard!")

            # Current user rating
            user_rating = get_rating(st.session_state.learner_profile.name)
            st.markdown(f"---\n**Your Rating:** {user_rating['rating']} | "
                        f"**Problems Solved:** {user_rating.get('problems_solved', 0)}")


# ━━━━━━━━━━━━━━━━━━━━ TAB 8 – INSIGHTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_insights:
    import pandas as pd
    import plotly.express as px

    st.subheader("📊 Insights & Evaluation Dashboard")

    @st.cache_data(ttl=30)
    def _load_interactions():
        return read_interactions()

    @st.cache_data(ttl=30)
    def _load_quiz_results():
        return read_quiz_results()

    interactions = _load_interactions()
    quiz_results = _load_quiz_results()

    if not interactions and not quiz_results:
        st.info("No data yet! Start chatting and taking quizzes to see insights here.")
    else:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f'<div class="metric-card"><h3>{len(interactions)}</h3><p>Interactions</p></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<div class="metric-card"><h3>{len(quiz_results)}</h3><p>Quiz Attempts</p></div>', unsafe_allow_html=True)
        with col_m3:
            avg_sim = sum(r.get("avg_similarity", 0) for r in interactions) / len(interactions) if interactions else 0
            st.markdown(f'<div class="metric-card"><h3>{avg_sim:.2f}</h3><p>Avg Retrieval</p></div>', unsafe_allow_html=True)
        with col_m4:
            hallu = sum(1 for r in interactions if r.get("hallucination_flag")) if interactions else 0
            st.markdown(f'<div class="metric-card"><h3>{hallu}</h3><p>Hallucination Flags</p></div>', unsafe_allow_html=True)

        if quiz_results:
            st.markdown("### 📈 Quiz Score Trend")
            df_quiz = pd.DataFrame(quiz_results)
            df_quiz["attempt"] = range(1, len(df_quiz) + 1)
            df_quiz["score_pct"] = df_quiz["overall_score"] * 100
            fig = px.line(df_quiz, x="attempt", y="score_pct", markers=True,
                labels={"attempt": "Attempt", "score_pct": "Score (%)"})
            fig.update_layout(yaxis_range=[0, 105], template="plotly_white", font=dict(family="Inter"))
            fig.update_traces(line=dict(color="#667eea", width=3), marker=dict(size=10, color="#764ba2"))
            st.plotly_chart(fig, use_container_width=True)

        if interactions:
            rouge_scores = [r["rouge_l"]["f1"] for r in interactions if r.get("rouge_l") and isinstance(r["rouge_l"], dict)]
            if rouge_scores:
                st.markdown("### 📉 RAG Quality")
                avg_rouge = sum(rouge_scores) / len(rouge_scores)
                st.metric("Avg ROUGE-L F1", f"{avg_rouge:.3f}")

        st.markdown("---")
        st.markdown("### 🎯 Weakness Analysis")
        if quiz_results:
            # Group scores by topic
            topic_scores = {}
            for qr in quiz_results:
                t = qr.get("topic", "General")
                s = qr.get("overall_score", 0)
                topic_scores.setdefault(t, []).append(s)

            topic_avg = {t: sum(scores)/len(scores)*100 for t, scores in topic_scores.items()}

            if topic_avg:
                # Color-coded topic cards
                cols = st.columns(min(len(topic_avg), 4))
                for i, (topic, avg) in enumerate(sorted(topic_avg.items(), key=lambda x: x[1])):
                    if avg >= 80:
                        color, label = "🟢", "Strong"
                    elif avg >= 50:
                        color, label = "🟡", "Needs Practice"
                    else:
                        color, label = "🔴", "Weak"
                    with cols[i % len(cols)]:
                        st.markdown(f'<div class="metric-card"><h3>{color} {avg:.0f}%</h3><p>{topic}<br><small>{label}</small></p></div>', unsafe_allow_html=True)

                # Bar chart
                df_topics = pd.DataFrame([
                    {"Topic": t, "Score (%)": s}
                    for t, s in sorted(topic_avg.items(), key=lambda x: x[1])
                ])
                fig_topics = px.bar(df_topics, x="Topic", y="Score (%)",
                    color="Score (%)", color_continuous_scale=["#ff6b6b", "#ffd93d", "#6bcb77"],
                    range_color=[0, 100])
                fig_topics.update_layout(template="plotly_white", font=dict(family="Inter"),
                    yaxis_range=[0, 105], showlegend=False)
                st.plotly_chart(fig_topics, use_container_width=True)

                # Recommended focus
                weak_topics = [t for t, s in topic_avg.items() if s < 70]
                if weak_topics:
                    st.markdown("#### 📋 Recommended Focus")
                    for t in weak_topics:
                        st.markdown(f"- 📌 **{t}** — Score: {topic_avg[t]:.0f}% — Review this topic and retake quizzes")
        else:
            st.info("Take some quizzes to see your weakness analysis here.")

        # Spaced review reminder
        if st.session_state.spaced_review_queue:
            st.markdown("---")
            st.markdown(f"### 🔁 Spaced Repetition")
            st.warning(f"You have **{len(st.session_state.spaced_review_queue)}** questions in your review queue. Head to the Quiz tab to practice them!")

        st.markdown("---")
        st.markdown("### 📥 Export Logs")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.download_button("📄 JSON", data=export_logs("json"),
                file_name="nexgen_logs.json", mime="application/json", use_container_width=True)
        with col_e2:
            st.download_button("📊 CSV", data=export_logs("csv"),
                file_name="nexgen_logs.csv", mime="text/csv", use_container_width=True)
