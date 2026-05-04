# 🎓 NextGen Tutor AI — Adaptive Learning Chatbot

A production-ready next-gen e-learning chatbot that provides personalized tutoring using a **RAG (Retrieval-Augmented Generation)** pipeline powered by **Groq API**.

---

## ✨ Features

- **📄 PDF Knowledge Ingestion** — Upload course PDFs, auto-chunk, embed, and store in ChromaDB
- **💬 RAG Chat Tutor** — Ask questions and get cited, step-by-step answers from your course materials
- **📝 Adaptive Micro-Quizzes** — Auto-generated MCQ + short answer quizzes with rubric-based grading
- **📊 Insights Dashboard** — Quiz score trends, ROUGE/BLEU metrics, hallucination flags, CSV/JSON export
- **🧠 Personalization** — Adapts explanation style based on learner level (Beginner/Intermediate/Advanced)

---

## 🛠️ Tech Stack

| Component       | Technology                                      |
| --------------- | ----------------------------------------------- |
| **UI**          | Streamlit                                       |
| **LLM**         | Groq API (`llama3-70b-8192` / `llama3-8b-8192`) |
| **Embeddings**  | HuggingFace `all-MiniLM-L6-v2` (local, free)   |
| **Vector Store** | ChromaDB (persistent, `./chroma_store`)         |
| **Orchestration** | LangChain                                     |
| **Evaluation**  | ROUGE-L, BLEU, hallucination risk flags         |

---

## 📁 Project Structure

```
nexgen-tutor-ai/
├── app.py                  # Main Streamlit app (4 tabs)
├── rag/
│   ├── ingest.py           # PDF loading, chunking, ChromaDB storage
│   ├── retriever.py        # Similarity search with scores
│   └── chain.py            # RAG chain with Groq LLM
├── tutor/
│   ├── personalize.py      # Learner profile & adaptive instructions
│   ├── quiz.py             # Micro-quiz generation from context
│   └── grader.py           # LLM-based rubric grading
├── eval/
│   ├── metrics.py          # ROUGE-L, BLEU, hallucination detection
│   └── logger.py           # Interaction & quiz logging, export
├── utils/
│   └── config.py           # Configuration, models, prompt templates
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### Prerequisites

- Python 3.10+
- A **Groq API key** — get one free at [console.groq.com](https://console.groq.com)

### Installation

#### macOS / Linux

```bash
# Clone or navigate to the project directory
cd nexgen-tutor-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

#### Windows

```powershell
# Clone or navigate to the project directory
cd nexgen-tutor-ai

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Optional: `.env` file

You can create a `.env` file in the project root (never committed):

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama3-70b-8192
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

> **Note:** The UI key entry is the primary method. `.env` is optional.

---

## 🧪 How to Test (Sample Flow)

1. **Start the app:** `streamlit run app.py`
2. **Setup tab:**
   - Paste your Groq API key
   - Enter your name, course, and skill level
   - Upload 1–2 course PDFs (any lecture notes or textbook chapters)
   - Click **"Build / Update Knowledge Base"**
3. **Chat Tutor tab:**
   - Ask a question about the uploaded material
   - View the cited answer with source pages
   - Click follow-up suggestions to explore deeper
4. **Practice Quiz tab:**
   - Click **"Generate Quiz from Last Chat Topic"**
   - Answer the questions and submit
   - Review scores, feedback, and hints
5. **Insights tab:**
   - View quiz score trends and improvement over time
   - Check RAG quality metrics (ROUGE-L, BLEU)
   - Export logs as JSON or CSV

---

## ⚙️ Customization Guide

### Changing the LLM Model

In `utils/config.py`, modify `DEFAULT_MODEL`:
```python
DEFAULT_MODEL = "llama3-70b-8192"   # For depth
# or
DEFAULT_MODEL = "llama3-8b-8192"    # For speed
```

You can also select the model from the Setup tab UI dropdown.

### Adjusting Chunk Parameters

In the Setup tab's "Chunking Configuration" expander, or in `utils/config.py`:
```python
DEFAULT_CHUNK_SIZE = 1000      # characters per chunk
DEFAULT_CHUNK_OVERLAP = 200    # overlap between chunks
```

### Customizing Quiz Style

Edit `QUIZ_GENERATION_TEMPLATE` in `utils/config.py` to change question types, difficulty levels, or format.

### Customizing Grading Rubric

Edit `GRADING_TEMPLATE` in `utils/config.py` to adjust scoring criteria, partial credit rules, or feedback style.

---

## 📝 Notes

- **Embeddings are local** — The app uses HuggingFace's `all-MiniLM-L6-v2` model, which runs locally. No API key is needed for embedding. The model is downloaded automatically on first run (~90 MB).
- **Vector store is persistent** — ChromaDB data is stored in `./chroma_store/`. Use the "Clear Vector Store" button to reset.
- **No OpenAI or Gemini** — This project uses Groq API exclusively for all LLM operations.

---

## 📜 License

MIT License — feel free to adapt for your educational needs.
