# Project Report: NextGen Tutor AI

## 1. Abstract
The **NextGen Tutor AI** is a production-ready, adaptive e-learning application designed to provide personalized, interactive tutoring. Leveraging a Retrieval-Augmented Generation (RAG) pipeline powered by the Groq API and local HuggingFace embeddings, the system transforms static course materials (PDFs) into an engaging, multi-modal learning experience.

## 2. Problem Statement
Traditional digital learning often relies on static materials (textbooks, recorded lectures) which cannot adapt to the individual pacing and skill levels of diverse learners. Students lack immediate, personalized feedback and interactive study aids, leading to lower engagement and knowledge retention.

## 3. Proposed Solution & Features
NextGen Tutor AI solves this by acting as a personalized AI teaching assistant that comprehensively understands uploaded course materials.

**Key Features:**
- **Dynamic PDF Knowledge Ingestion:** Automatic chunking and embedding of course materials into a persistent ChromaDB vector store.
- **RAG-Powered Chat Tutor:** Highly accurate, context-aware Q&A with direct source citations (document name and page number) to prevent hallucinations.
- **Adaptive Micro-Quizzes:** Automated generation of multiple-choice and short-answer questions tailored to the user's skill level. Includes an LLM-based intelligent grading rubric with partial credit and constructive hints.
- **Multi-Modal Study Tools:** 
  - *Flashcards:* Auto-generated key-term definitions.
  - *Mind-maps:* Hierarchical summaries of complex topics.
  - *Podcast Generation:* Chat transcripts converted into engaging conversational audio scripts.
  - *Debate Mode:* An AI "Devil's Advocate" to challenge student reasoning.
- **Insights & Analytics:** Continuous tracking of system accuracy using ROUGE-L and BLEU metrics, alongside learner performance dashboards.

## 4. System Architecture & Tech Stack

### 4.1 Tech Stack
- **Frontend / UI:** Streamlit
- **LLM Engine:** Groq API (`llama-3.3-70b-versatile` for deep reasoning, `llama-3.1-8b-instant` for fast tasks).
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Running entirely locally).
- **Vector Database:** ChromaDB (Persistent storage).
- **Core Orchestration:** LangChain.
- **Infrastructure:** Docker containerized for universal deployment.

### 4.2 The RAG Pipeline
1. **Ingestion:** Documents are loaded, split into overlapping chunks (e.g., 1000 characters with 200 overlap), and embedded locally using HuggingFace.
2. **Retrieval:** Student queries are vectorized. The top-$K$ most semantically similar text chunks are retrieved from ChromaDB.
3. **Generation:** The retrieved context is formatted alongside the student's profile (skill level, goals) and passed into the Groq LLM using highly structured prompt templates to ensure accuracy and limit hallucinations.

## 5. Performance & Quality Assurance
The application includes a built-in evaluation module to ensure the AI remains factual:
- **ROUGE-L & BLEU Scores:** Automatically calculated against reference texts to ensure semantic overlap and factual density.
- **Hallucination Detection:** System prompts rigorously instruct the LLM to decline answering if the context is insufficient, lowering the risk of fabricated information.

## 6. Future Scope
- Transitioning to a production cloud database (e.g., Pinecone or remote Chroma).
- Integration with standard Learning Management Systems (LMS) like Canvas or Blackboard.
- Real-time voice interaction using WebRTC for the Podcast and Chat modules.

## 7. Conclusion
NextGen Tutor AI successfully demonstrates how modern open-weight LLMs, combined with ultra-fast inference APIs (Groq) and classical RAG techniques, can democratize personalized education. It provides a robust, scalable foundation that enhances both teaching efficiency and student engagement.
