RAG-Based Gen-AI Chatbot - Q&A over YouTube Video Transcripts
==============================================================

Topic: Generative AI & Large Language Models (YouTube technical talks).

What this project does
-----------------------
A Retrieval-Augmented Generation (RAG) chatbot that answers questions ONLY from the
transcripts of selected YouTube videos. If the answer is not contained in the
transcripts, it replies "I don't know."

Main deliverable
-----------------
RAG_YouTube_Chatbot.ipynb - a single, executable Jupyter notebook containing the
complete pipeline:
  1. Install & import libraries
  2. Select topic & YouTube videos, extract video IDs
  3. Fetch transcripts via YouTube Transcript API
  4. Save transcripts to text files (transcripts/<video_id>.txt, with timestamps)
  5. Load & split transcripts into chunks (LangChain RecursiveCharacterTextSplitter)
  6. Generate embeddings (HuggingFace sentence-transformers) & store in FAISS
  7. Set up a similarity-search retriever
  8. Integrate an LLM (Groq / OpenAI / Gemini / free local HuggingFace fallback)
  9. Build a RAG chain with a strict "answer only from context" prompt
  10. Test end-to-end with in-context and out-of-context questions
  11. Interactive CLI chat loop (type 'exit' to quit)
  12. Gradio chat UI (launched inline in the notebook)

Setup
-----
1. pip install -r requirements.txt
2. Copy .env.example to .env and optionally add ONE API key (GROQ_API_KEY,
   OPENAI_API_KEY, or GOOGLE_API_KEY). No key is required - without one, the
   pipeline uses a free local HuggingFace model.
3. Open RAG_YouTube_Chatbot.ipynb and run all cells top to bottom, OR run one
   of the standalone scripts below (they share the same src/rag_pipeline.py).

Standalone source code & UI (in addition to the notebook)
-----------------------------------------------------------
src/rag_pipeline.py  - reusable pipeline: fetch/save transcripts, chunk, embed,
                       FAISS vector store, retriever, LLM, RAG chain, chat loop
app.py               - Gradio UI            -> python app.py
cli.py               - terminal chat loop   -> python cli.py  (type 'exit' to quit)

Files
-----
RAG_YouTube_Chatbot.ipynb  - the complete, executable notebook
src/rag_pipeline.py        - shared pipeline source code used by app.py/cli.py
app.py                     - Gradio UI
cli.py                     - terminal chat loop entry point
requirements.txt           - Python dependencies
.env.example                - template for optional API keys
transcripts/                - generated at runtime, one .txt file per video
faiss_index/                - generated at runtime, persisted vector store

Notes
-----
- If a video has no captions available (or there is no internet access), the
  notebook falls back to a bundled sample transcript about Generative AI so the
  whole pipeline remains fully executable offline.
- Swap the topic by changing VIDEO_URLS in the notebook (section 2).
- Swap FAISS for Chroma/Pinecone by changing the vector-store cell (section 6).
