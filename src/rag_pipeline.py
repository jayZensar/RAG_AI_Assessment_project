"""Core RAG pipeline: fetch YouTube transcripts, chunk, embed, retrieve, and answer.

This module contains the same logic as RAG_YouTube_Chatbot.ipynb, refactored into
importable functions/classes so it can be reused by both the Gradio UI (app.py)
and the CLI (cli.py).
"""
import os
import re

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

try:
    from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
except ImportError:  # pragma: no cover - depends on installed library version
    TranscriptsDisabled = NoTranscriptFound = VideoUnavailable = Exception

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# --------------------------------------------------------------------------------------
# Default configuration - swap the topic by changing VIDEO_URLS.
# --------------------------------------------------------------------------------------
TOPIC = "Generative AI & Large Language Models"

VIDEO_URLS = [
    "https://www.youtube.com/watch?v=zjkBMFhNj_g",  # Intro to Large Language Models - Andrej Karpathy
    "https://www.youtube.com/watch?v=bZQun8Y4L2A",  # State of GPT - Andrej Karpathy
    "https://youtu.be/kCc8FmEb1nY",                 # Let's build GPT, from scratch - Andrej Karpathy
]

TRANSCRIPTS_DIR = "transcripts"
INDEX_DIR = "faiss_index"
BLOCK_CHAR_SIZE = 800   # caption lines are grouped into ~800-char timestamped blocks
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
RETRIEVER_K = 4

LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$")

# Bundled fallback so the pipeline is fully executable even without internet/caption access.
SAMPLE_TRANSCRIPT = [
    {"start": 0, "text": "Welcome to this introduction to generative AI and large language models."},
    {"start": 9, "text": "A large language model, or LLM, is trained on huge amounts of text to predict the next token in a sequence."},
    {"start": 20, "text": "The transformer architecture, introduced in the paper Attention Is All You Need, underlies most modern LLMs."},
    {"start": 32, "text": "Transformers use a self-attention mechanism that lets the model weigh the importance of different words in context."},
    {"start": 45, "text": "Training happens in stages: pretraining on internet-scale text, followed by supervised fine-tuning."},
    {"start": 58, "text": "Reinforcement learning from human feedback, or RLHF, is often used to align model behavior with human preferences."},
    {"start": 70, "text": "Embeddings are vector representations of text that capture semantic meaning, placing similar concepts close together."},
    {"start": 83, "text": "Retrieval-Augmented Generation, or RAG, combines a retriever with a language model so answers are grounded in external documents."},
    {"start": 96, "text": "In a RAG pipeline, documents are split into chunks, embedded, and stored in a vector database such as FAISS or Chroma."},
    {"start": 110, "text": "When a user asks a question, the system retrieves the most relevant chunks and passes them to the LLM as context."},
    {"start": 123, "text": "A well designed prompt instructs the model to answer only from the provided context and avoid hallucinating facts."},
    {"start": 136, "text": "If the retrieved context does not contain the answer, a well behaved RAG chatbot should say it does not know."},
    {"start": 148, "text": "Fine-tuning adapts a pretrained model to a specific task or domain using a smaller, labeled dataset."},
    {"start": 160, "text": "Hallucination refers to a model generating plausible sounding but factually incorrect or unsupported statements."},
    {"start": 172, "text": "Prompt engineering is the practice of crafting inputs that steer a model toward the desired kind of response."},
    {"start": 184, "text": "Vector databases index embeddings so that semantically similar chunks can be retrieved quickly using similarity search."},
    {"start": 196, "text": "Popular free and open embedding models include the sentence-transformers family, such as all-MiniLM-L6-v2."},
    {"start": 208, "text": "Thanks for watching, that concludes this overview of generative AI, transformers, and retrieval augmented generation."},
]

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a Q&A assistant that answers questions ONLY using the transcript context below.

Rules:
- Use ONLY the information contained in the context to answer the question.
- If the answer is not present in the context, reply with exactly: I don't know.
- Do not use any outside knowledge or make assumptions beyond the context.
- Keep the answer concise and factual.

Context:
{context}

Question: {question}

Answer:"""
)


# --------------------------------------------------------------------------------------
# Step 2-4: video IDs + transcript fetch/save
# --------------------------------------------------------------------------------------
def extract_video_id(url_or_id: str) -> str:
    """Extract the 11-character YouTube video ID from a URL, or pass through a bare ID."""
    url_or_id = url_or_id.strip()
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id):
        return url_or_id
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([0-9A-Za-z_-]{11})", url_or_id)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract a video ID from: {url_or_id}")


def format_timestamp(seconds) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fetch_transcript(video_id: str):
    """Return a list of {'text','start','duration'} dicts, across youtube-transcript-api versions."""
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except AttributeError:
        fetched = YouTubeTranscriptApi().fetch(video_id)
        return fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)


def fetch_and_save_one(video_url_or_id: str, transcripts_dir: str = TRANSCRIPTS_DIR):
    """Fetch a single video's transcript and save it to transcripts_dir/<video_id>.txt.

    Returns the resolved video_id on success, or None if no transcript could be fetched
    (captions disabled, video unavailable, etc.) - the caller decides how to handle that.
    """
    os.makedirs(transcripts_dir, exist_ok=True)
    video_id = extract_video_id(video_url_or_id)
    out_path = os.path.join(transcripts_dir, f"{video_id}.txt")
    try:
        segments = fetch_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, Exception) as exc:
        print(f"Could not fetch transcript for {video_id} ({exc}).")
        return None
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{format_timestamp(seg['start'])}] {seg['text']}\n")
    print(f"Saved transcript -> {out_path} ({len(segments)} segments)")
    return video_id


def fetch_and_save_transcripts(video_urls, transcripts_dir: str = TRANSCRIPTS_DIR) -> list:
    """Fetch each video's transcript and save it to transcripts_dir/<video_id>.txt.

    Falls back to a bundled sample transcript if no video could be fetched (e.g. no
    internet access or captions disabled), so the pipeline always has data to index.
    Returns the list of video IDs that now have a saved transcript file.
    """
    saved_ids = [vid for u in video_urls if (vid := fetch_and_save_one(u, transcripts_dir))]

    if not saved_ids:
        print("No live transcripts could be fetched. Using bundled sample transcript instead.")
        video_id = "sample_genai_intro"
        os.makedirs(transcripts_dir, exist_ok=True)
        out_path = os.path.join(transcripts_dir, f"{video_id}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for seg in SAMPLE_TRANSCRIPT:
                f.write(f"[{format_timestamp(seg['start'])}] {seg['text']}\n")
        saved_ids = [video_id]
        print(f"Saved fallback transcript -> {out_path}")

    return saved_ids


# --------------------------------------------------------------------------------------
# Step 5: load + chunk transcripts (with timestamp metadata)
# --------------------------------------------------------------------------------------
def timestamp_to_seconds(ts: str) -> int:
    h, m, s = (int(p) for p in ts.split(":"))
    return h * 3600 + m * 60 + s


def load_transcript_lines(path: str):
    """Parse '[HH:MM:SS] text' lines back into (seconds, timestamp, text) tuples."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            match = LINE_RE.match(raw_line.strip())
            if match:
                ts, text = match.groups()
                lines.append((timestamp_to_seconds(ts), ts, text))
    return lines


def build_chunks(video_ids, transcripts_dir: str = TRANSCRIPTS_DIR) -> list:
    """Group captions into timestamped blocks, then split into overlapping chunks."""
    block_documents = []
    for video_id in video_ids:
        path = os.path.join(transcripts_dir, f"{video_id}.txt")
        if not os.path.exists(path):
            continue
        buffer_text, block_start_ts, char_count = [], None, 0
        for _seconds, ts, text in load_transcript_lines(path):
            if block_start_ts is None:
                block_start_ts = ts
            buffer_text.append(text)
            char_count += len(text)
            if char_count >= BLOCK_CHAR_SIZE:
                block_documents.append(Document(
                    page_content=" ".join(buffer_text),
                    metadata={"video_id": video_id, "timestamp": block_start_ts, "source": path},
                ))
                buffer_text, block_start_ts, char_count = [], None, 0
        if buffer_text:
            block_documents.append(Document(
                page_content=" ".join(buffer_text),
                metadata={"video_id": video_id, "timestamp": block_start_ts, "source": path},
            ))

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(block_documents)


# --------------------------------------------------------------------------------------
# Step 6-7: embeddings, vector store, retriever
# --------------------------------------------------------------------------------------
class TfidfEmbeddings(Embeddings):
    """Pure scikit-learn fallback embedding model - no downloads, works fully offline.

    Used only when the sentence-transformers model can't be downloaded from
    huggingface.co (e.g. a corporate proxy blocks the domain).
    """

    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    def embed_documents(self, texts):
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray().tolist()

    def embed_query(self, text):
        if not self._fitted:
            self.vectorizer.fit([text])
            self._fitted = True
        return self.vectorizer.transform([text]).toarray()[0].tolist()


def get_embedding_model():
    """Try the free HuggingFace embedding model; fall back to local TF-IDF if unreachable."""
    try:
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        model.embed_query("connectivity check")  # forces a real download/verify attempt
        return model
    except Exception as exc:
        print(f"Could not load HuggingFace embeddings ({exc}).")
        print("Falling back to local TF-IDF embeddings (fully offline, no downloads, lower quality).")
        return TfidfEmbeddings()


def build_vectorstore(chunks, embedding_model, index_dir: str = INDEX_DIR):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(index_dir)
    return vectorstore


def load_vectorstore(embedding_model, index_dir: str = INDEX_DIR):
    return FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)


# --------------------------------------------------------------------------------------
# Step 8: LLM integration (dynamic provider selection)
# --------------------------------------------------------------------------------------
def get_llm():
    """Pick an LLM based on whichever API key is available; falls back to a free local model."""
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        print("Using Groq (llama-3.1-8b-instant).")
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        print("Using OpenAI (gpt-4o-mini).")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("Using Google Gemini (gemini-3.6-flash).")
        return ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

    print("No API key found - trying the free local HuggingFace model 'google/flan-t5-base'.")
    try:
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline

        pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as exc:
        print(f"Could not load local HuggingFace model ({exc}).")
        print("Falling back to extractive mode: answers will be the most relevant transcript "
              "excerpt instead of an LLM-generated response.")
        return None


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata['video_id']} @ {d.metadata['timestamp']}]\n{d.page_content}"
        for d in docs
    )


def _content_to_text(content) -> str:
    """Chat model .content is usually a str, but some providers (e.g. Gemini) return a list
    of content blocks - normalize either shape into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# --------------------------------------------------------------------------------------
# Public API: YouTubeRagChatbot ties everything together.
# --------------------------------------------------------------------------------------
class YouTubeRagChatbot:
    """End-to-end RAG chatbot over a set of YouTube video transcripts."""

    def __init__(self, video_urls=None, topic=None,
                 transcripts_dir: str = TRANSCRIPTS_DIR, index_dir: str = INDEX_DIR):
        self.video_urls = video_urls if video_urls is not None else VIDEO_URLS
        self.topic = topic or TOPIC
        self.transcripts_dir = transcripts_dir
        self.index_dir = index_dir
        self.embedding_model = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self._answer_chain = None
        self.chunks = []
        self.loaded_video_ids = []

    def initialize_models(self) -> None:
        """Load the embedding model and LLM once, without fetching/indexing any video yet."""
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()
        if self.llm is None:
            self.llm = get_llm()
            self._answer_chain = RAG_PROMPT | self.llm | StrOutputParser() if self.llm else None

    def setup(self, force_rebuild: bool = False) -> None:
        """Fetch the configured transcripts (if needed), build/load the vector index, and load the LLM."""
        self.initialize_models()
        # TF-IDF vectorizer state isn't persisted, so always rebuild in that fallback mode.
        force_rebuild = force_rebuild or isinstance(self.embedding_model, TfidfEmbeddings)

        if not force_rebuild and os.path.isdir(self.index_dir) and os.listdir(self.index_dir):
            print(f"Loading existing FAISS index from '{self.index_dir}/'.")
            self.vectorstore = load_vectorstore(self.embedding_model, self.index_dir)
            self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVER_K})
            return

        for url in self.video_urls:
            self._ingest_video(url)
        self._rebuild_index()

    def _ingest_video(self, video_url_or_id: str):
        """Fetch+save+chunk a single video and append it to the in-memory corpus.

        Returns (video_id, num_new_chunks) on success, or None if no transcript was found.
        """
        video_id = fetch_and_save_one(video_url_or_id, self.transcripts_dir)
        if video_id is None:
            return None
        if video_id in self.loaded_video_ids:
            return (video_id, 0)
        new_chunks = build_chunks([video_id], self.transcripts_dir)
        self.chunks.extend(new_chunks)
        self.loaded_video_ids.append(video_id)
        return (video_id, len(new_chunks))

    def _rebuild_index(self) -> None:
        """(Re)build the FAISS index from the full in-memory corpus (self.chunks)."""
        print(f"Indexing {len(self.chunks)} chunk(s)...")
        self.vectorstore = build_vectorstore(self.chunks, self.embedding_model, self.index_dir)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVER_K})

    def add_video(self, video_url_or_id: str) -> str:
        """Fetch a video's transcript at runtime, index it, and return a status message for the UI."""
        self.initialize_models()
        try:
            result = self._ingest_video(video_url_or_id)
        except ValueError as exc:
            return f"Could not parse that input: {exc}"
        if result is None:
            return (
                "Could not fetch a transcript for that video. It may have captions disabled, "
                "be private/unavailable, or the URL/ID may be invalid."
            )
        video_id, num_new_chunks = result
        if num_new_chunks == 0:
            return f"Video '{video_id}' is already loaded ({len(self.chunks)} chunk(s) indexed so far)."
        self._rebuild_index()
        return (
            f"Loaded video '{video_id}': {num_new_chunks} new chunk(s) indexed. "
            f"Videos loaded: {len(self.loaded_video_ids)}, total chunks: {len(self.chunks)}."
        )

    def reset(self) -> None:
        """Drop all loaded videos/chunks/index so the user can start over with new videos."""
        self.chunks = []
        self.loaded_video_ids = []
        self.vectorstore = None
        self.retriever = None

    def summarize_video(self, video_id: str, max_preview_chars: int = 2000) -> str:
        """Short 'what's in this video' blurb so the user knows what to ask about.

        Uses the LLM if one is configured; falls back to a raw transcript excerpt otherwise
        (or if the LLM call fails).
        """
        video_chunks = [c for c in self.chunks if c.metadata.get("video_id") == video_id]
        if not video_chunks:
            return ""
        preview_text = " ".join(c.page_content for c in video_chunks)[:max_preview_chars]

        if self.llm is not None:
            try:
                prompt = (
                    "In 2-3 short sentences, summarize what topics this video transcript covers "
                    "so a user knows what they could ask questions about. Don't answer any "
                    "question, just summarize the topic.\n\nTranscript excerpt:\n" + preview_text
                )
                result = self.llm.invoke(prompt)
                content = result.content if hasattr(result, "content") else result
                return _content_to_text(content).strip()
            except Exception as exc:
                print(f"Could not summarize video ({exc}); falling back to a plain excerpt.")

        return preview_text[:280].strip() + "..."

    def ask(self, question: str):
        """Retrieve relevant chunks, generate a context-only answer, and return (answer, sources)."""
        if self.retriever is None:
            if not self.chunks and not self.video_urls:
                return "No video loaded yet. Please add a YouTube video URL first.", []
            self.setup()
        if self._answer_chain is None:
            return self._ask_extractive(question)
        docs = self.retriever.invoke(question)
        context = format_docs(docs)
        try:
            answer = self._answer_chain.invoke({"context": context, "question": question}).strip()
        except Exception as exc:
            # LLM call failed (rate limit, network error, etc.) - degrade gracefully instead of
            # letting the UI hang or crash.
            print(f"LLM call failed ({exc}); falling back to extractive mode for this question.")
            answer, sources = self._ask_extractive(question)
            return f"\u26A0\uFE0F (LLM unavailable, showing closest excerpt instead) {answer}", sources
        sources = [] if answer == "I don't know." else [
            f"{d.metadata['video_id']} @ {d.metadata['timestamp']}" for d in docs
        ]
        return answer, sources

    def _ask_extractive(self, question: str, distance_threshold: float = 1.3):
        """No-LLM fallback: return the closest transcript excerpt instead of a generated answer."""
        hits = self.vectorstore.similarity_search_with_score(question, k=RETRIEVER_K)
        if not hits or hits[0][1] > distance_threshold:
            return "I don't know.", []
        doc, _score = hits[0]
        excerpt = doc.page_content.strip()[:400]
        answer = f"(Extractive mode - no LLM available) Closest transcript excerpt: \"{excerpt}\""
        sources = [f"{d.metadata['video_id']} @ {d.metadata['timestamp']}" for d, _ in hits]
        return answer, sources

    def chat_loop(self) -> None:
        """Interactive CLI loop; keeps asking for input until the user types 'exit'."""
        print(f"YouTube RAG Chatbot ready. Topic: {self.topic}")
        print("Type your question, or 'exit' to quit.\n")
        while True:
            question = input("You: ").strip()
            if question.lower() == "exit":
                print("Bot: Goodbye!")
                break
            if not question:
                continue
            answer, sources = self.ask(question)
            print(f"Bot: {answer}")
            if sources:
                print(f"     Sources: {', '.join(sources)}")
