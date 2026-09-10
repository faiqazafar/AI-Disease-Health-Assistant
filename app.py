import base64
import json
import os
from typing import Any, Dict, List, Tuple, Optional

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from PIL import Image
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="AI Disease & Health Assistant",
    page_icon="🩺",
    layout="wide",
)

DEFAULT_TEXT_MODEL = "qwen/qwen3.6-27b"
DEFAULT_VISION_MODEL = "qwen/qwen3.6-27b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_IMAGE_BYTES = 4 * 1024 * 1024

EMERGENCY_PHRASES = [
    "difficulty breathing", "shortness of breath", "cannot breathe",
    "can't breathe", "chest pain", "severe chest pain", "severe bleeding",
    "heavy bleeding", "uncontrolled bleeding", "unconscious",
    "loss of consciousness", "seizure", "severe confusion", "blue lips",
    "fainted", "fainting", "not responding", "coughing blood",
    "vomiting blood", "black stool", "sudden weakness", "face drooping",
    "slurred speech",
]

def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return os.getenv(name, default).strip()

def language_instruction(language: str) -> str:
    if language == "Urdu":
        return """
LANGUAGE RULE:
- Answer ONLY in natural, clear Urdu script (اردو).
- Do not switch to English for the main answer.
- Medical terms may include an English term in parentheses only when useful.
- Keep headings and bullet points in Urdu too.
"""
    return """
LANGUAGE RULE:
- Answer ONLY in clear, natural English.
- Use simple wording suitable for a general reader.
"""

def emergency_check(text: str) -> List[str]:
    lower = text.lower()
    return [p for p in EMERGENCY_PHRASES if p in lower]

api_key = get_secret("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY is not configured.")
    st.info(
        'Streamlit Cloud: open your app → Settings → Secrets and add '
        'GROQ_API_KEY = "your_key_here"'
    )
    st.stop()

client = Groq(api_key=api_key)
TEXT_MODEL = get_secret("GROQ_TEXT_MODEL", DEFAULT_TEXT_MODEL)
VISION_MODEL = get_secret("GROQ_VISION_MODEL", DEFAULT_VISION_MODEL)

@st.cache_data
def load_knowledge_base() -> List[Dict[str, Any]]:
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        return json.load(f)

knowledge_base = load_knowledge_base()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

embedding_model = load_embedding_model()

@st.cache_resource
def create_faiss_index(records_json: str):
    records = json.loads(records_json)
    texts = [
        f"{x.get('condition','')}. {x.get('category','')}. {x.get('text','')}"
        for x in records
    ]
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    return idx

index = create_faiss_index(json.dumps(knowledge_base, ensure_ascii=False))

def search_knowledge(
    query: str,
    top_k: int = 5,
    min_score: float = 0.28,
) -> List[Dict[str, Any]]:
    q = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    scores, indices = index.search(q, min(top_k, len(knowledge_base)))
    results = []
    for score, i in zip(scores[0], indices[0]):
        if i < 0 or i >= len(knowledge_base):
            continue
        if float(score) < min_score:
            continue
        item = dict(knowledge_base[i])
        item["_score"] = float(score)
        results.append(item)
    return results

def build_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No sufficiently relevant medical knowledge was retrieved."
    blocks = []
    for n, item in enumerate(results, 1):
        blocks.append(
            f"""SOURCE {n}
Condition: {item.get('condition','')}
Category: {item.get('category','')}
Information: {item.get('text','')}
Source: {item.get('source','')}
URL: {item.get('url','')}
"""
        )
    return "\n-----------------------\n".join(blocks)

def unique_sources(results: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    seen = set()
    out = []
    for x in results:
        pair = (x.get("source", "Unknown source"), x.get("url", ""))
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out

def create_health_prompt(
    question: str,
    context: str,
    language: str,
    history: List[Dict[str, str]],
) -> str:
    recent = history[-6:]
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in recent
    ) or "No previous conversation."

    return f"""
You are a careful, evidence-grounded health information assistant.

Your job is to explain health information, not to diagnose a person.

CORE RULES:
- Use the retrieved medical knowledge as the primary factual source.
- Do not invent medical facts unsupported by the retrieved knowledge.
- Do not give a definitive diagnosis.
- Do not prescribe medicines.
- Do not provide medication doses or self-medication instructions.
- Do not claim symptoms alone prove a disease.
- If several conditions can look similar, say so briefly.
- If the retrieved knowledge is insufficient, clearly say so.
- If emergency warning signs are present, advise urgent professional medical evaluation.
- Be calm, respectful, and non-alarming.
- Answer the actual question first.

{language_instruction(language)}

RESPONSE STYLE:
- Start with a direct answer in 1–2 sentences.
- Use short bullets when helpful.
- For simple questions, keep it short.
- For normal questions, aim for about 150–250 words.
- Do not force the same headings into every answer.
- Avoid unnecessary repetition of disclaimers.

RECENT CONVERSATION:
{history_text}

USER QUESTION:
{question}

RETRIEVED MEDICAL KNOWLEDGE:
{context}
"""

def ask_groq(
    question: str,
    context: str,
    language: str = "English",
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    prompt = create_health_prompt(question, context, language, history or [])
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a cautious, evidence-grounded health information "
                    "assistant. Never present a diagnosis as certain."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=700,
        reasoning_effort="none",
    )
    return (response.choices[0].message.content or "").strip()

def analyze_image(
    image_bytes: bytes,
    mime_type: str,
    extra_information: str = "",
    language: str = "English",
) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    prompt = f"""
You are a cautious health image screening assistant.

Analyze ONLY visible features. This is educational screening, not diagnosis.

IMPORTANT:
- Do not identify a disease with certainty.
- Do not say an image proves a disease.
- Do not prescribe medication or dosage.
- Do not claim information that cannot be seen.
- State uncertainty clearly.
- If the image is blurry, dark, distant, obstructed, or not relevant to a
  visible health/skin concern, say that the image is insufficient.
- Recommend professional medical evaluation when appropriate.

Describe:
1. Visible observations
2. Broad possible categories that may look similar
3. Important missing information
4. Warning signs requiring prompt medical attention
5. When professional evaluation may be appropriate

{language_instruction(language)}

Additional user information:
{extra_information or "None provided."}

Keep the response concise and patient-friendly.
"""
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded}"
                        },
                    },
                ],
            }
        ],
        temperature=0.2,
        max_completion_tokens=550,
        reasoning_effort="none",
    )
    return (response.choices[0].message.content or "").strip()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

st.sidebar.title("🩺 Navigation")
mode = st.sidebar.radio(
    "Choose a feature:",
    ["Health Chat", "Image Screening", "How It Works"],
)
language = st.sidebar.selectbox("Response language:", ["English", "Urdu"])

if mode == "Health Chat" and st.sidebar.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.session_state.last_sources = []
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("Knowledge-grounded health information")
st.sidebar.caption(f"Knowledge records: {len(knowledge_base)}")

st.title("🩺 AI Disease & Health Assistant")
st.caption(
    "RAG-based health information assistant with semantic search, "
    "Groq AI, image screening, English/Urdu responses and safety checks."
)
st.warning(
    "⚠️ Educational information only. This app does not diagnose disease, "
    "prescribe treatment, or replace a qualified healthcare professional."
)

if mode == "Health Chat":
    st.header("💬 Health Information Chat")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a health question… / صحت سے متعلق سوال پوچھیں…"
    )

    if question:
        question = question.strip()
        previous_history = list(st.session_state.chat_history)
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        emergency = emergency_check(question)

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching medical knowledge…"):
                    results = search_knowledge(
                        question, top_k=5, min_score=0.28
                    )
                    context = build_context(results)

                if not results:
                    st.info(
                        "I could not find sufficiently relevant information "
                        "in the current medical knowledge base."
                    )

                with st.spinner("Preparing a response…"):
                    answer = ask_groq(
                        question, context, language, previous_history
                    )

                if emergency:
                    st.error(
                        "⚠️ Possible emergency warning signs detected. "
                        "Please seek urgent professional medical care."
                    )

                st.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.last_sources = results

            except Exception as exc:
                st.error("The AI service could not generate a response right now.")
                st.caption(f"Technical detail: {exc}")

    if st.session_state.last_sources:
        st.divider()
        with st.expander("🔍 RAG retrieval details"):
            for item in st.session_state.last_sources:
                st.markdown(
                    f"**{item['condition']} — {item['category']}** "
                    f"(similarity: {item.get('_score', 0):.2f})"
                )
                st.write(item["text"])

        st.subheader("📚 Sources")
        for name, url in unique_sources(st.session_state.last_sources):
            st.markdown(f"- **{name}** — {url}")

elif mode == "Image Screening":
    st.header("🖼️ AI Image Screening")
    st.info(
        "Upload a clear image of a visible skin/health concern. The AI "
        "describes visible features and provides general educational "
        "information. It does not diagnose disease."
    )

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    extra = st.text_area(
        "Additional information (optional)",
        placeholder=(
            "Example: The area has been itchy for three days."
        ),
    )

    if uploaded:
        data = uploaded.getvalue()
        if len(data) > MAX_IMAGE_BYTES:
            st.error("Please upload an image smaller than 4 MB.")
            st.stop()

        try:
            image = Image.open(uploaded)
            image.verify()
        except Exception:
            st.error("The uploaded file is not a valid readable image.")
            st.stop()

        uploaded.seek(0)
        st.image(uploaded, caption="Uploaded image", width="stretch")

        if st.button("Analyze Image", type="primary"):
            try:
                with st.spinner("Analyzing visible features…"):
                    findings = analyze_image(
                        data,
                        uploaded.type or "image/jpeg",
                        extra,
                        language,
                    )

                st.subheader("🔎 Visual Screening")
                st.markdown(findings)

                with st.spinner("Searching medical knowledge…"):
                    results = search_knowledge(
                        findings, top_k=5, min_score=0.28
                    )
                    context = build_context(results)

                with st.spinner("Preparing related health information…"):
                    final_answer = ask_groq(
                        findings, context, language, []
                    )

                st.subheader("🩺 Related Health Information")
                st.markdown(final_answer)

                if results:
                    with st.expander("🔍 RAG retrieval details"):
                        for item in results:
                            st.markdown(
                                f"**{item['condition']} — {item['category']}** "
                                f"(similarity: {item.get('_score', 0):.2f})"
                            )
                            st.write(item["text"])

                    st.subheader("📚 Sources")
                    for name, url in unique_sources(results):
                        st.markdown(f"- **{name}** — {url}")
                else:
                    st.info(
                        "No sufficiently relevant article was found in the "
                        "current knowledge base."
                    )
            except Exception as exc:
                st.error(
                    "The image screening service could not complete the request."
                )
                st.caption(f"Technical detail: {exc}")

else:
    st.header("🧠 How the AI Works")
    st.markdown(
        """
### 1. User input
The user asks a health question or uploads an image.

### 2. Semantic retrieval
A Sentence Transformer converts text into an embedding. FAISS searches the
medical knowledge base for semantically similar information.

### 3. Relevance filtering
A similarity threshold removes weak or unrelated retrieval results.

### 4. Groq AI
The retrieved information is supplied to the language model with instructions
to avoid diagnosis, prescriptions and unsupported claims.

### 5. Safety layer
The app checks the user's message for common emergency warning phrases.

### 6. Image screening
The vision model describes visible features first. Those findings are then
used as the query for the RAG system before the final response.

### Architecture

`User → Sentence Transformer → FAISS → Medical Knowledge → Groq → Response`

Image flow:

`Image → Groq Vision → Visible Findings → FAISS → Medical Knowledge → Groq`
"""
    )
    st.success(
        "The system is designed as a health-information assistant, "
        "not an automated diagnostic system."
    )

st.divider()
st.caption(
    "AI Disease & Health Assistant | Educational use only | "
    "Not a substitute for professional medical care"
)
