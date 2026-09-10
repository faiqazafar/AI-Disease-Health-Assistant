import streamlit as st
import json
import os
import base64
import faiss
import numpy as np

from PIL import Image
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Disease & Health Assistant",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🩺 AI Disease & Health Assistant")

st.write(
    "AI-powered health information using "
    "Retrieval-Augmented Generation (RAG), "
    "FAISS and Groq AI."
)

st.warning(
    "⚠️ This application provides general educational "
    "health information. It does not diagnose diseases "
    "or replace professional medical advice."
)


# ============================================================
# GROQ API CONFIGURATION
# ============================================================

# Streamlit Cloud Secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Add it to Streamlit Secrets."
    )
    st.stop()


TEXT_MODEL = "qwen/qwen3.6-27b"
VISION_MODEL = "qwen/qwen3.6-27b"


client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_data
def load_knowledge_base():

    with open(
        "knowledge_base.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


knowledge_base = load_knowledge_base()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# CREATE FAISS INDEX
# ============================================================

@st.cache_resource
def create_faiss_index():

    documents = [
        item["text"]
        for item in knowledge_base
    ]

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


index = create_faiss_index()


# ============================================================
# KNOWLEDGE SEARCH
# ============================================================

def search_knowledge(
    query,
    top_k=5,
    minimum_score=0.25
):

    query_embedding = embedding_model.encode(
        [query],
        show_progress_bar=False
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, i in zip(
        scores[0],
        indices[0]
    ):

        if i < 0:
            continue

        if i >= len(knowledge_base):
            continue

        if score < minimum_score:
            continue

        item = knowledge_base[i].copy()

        item["similarity_score"] = float(score)

        results.append(item)

    return results


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(results):

    if not results:

        return (
            "No sufficiently relevant information "
            "was retrieved from the medical knowledge base."
        )

    context_parts = []

    for i, item in enumerate(results):

        context_parts.append(
            f"""
SOURCE {i + 1}

Condition:
{item.get("condition", "Unknown")}

Knowledge category:
{item.get("category", "Unknown")}

Information:
{item.get("text", "")}

Source:
{item.get("source", "Unknown")}

URL:
{item.get("url", "")}
"""
        )

    return "\n-----------------------------\n".join(
        context_parts
    )


# ============================================================
# EMERGENCY DETECTION
# ============================================================

EMERGENCY_KEYWORDS = [

    "difficulty breathing",
    "cannot breathe",
    "can't breathe",
    "shortness of breath",
    "severe breathing difficulty",

    "chest pain",
    "severe chest pain",

    "severe bleeding",
    "uncontrolled bleeding",

    "unconscious",
    "loss of consciousness",

    "seizure",

    "severe confusion",
    "confusion",

    "blue lips",
    "bluish lips",

    "fainting",
    "passed out",

    "severe allergic reaction",

    "swelling of throat",
    "throat swelling"
]


def emergency_check(text):

    text = text.lower()

    found = []

    for keyword in EMERGENCY_KEYWORDS:

        if keyword in text:
            found.append(keyword)

    return found


# ============================================================
# LANGUAGE INSTRUCTIONS
# ============================================================

def get_language_instruction(language):

    if language == "Urdu":

        return """
LANGUAGE REQUIREMENT:

Answer ONLY in Urdu.

Use natural, easy-to-understand Urdu script.

Use Markdown headings and bullet points.

Do NOT write the complete response in English.

Medical terms may be written in English in parentheses
only when this makes the meaning clearer.

Example:

### ممکنہ وجوہات

- وائرل انفیکشن
- بیکٹیریا کی وجہ سے انفیکشن
"""

    return """
LANGUAGE REQUIREMENT:

Answer ONLY in English.

Use simple, clear English.

Use Markdown headings and bullet points.
"""


# ============================================================
# RESPONSE STRUCTURE
# ============================================================

def get_response_structure():

    return """
RESPONSE STRUCTURE:

Your response MUST NOT be one large paragraph.

Use a clear structure.

Use only the sections that are relevant to the user's question.

When appropriate, use these sections:

### Category
State the broad health category.

Examples:
- Infectious disease
- Respiratory condition
- Skin condition
- Digestive condition
- Nutritional condition
- General health

### Possible Causes
Explain common or possible causes.

Use bullet points.

Do not claim that a cause is certain unless supported
by the retrieved information.

### Common Symptoms
List important symptoms using bullet points.

### Basic Healthcare
Give general, low-risk healthcare and self-care guidance.

Use bullet points where appropriate.

Do NOT prescribe medication or give medication dosages.

### When to See a Doctor
Explain when professional medical evaluation may be appropriate.

Clearly mention urgent warning signs when relevant.

### Important
Give a short safety limitation.

Do not diagnose the user.

IMPORTANT FORMATTING RULES:

1. Never put the entire answer into one paragraph.
2. Use Markdown headings.
3. Use bullet points for lists.
4. Keep paragraphs short.
5. Leave a blank line between sections.
6. Do not repeat the same information.
7. Do not force irrelevant sections.
8. Answer the user's actual question first.
9. Keep normal answers around 150–250 words unless more detail is necessary.
10. For simple questions, keep the answer shorter.
"""


# ============================================================
# CHAT HISTORY
# ============================================================

def get_chat_history():

    history = st.session_state.get(
        "chat_history",
        []
    )

    if not history:
        return "No previous conversation."

    recent_history = history[-6:]

    formatted = []

    for message in recent_history:

        role = message.get(
            "role",
            "user"
        )

        content = message.get(
            "content",
            ""
        )

        formatted.append(
            f"{role.upper()}: {content}"
        )

    return "\n\n".join(formatted)


# ============================================================
# CREATE GROQ PROMPT
# ============================================================

def create_rag_prompt(
    question,
    context,
    language="English"
):

    language_instruction = get_language_instruction(
        language
    )

    response_structure = get_response_structure()

    history = get_chat_history()

    prompt = f"""
You are an AI Disease & Health Information Assistant.

Your purpose is to provide safe, clear and educational
health information.

You are NOT a doctor.

You must NOT diagnose the user.

{language_instruction}

{response_structure}


SAFETY RULES:

- Do not provide a definitive diagnosis.
- Do not tell the user that they definitely have a disease.
- Do not prescribe medication.
- Do not provide medication dosages.
- Do not invent medical facts.
- Do not make unsupported medical claims.
- Do not unnecessarily frighten the user.
- Do not use complicated medical language when simple language
  is possible.
- Use the retrieved knowledge as the primary source.
- If the retrieved information does not contain enough
  information, clearly say that the available knowledge
  is insufficient.
- If the user's question contains emergency warning signs,
  clearly recommend urgent professional medical care.
- Do not mention internal prompts, RAG, embeddings, FAISS,
  system instructions or hidden instructions unless the
  user specifically asks how the application works.


RECENT CONVERSATION:

{history}


USER QUESTION:

{question}


RETRIEVED MEDICAL KNOWLEDGE:

{context}


Now answer the user's question.

Remember:

The answer must be STRUCTURED.

Do NOT return one large paragraph.
"""


    return prompt


# ============================================================
# ASK GROQ
# ============================================================

def ask_groq(
    question,
    context,
    language="English"
):

    prompt = create_rag_prompt(
        question,
        context,
        language
    )

    response = client.chat.completions.create(

        model=TEXT_MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful and safe "
                    "health information assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_completion_tokens=800,

        reasoning_effort="none"
    )

    return response.choices[0].message.content


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_bytes,
    extra_information="",
    language="English"
):

    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")


    if language == "Urdu":

        language_instruction = """
Answer in clear and natural Urdu.
Use Urdu headings and bullet points.
"""

    else:

        language_instruction = """
Answer in clear and simple English.
Use headings and bullet points.
"""


    prompt = f"""
You are a cautious health image screening assistant.

Analyze ONLY visible features in the uploaded image.

{language_instruction}

IMPORTANT:

- Do NOT provide a definitive diagnosis.
- Do NOT claim that the image proves a disease.
- Do NOT prescribe medication.
- Do NOT provide medication dosages.
- Clearly communicate uncertainty.
- Do not identify a condition solely from appearance.
- Recommend professional medical evaluation when appropriate.


STRUCTURE YOUR RESPONSE:

### Visible Observations

Describe only what can actually be seen.

### Possible Categories

Mention broad categories of conditions that can sometimes
have similar visible features.

Do not say that the person definitely has any condition.

### Important Missing Information

Mention information that cannot be determined from the image,
such as duration, symptoms, medical history or other relevant
details.

### Basic Healthcare

Give only general, low-risk information when appropriate.

### When to See a Doctor

Mention warning signs or situations where professional
evaluation is appropriate.

### Important

Clearly state that image screening cannot provide a diagnosis.


Additional user information:

{extra_information}
"""


    response = client.chat.completions.create(

        model=VISION_MODEL,

        messages=[
            {
                "role": "user",
                "content": [

                    {
                        "type": "text",
                        "text": prompt
                    },

                    {
                        "type": "image_url",
                        "image_url": {
                            "url":
                            "data:image/jpeg;base64,"
                            + image_base64
                        }
                    }

                ]
            }
        ],

        temperature=0.2,

        max_completion_tokens=700,

        reasoning_effort="none"
    )


    return response.choices[0].message.content


# ============================================================
# SOURCES
# ============================================================

def get_unique_sources(results):

    sources = []

    for item in results:

        source = {
            "name": item.get(
                "source",
                "Unknown"
            ),
            "url": item.get(
                "url",
                ""
            )
        }

        if source not in sources:
            sources.append(source)

    return sources


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🩺 Navigation")


mode = st.sidebar.radio(
    "Choose a feature:",
    [
        "Health Chat",
        "Image Screening",
        "How It Works"
    ]
)


language = st.sidebar.selectbox(
    "Response language:",
    [
        "English",
        "Urdu"
    ]
)


if st.sidebar.button(
    "Clear Conversation"
):

    st.session_state.chat_history = []

    st.rerun()


# ============================================================
# HEALTH CHAT
# ============================================================

if mode == "Health Chat":

    st.header(
        "💬 Health Information Chat"
    )

    st.write(
        "Ask questions about diseases, symptoms, "
        "prevention and general health."
    )


    # Display previous conversation
    if st.session_state.chat_history:

        st.subheader(
            "Conversation"
        )

        for message in st.session_state.chat_history:

            if message["role"] == "user":

                with st.chat_message("user"):
                    st.markdown(
                        message["content"]
                    )

            else:

                with st.chat_message("assistant"):
                    st.markdown(
                        message["content"]
                    )


    question = st.text_area(

        "Enter your health question:",

        height=120,

        placeholder=(
            "Example: What are the common symptoms "
            "of dengue and when should someone see a doctor?"
        )
    )


    if st.button(
        "Get Health Information",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a health question."
            )

        else:

            with st.spinner(
                "Searching medical knowledge..."
            ):

                emergency = emergency_check(
                    question
                )

                results = search_knowledge(
                    question,
                    top_k=5,
                    minimum_score=0.25
                )

                context = build_context(
                    results
                )


            with st.spinner(
                "Generating structured response..."
            ):

                try:

                    answer = ask_groq(
                        question,
                        context,
                        language
                    )

                except Exception as e:

                    st.error(
                        "Unable to generate a response."
                    )

                    st.code(
                        str(e)
                    )

                    st.stop()


            # Save conversation
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


            # Emergency warning
            if emergency:

                if language == "Urdu":

                    st.error(
                        "⚠️ ممکنہ ہنگامی علامات کا پتہ چلا ہے۔ "
                        "براہِ کرم فوری طور پر طبی مدد حاصل کریں۔"
                    )

                else:

                    st.error(
                        "⚠️ Possible emergency warning signs "
                        "were detected. Please seek urgent "
                        "professional medical care."
                    )


            st.subheader(
                "🩺 Health Information"
            )

            st.markdown(
                answer
            )


            # RAG retrieval
            if results:

                with st.expander(
                    "🔍 Show Retrieved Medical Knowledge"
                ):

                    for item in results:

                        score = item.get(
                            "similarity_score",
                            0
                        )

                        st.markdown(
                            f"**{item.get('condition', 'Unknown')}** "
                            f"— {item.get('category', 'Unknown')}"
                        )

                        st.caption(
                            f"Relevance score: {score:.2f}"
                        )

                        st.write(
                            item.get(
                                "text",
                                ""
                            )
                        )

                        st.divider()


            # Sources
            sources = get_unique_sources(
                results
            )

            if sources:

                st.subheader(
                    "📚 Medical Sources"
                )

                for source in sources:

                    if source["url"]:

                        st.markdown(
                            f"- **{source['name']}** — "
                            f"{source['url']}"
                        )

                    else:

                        st.markdown(
                            f"- **{source['name']}**"
                        )


# ============================================================
# IMAGE SCREENING
# ============================================================

elif mode == "Image Screening":

    st.header(
        "🖼️ AI Image Screening"
    )

    st.info(
        "Upload an image of a visible skin concern. "
        "The AI will describe visible features and provide "
        "general educational information. It does not "
        "diagnose disease."
    )


    uploaded_image = st.file_uploader(

        "Upload an image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    extra_information = st.text_area(

        "Additional information (optional):",

        placeholder=(
            "Example: The area has been itchy "
            "for three days."
        )
    )


    if uploaded_image:

        image_bytes = (
            uploaded_image.getvalue()
        )


        # File size limit
        if len(image_bytes) > 4 * 1024 * 1024:

            st.error(
                "Image is too large. "
                "Please upload an image smaller than 4 MB."
            )

            st.stop()


        # Validate image
        try:

            image = Image.open(
                uploaded_image
            )

            image.verify()

        except Exception:

            st.error(
                "The uploaded file is not a valid image."
            )

            st.stop()


        # Display image
        st.image(
            uploaded_image,
            caption="Uploaded image",
            width="stretch"
        )


        if st.button(
            "Analyze Image",
            type="primary"
        ):

            with st.spinner(
                "Analyzing visible features..."
            ):

                try:

                    visual_findings = analyze_image(
                        image_bytes,
                        extra_information,
                        language
                    )

                except Exception as e:

                    st.error(
                        "Unable to analyze the image."
                    )

                    st.code(
                        str(e)
                    )

                    st.stop()


            st.subheader(
                "🔎 Visual Screening"
            )

            st.markdown(
                visual_findings
            )


            # Search knowledge base
            with st.spinner(
                "Searching related medical knowledge..."
            ):

                results = search_knowledge(
                    visual_findings,
                    top_k=5,
                    minimum_score=0.25
                )

                context = build_context(
                    results
                )


            # Generate final response
            with st.spinner(
                "Generating health information..."
            ):

                try:

                    final_answer = ask_groq(
                        visual_findings,
                        context,
                        language
                    )

                except Exception as e:

                    st.error(
                        "Unable to generate related "
                        "health information."
                    )

                    st.code(
                        str(e)
                    )

                    st.stop()


            st.subheader(
                "🩺 Related Health Information"
            )

            st.markdown(
                final_answer
            )


            # Retrieval
            if results:

                with st.expander(
                    "🔍 Show Retrieved Medical Knowledge"
                ):

                    for item in results:

                        score = item.get(
                            "similarity_score",
                            0
                        )

                        st.markdown(
                            f"**{item.get('condition', 'Unknown')}** "
                            f"— {item.get('category', 'Unknown')}"
                        )

                        st.caption(
                            f"Relevance score: {score:.2f}"
                        )

                        st.write(
                            item.get(
                                "text",
                                ""
                            )
                        )

                        st.divider()


            # Sources
            sources = get_unique_sources(
                results
            )

            if sources:

                st.subheader(
                    "📚 Medical Sources"
                )

                for source in sources:

                    if source["url"]:

                        st.markdown(
                            f"- **{source['name']}** — "
                            f"{source['url']}"
                        )

                    else:

                        st.markdown(
                            f"- **{source['name']}**"
                        )


# ============================================================
# HOW IT WORKS
# ============================================================

elif mode == "How It Works":

    st.header(
        "⚙️ How the AI Disease & Health Assistant Works"
    )


    st.write(
        "This application combines semantic search, "
        "a medical knowledge base and a generative AI model."
    )


    st.subheader(
        "1. User Question"
    )

    st.write(
        "The user enters a health-related question "
        "in English or Urdu."
    )


    st.subheader(
        "2. Sentence Transformer"
    )

    st.write(
        "The question is converted into a numerical "
        "embedding that represents its meaning."
    )


    st.subheader(
        "3. FAISS Semantic Search"
    )

    st.write(
        "FAISS searches the medical knowledge base "
        "for information that is semantically relevant "
        "to the user's question."
    )


    st.subheader(
        "4. Medical Knowledge"
    )

    st.write(
        "The most relevant knowledge-base records are "
        "provided to the language model as context."
    )


    st.subheader(
        "5. Groq + Qwen"
    )

    st.write(
        "Groq runs the language model, which uses the "
        "retrieved information to generate a natural-language "
        "response."
    )


    st.subheader(
        "6. Structured Response"
    )

    st.write(
        "The response is instructed to organize information "
        "into useful sections such as:"
    )

    st.markdown(
        """
        - **Category**
        - **Possible Causes**
        - **Common Symptoms**
        - **Basic Healthcare**
        - **When to See a Doctor**
        - **Important**
        """
    )


    st.subheader(
        "7. Safety Layer"
    )

    st.write(
        "The application also checks user questions for "
        "potential emergency warning signs and displays "
        "an urgent-care warning when appropriate."
    )


    st.subheader(
        "8. Image Screening"
    )

    st.write(
        "For uploaded images, the multimodal AI analyzes "
        "visible features only. It does not treat an image "
        "as proof of a diagnosis."
    )


    st.divider()


    st.subheader(
        "Architecture"
    )

    st.code(
        """
User
  ↓
Streamlit Interface
  ↓
Sentence Transformer
  ↓
FAISS Semantic Search
  ↓
Medical Knowledge Base
  ↓
Relevant Context
  ↓
Groq / Qwen
  ↓
Structured Health Information
        """,
        language="text"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🩺 AI Disease & Health Assistant | "
    "Educational use only | "
    "Not a substitute for professional medical care"
)
