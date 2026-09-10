# 🩺 AI Disease & Health Assistant

Streamlit health-information assistant using Groq/Qwen, RAG, FAISS, Sentence Transformers and multimodal image screening.

## Files
- `app.py` — main Streamlit application
- `knowledge_base.json` — curated medical retrieval data
- `requirements.txt` — Python dependencies
- `README.md` — setup/deployment guide
- `.gitignore` — prevents secrets from being committed

## Local setup

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_actual_key_here"
GROQ_TEXT_MODEL = "qwen/qwen3.6-27b"
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"
```

Run:

```bash
streamlit run app.py
```

## Streamlit Community Cloud

1. Upload/push the project files to GitHub.
2. Create a Streamlit app from the repository.
3. Open the app's Settings → Secrets.
4. Add:

```toml
GROQ_API_KEY = "your_actual_key_here"
```

Never put the real API key in GitHub or `app.py`.

## Improvements in this version

- Natural conversational answers instead of rigid repeated headings
- Strong English/Urdu language enforcement
- Follow-up conversation history
- FAISS cosine-similarity retrieval
- Relevance threshold to reduce unrelated context
- Expanded knowledge base
- Better emergency phrase detection
- Image MIME-type support and validation
- 4 MB image limit
- Error handling
- Source display
- "How It Works" page for project demonstrations

## Safety

This is an educational health-information project. It is not a diagnostic tool and does not replace professional medical care.
