# 🩺 AI Disease & Health Assistant

A Streamlit-based health-information assistant that uses **Groq/Qwen**, **Retrieval-Augmented Generation (RAG)**, **FAISS**, and **Sentence Transformers** to provide educational health information.

The application also includes an **image screening feature** for uploaded health-related images.

> ⚠️ This project is for educational and informational purposes only. It is not a diagnostic tool and does not replace professional medical advice.

## 📁 Project Files

* `app.py` — Main Streamlit application
* `knowledge_base.json` — Curated medical knowledge used for retrieval
* `requirements.txt` — Python dependencies
* `README.md` — Project documentation and setup instructions
* `.gitignore` — Prevents sensitive or unnecessary files from being committed

## ⚙️ Technologies Used

* **Python**
* **Streamlit** — Web application interface
* **Groq API** — AI model inference
* **Qwen** — Language and image analysis model
* **Sentence Transformers** — Text embeddings
* **FAISS** — Similarity search and retrieval
* **NumPy** — Numerical operations
* **Pillow** — Image validation and processing

## 🧠 How It Works

The application follows a Retrieval-Augmented Generation (RAG) approach:

1. The user enters a health-related question.
2. The question is converted into an embedding using Sentence Transformers.
3. FAISS searches the medical knowledge base for relevant information.
4. Relevant retrieved information is provided to the Groq/Qwen model as context.
5. The model generates a structured health-information response.
6. The application can also display the retrieved sources.

For image screening:

1. The user uploads an image.
2. The application validates the uploaded image.
3. The image is sent to the vision-capable model.
4. The model describes visible features without claiming a definitive diagnosis.
5. The user is advised to seek professional medical evaluation when appropriate.

## 📚 Knowledge Base

The `knowledge_base.json` file contains curated health-information records covering topics such as:

* Dengue
* Malaria
* Diabetes
* Asthma
* Influenza
* Tuberculosis
* Pneumonia
* Hypertension
* Viral hepatitis
* Diarrhoeal disease
* Anaemia
* Eczema
* Acne
* Dehydration
* Common cold
* Headache
* Fever
* Food poisoning
* Vitamin D
* Iron
* Allergy
* Allergic rhinitis
* Constipation
* Migraine
* Kidney stones
* Urinary tract infection (UTI)
* Heat-related health concerns
* Sleep
* General health and self-care

The knowledge base is designed so that additional sources and health topics can be added later.

## 🌐 Language Support

The assistant supports:

* 🇬🇧 English
* 🇵🇰 Urdu

The selected language is explicitly passed to the AI model so that responses follow the user's selected language.

## 🚀 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/faiqazafar/AI-Disease-Health-Assistant.git
cd AI-Disease-Health-Assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create Streamlit secrets

Create the following file:

```text
.streamlit/secrets.toml
```

Add your Groq API key:

```toml
GROQ_API_KEY = "your_actual_key_here"
```

If your `app.py` supports custom model settings through secrets, you can also add:

```toml
GROQ_TEXT_MODEL = "qwen/qwen3.6-27b"
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"
```

**Never upload your real API key to GitHub.**

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Streamlit Community Cloud Deployment

1. Push the project files to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from your GitHub repository.
4. Select `app.py` as the main file.
5. Open the application's **Settings → Secrets**.
6. Add:

```toml
GROQ_API_KEY = "your_actual_key_here"
```

7. Save the secrets and allow the application to redeploy.

> 🔐 Never put the real API key inside `app.py`, `knowledge_base.json`, or GitHub.

## ✨ Features

* 💬 Health-information chatbot
* 🧠 Retrieval-Augmented Generation (RAG)
* 🔎 FAISS similarity search
* 📚 Curated medical knowledge base
* 🇬🇧 English and 🇵🇰 Urdu responses
* 💭 Conversation history and follow-up questions
* 🚨 Emergency warning detection
* 🖼️ Health-image screening
* 🔐 Secure API-key handling through Streamlit secrets
* 📖 Retrieved source display
* 🛡️ Error handling and input validation
* 📱 Streamlit-based user interface
* ℹ️ "How It Works" section for project demonstrations

## 🔒 Safety

This application provides general educational health information.

It should **not** be used to:

* Diagnose a medical condition
* Replace a doctor or other qualified healthcare professional
* Make emergency medical decisions without professional help
* Provide personalized medication prescriptions or dosages

If a user reports serious or emergency warning signs, the application should encourage them to seek appropriate urgent medical care.

## 🔮 Future Improvements

Possible future improvements include:

* Expanding the medical knowledge base
* Adding more trusted medical sources
* Improving retrieval accuracy
* Adding multilingual support
* Adding conversation export
* Adding user feedback for retrieved answers
* Improving image-screening capabilities
* Adding additional healthcare information sources

## 👩‍💻 Project

**AI Disease & Health Assistant**

Built with Python, Streamlit, Groq/Qwen, Sentence Transformers, FAISS, and a curated health-information knowledge base.
