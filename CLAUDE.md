# Baobaopedia GS — Project Documentation

## What This Project Does

A RAG (Retrieval-Augmented Generation) app for pregnancy Q&A. It reads two pregnancy books (PDFs), creates embeddings, stores them locally in ChromaDB, uploads them to Supabase, and answers questions via a Gradio web UI using a Llama model from HuggingFace.

**Three main scripts:**
- `read_pdf.py` — Extracts text from PDFs, chunks it, creates embeddings, stores in local ChromaDB
- `upload_data_to_superbase.py` — Uploads embeddings to Supabase (pgvector)
- `app.py` — Gradio web UI for asking questions (RAG pipeline)

---

## Environment Setup (Poetry)

### Step 1 — Install Python 3.11

Python 3.14 is too new for some ML packages (chromadb, sentence-transformers). Python 3.11 is the sweet spot for compatibility.

```bash
brew install python@3.11
```

Verify:
```bash
python3.11 --version
```

---

### Step 2 — Install Poetry

Poetry is a dependency manager — cleaner than pip, handles virtual environments automatically.

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to your PATH by opening `~/.zshrc`:
```bash
nano ~/.zshrc
```
Add this line at the bottom:
```bash
export PATH="$HOME/.local/bin:$PATH"
```
Press `Ctrl + X`, then `Y` to save and exit.

Apply the changes:
```bash
source ~/.zshrc
```

Verify Poetry is working:
```bash
poetry --version
```

---

### Step 3 — Initialize the Poetry project

Navigate to the project folder:
```bash
cd ~/Desktop/learning/gs/baobaopedia_gs
```

Initialize Poetry (creates `pyproject.toml`):
```bash
poetry init
```
Press Enter to accept defaults. When asked about dependencies, say **no** — we'll add them in the next step.

---

### Step 4 — Tell Poetry to use Python 3.11

```bash
poetry env use python3.11
```

---

### Step 5 — Fix Python version range in pyproject.toml

**Important gotcha:** Poetry defaults to `requires-python = ">=3.11"` which extends to Python 4.0. Some packages (like `langchain-text-splitters`) explicitly don't support Python 4.0 yet, which causes a dependency resolution error.

Open `pyproject.toml` and change:
```
requires-python = ">=3.11"
```
to:
```
requires-python = ">=3.11,<4.0.0"
```

---

### Step 6 — Add all dependencies

```bash
poetry add supabase sentence-transformers huggingface-hub numpy gradio PyPDF2 chromadb langchain-text-splitters python-dotenv
```

This installs 9 libraries into an isolated virtual environment. Takes 3–5 minutes due to the size of `sentence-transformers` and `chromadb`.

---

### Step 7 — Set up API keys

Copy the example env file and fill in your real keys:
```bash
cp .env.example .env
```

Edit `.env` with your actual values:
```
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=eyJhbGci...
HF_TOKEN=hf_xxxx...
```

Where to find each key:
- `SUPABASE_URL` — Supabase project → Settings → API → "Project URL"
- `SUPABASE_KEY` — Same page → "anon public" key
- `HF_TOKEN` — HuggingFace → Settings → Access Tokens

Verify all keys are loaded:
```bash
poetry run python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('SUPABASE_URL:', 'set' if os.environ.get('SUPABASE_URL') else 'MISSING')
print('SUPABASE_KEY:', 'set' if os.environ.get('SUPABASE_KEY') else 'MISSING')
print('HF_TOKEN:', 'set' if os.environ.get('HF_TOKEN') else 'MISSING')
"
```

---

### Step 8 — Verify the environment

```bash
poetry run python -c "import gradio, chromadb, supabase, sentence_transformers; print('All good!')"
```

You should see: `All good!`

---

## Dependencies

| Library | Purpose |
|---|---|
| `supabase` | Supabase client for pgvector database |
| `sentence-transformers` | Creates text embeddings (all-MiniLM-L12-v2 model) |
| `huggingface-hub` | Access Llama 3.1 8B model for inference |
| `numpy` | Numerical operations on embeddings |
| `gradio` | Web UI for the Q&A interface |
| `PyPDF2` | Extract text from PDF books |
| `chromadb` | Local persistent vector database |
| `langchain-text-splitters` | Split PDF text into chunks |
| `python-dotenv` | Loads API keys from `.env` file |

---

## Running the Project

**Step 1: Process PDFs and build local ChromaDB**
```bash
poetry run python read_pdf.py
```

**Step 2: Upload embeddings to Supabase**
```bash
poetry run python upload_data_to_superbase.py
```

**Step 3: Launch the Q&A web app**
```bash
poetry run python app.py
```

---

## API Keys

Keys are stored in `.env` (never committed to git — it's in `.gitignore`).
Use `.env.example` as a safe template to share or reference.

```
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_or_service_role_key
HF_TOKEN=your_huggingface_token
```
