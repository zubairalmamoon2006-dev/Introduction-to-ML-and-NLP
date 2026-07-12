# IITB Insti-Assist — Academic Assistant (RAG)

A Retrieval-Augmented Generation assistant that answers questions about
**IIT Bombay's academic policies** — course registration, grading (SPI/CPI),
the academic calendar, examination rules, academic probation, and branch
change / minors / honours — grounded strictly in a local document knowledge
base. It refuses to answer ("I don't know based on the available
documents") when the knowledge base doesn't actually contain the answer.

Built for the WnCC LS'26 NLP Final Project. **Chosen scope: Academic
Assistant.**

---

## How it works

```
data/*.md  --(ingest.py: header-aware chunking)-->  Chunks
Chunks     --(embed_store.py: all-MiniLM-L6-v2)-->  384-dim vectors
Vectors    --(FAISS IndexFlatIP)-->                 index_store/
Query      --(embed + FAISS top-k search)-->        retrieved chunks + scores
Retrieved chunks + query --(rag_pipeline.py + Claude API)--> grounded answer
                                                     (or "I don't know")
```

- **Chunking** (`src/ingest.py`): splits each Markdown document on its
  `##`/`###` section headers first (so a chunk never mixes two unrelated
  policies), then recursively sub-splits any section still longer than
  ~900 characters, with a small overlap between pieces.
- **Embedding** (`src/embed_store.py`): local `sentence-transformers`
  model (`all-MiniLM-L6-v2`) — free, no API calls needed just to embed,
  runs fine on CPU. Vectors are normalized so FAISS's inner-product index
  behaves as cosine similarity.
- **Retrieval**: top-k (default 4) nearest chunks per query.
- **Grounded generation** (`src/rag_pipeline.py`): calls an LLM — **either
  Claude (Anthropic) or Gemini (Google), your choice** — with a system
  prompt that forbids using anything outside the retrieved `<context>`,
  requires citing which source(s) were used, and requires an explicit
  "I don't know" when context is insufficient. A similarity-score floor
  also short-circuits obviously out-of-scope questions before an API call
  is even made. Retrieval is 100% identical regardless of which provider
  you pick — only the final generation call changes.
- **Web UI** (`src/app.py`): Streamlit chat interface.

### Bonus features implemented
- ✅ **Multi-turn conversational memory** — follow-up questions reuse
  recent chat history.
- ✅ **Citation highlighting** — each answer has an expandable "Sources
  used" panel showing the *exact* retrieved chunk text, not just the file
  name.
- ✅ **Live PDF upload** — upload an extra IITB PDF from the sidebar and
  it's chunked + embedded into the running session's index on the fly.
- ✅ **Confidence / grounded indicator** — every answer is tagged
  ✅ Grounded / ⚠️ Low confidence / ❌ Not grounded, based on retrieval
  similarity, independent of what the LLM itself claims.

---

## Setup

```bash
git clone <this-repo-url>
cd academic-assistant-rag
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Set **one** of the following API keys — pick whichever LLM provider you
want to use (you don't need both):

**Option A: Claude (Anthropic)** — get a key at https://console.anthropic.com/
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Option B: Gemini (Google)** — get a free key at https://aistudio.google.com/app/apikey
```bash
export GEMINI_API_KEY=...
```

(You can also just paste the key into the sidebar text box when the app is
running — it's kept only for that session. The sidebar has a toggle to
switch between "Claude (Anthropic)" and "Gemini (Google)" at any time.)

## Build the index (run once, or whenever you edit files in `data/`)

```bash
python build_index.py
```

This downloads `all-MiniLM-L6-v2` the first time (~90 MB, one-time,
requires internet access to huggingface.co) and writes the FAISS index to
`index_store/`.

## Run the app

```bash
streamlit run src/app.py
```

Open the printed local URL (usually `http://localhost:8501`) and start
asking questions, e.g.:
- "What's the minimum CPI to avoid academic probation?"
- "How does the add/drop period work?"
- "What happens if I'm caught using unfair means in an exam?"
- "What is the boiling point of ethanol?" *(should correctly refuse — out of scope)*

---

## Project structure

```
.
├── data/                       # knowledge base source documents (6 docs)
│   ├── 01_grading_system.md
│   ├── 02_course_registration.md
│   ├── 03_academic_calendar.md
│   ├── 04_examination_rules.md
│   ├── 05_academic_probation.md
│   └── 06_branch_change_minors_honours.md
├── src/
│   ├── ingest.py                # document loading + chunking
│   ├── embed_store.py           # embeddings + FAISS vector store
│   ├── rag_pipeline.py          # retrieval + grounded generation
│   └── app.py                   # Streamlit UI
├── build_index.py               # one-time/offline index-building script
├── index_store/                 # generated FAISS index + chunk metadata (gitignored)
├── requirements.txt
├── .env.example
├── WRITEUP.md                   # 2-page project write-up (scope, data, chunking, limitations)
└── README.md
```

## A note on the knowledge base documents

The six documents in `data/` are **representative reference documents I
wrote summarizing IIT Bombay's publicly known academic structure** (grading
scale, SPI/CPI, registration/add-drop workflow, semester calendar shape,
exam conduct rules, probation policy, branch change/minors/honours) —
they are not scraped verbatim from the live ASC/Academic Office portal.
See `WRITEUP.md` for why, and for how to swap in official PDFs/scraped
pages instead (the ingestion pipeline in `src/ingest.py` already has a
`load_pdf_and_chunk()` path ready for that — just drop real PDFs in `data/`
and add a matching loader call in `build_index.py`, or use the in-app PDF
uploader for a quick session-only test).
