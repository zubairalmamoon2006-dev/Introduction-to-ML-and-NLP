# WRITEUP — IITB Insti-Assist (Academic Assistant)
### WnCC LS'26 NLP Final Project

## 1. Chosen scope and why

I chose the **Academic Assistant** scope: course registration, grading
policy (SPI/CPI), the academic calendar, examination rules, academic
probation, and branch change/minors/honours.

I picked this over Hostel & Campus Life or Council/Club because academic
policy questions are the ones I (and most students I know) actually
mistype into a search engine at 2 a.m. — "what's the minimum CPI before
probation," "does W affect my CPI," "can I still add a course in week 2" —
and because the domain has a genuinely *closed, rule-based* structure
(grade tables, deadlines, credit thresholds) that makes the "ground it or
say I don't know" requirement meaningfully testable: it's easy to write
questions that are clearly in-scope, clearly out-of-scope, or
plausible-sounding-but-unanswerable, and check the assistant's behavior on
each. A "General Insti Assistant" would have been harder to keep honest
within a week, and Hostel/Council content is comparatively less rule-dense.

## 2. Data sources used

The knowledge base (`data/`) consists of **6 Markdown documents**, one per
sub-topic: grading system, course registration, academic calendar,
examination rules, academic probation, and branch change/minors/honours.

**Important limitation, stated honestly (in the spirit of the project's
own "don't hallucinate" goal):** these documents are ones I wrote,
summarizing IIT Bombay's *publicly known, general* academic structure
(the 10-point AA–FF grading scale, SPI/CPI formulas, the
pre-registration → registration → add/drop → withdrawal pipeline, the
two-semester calendar shape, UFM/exam-conduct handling, probation
mechanics, and branch change/minor/honours rules) — they are **not**
scraped verbatim from the live ASC portal or the current official Academic
Calendar PDF. Two things drove this: (a) this system was built inside a
sandboxed environment without general internet access to IITB's own
portal, and (b) several of the *exact* numeric cutoffs (CPI thresholds for
probation, minimum/maximum credit loads, specific dates) are revised
year-to-year by the Academic Office/Senate and are exactly the kind of
detail that should come from the current, official PDF rather than my
memory — so where I wasn't confident of an exact figure, the documents say
so explicitly (e.g., "commonly cited around 5.0–5.5, though the exact
figure is set... and may differ across years") rather than inventing a
precise number.

**This is designed to be a drop-in placeholder, not a permanent
shortcut.** The pipeline is source-format-agnostic by construction:
`src/ingest.py` already includes `load_pdf_and_chunk()`, and the app has a
live PDF-upload feature, specifically so the real deliverable — this
architecture — can be pointed at the actual official documents (Academic
Calendar PDF, UG Rules & Regulations, department grading circulars, ASC
FAQ pages) the moment they're available, without changing any retrieval or
generation code. I'd treat doing that swap as the very first follow-up
task, not a "someday" item.

## 3. Chunking strategy and why

Documents are split in `src/ingest.py` using a **header-aware, two-stage**
strategy rather than a naive fixed-size sliding window:

1. **Stage 1 — split on Markdown section headers (`##`/`###`).** Each
   document already has semantically distinct sections ("Grade Grievance /
   Revaluation," "Credit System," etc.). Splitting here first guarantees a
   chunk never silently mixes two unrelated policies — which matters a lot
   for a grounding system, since a chunk that blends "grading" and
   "add/drop" content could get retrieved for either query and mislead the
   LLM with irrelevant neighboring text.
2. **Stage 2 — recursive sub-split within a section if it's still >~900
   characters**, first trying to break on paragraph boundaries, falling
   back to a hard character split only if a single paragraph is
   unavoidably long (e.g. the grading table), with a 150-character overlap
   stitched onto the start of each subsequent piece so a rule that happens
   to fall near a split boundary isn't stranded without its immediate
   context on either side.
3. Tiny sections (e.g. a lone sub-header with one sentence) are merged
   forward into the next section rather than kept as their own
   low-signal, easily-over-retrieved chunk.

This produced 37 chunks from 6 documents (~6 chunks/doc), each cleanly
scoped to one policy sub-topic — verified by inspecting the actual output
during development (see the "Chunking OK" check the project's test harness
prints). The trade-off is that chunk sizes are less uniform than a
fixed-window approach, but I judged topical coherence more valuable than
uniformity for a grounding-focused assistant, since the failure mode we
most want to avoid is retrieving a chunk that *looks* relevant by keyword
overlap but doesn't actually answer the question.

## 4. Known limitations / what I'd improve with more time

- **Placeholder knowledge base (see §2).** The single highest-priority
  next step is swapping in the actual official documents — Academic
  Calendar PDF, UG Rules & Regulations booklet, department-specific
  grading circulars — via the PDF ingestion path that already exists.
- **Similarity threshold calibration.** The "I don't know" floor
  (`SIM_FLOOR` in `rag_pipeline.py`) is a fixed cosine-similarity cutoff
  tuned by inspection on this corpus. With more time I'd build a small
  labeled eval set (in-scope / ambiguous / out-of-scope questions) and
  tune the floor against it quantitatively rather than by eyeballing a
  handful of examples, and consider a re-ranker on top of the initial
  FAISS retrieval for the ambiguous middle band.
- **No re-ranking / hybrid search.** Pure dense retrieval (cosine
  similarity over MiniLM embeddings) can miss exact-keyword queries (e.g.
  a specific grade code like "FR") that a lexical/BM25 pass would catch
  more reliably; a hybrid dense+sparse retriever would likely improve
  precision on short, code-like queries.
- **Single embedding model, no evaluation harness for answer quality**
  beyond manual spot-checks — a held-out QA set with both "answerable" and
  deliberately "unanswerable" questions, scored for correct grounding vs.
  hallucination, would make the "doesn't hallucinate" claim measurable
  rather than anecdotal.
- **Session-only PDF upload.** The live-upload bonus feature adds chunks
  only to the in-memory session index; a persistent "admin re-index"
  button to fold uploads into the saved FAISS index would make it useful
  beyond a single session.
