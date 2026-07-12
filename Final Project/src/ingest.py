"""
ingest.py
---------
Loads raw source documents (Markdown / PDF / plain text) and splits them
into retrievable chunks with metadata (source file, section heading,
chunk index).

Chunking strategy
------------------
IITB policy documents are naturally organised as Markdown with `##`/`###`
section headers. We exploit that structure instead of blindly splitting by
a fixed character window:

1. Split each document on section headers, so a chunk never straddles two
   unrelated policy sections (e.g. "Grade Grievance" and "Credit System"
   never end up in the same chunk).
2. Within a section, if the text is still longer than `MAX_CHARS`, fall
   back to a recursive character splitter with `OVERLAP` characters of
   overlap between consecutive chunks, so a sentence/rule that happens to
   fall on a boundary is not lost from context on either side.
3. Very short sections (e.g. a lone table header) are merged with the
   following section rather than kept as a near-empty, low-signal chunk.

This keeps chunks topically coherent (good for precision at retrieval
time) while still bounding their size (good for embedding quality and
prompt budget).
"""

import os
import re
import glob
from dataclasses import dataclass, field
from typing import List

MAX_CHARS = 900      # soft cap on chunk size (characters, not tokens)
OVERLAP = 150         # overlap between forced sub-splits of a long section
MIN_SECTION_CHARS = 40  # sections shorter than this get merged forward


@dataclass
class Chunk:
    doc_id: str          # e.g. "01_grading_system.md"
    doc_title: str       # human-readable title (from the H1 of the doc)
    section: str         # nearest section heading (H2/H3)
    chunk_index: int     # position of this chunk within its section
    text: str            # the actual chunk text used for embedding
    id: str = field(init=False)

    def __post_init__(self):
        self.id = f"{self.doc_id}::{self.section}::{self.chunk_index}"


def _split_long_section(text: str, max_chars=MAX_CHARS, overlap=OVERLAP) -> List[str]:
    """Recursively split on paragraph, then sentence, then hard character
    boundaries, respecting max_chars, with overlap between pieces."""
    if len(text) <= max_chars:
        return [text.strip()]

    paras = text.split("\n\n")
    pieces, current = [], ""
    for p in paras:
        candidate = (current + "\n\n" + p).strip() if current else p
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            if len(p) <= max_chars:
                current = p
            else:
                # hard split with overlap on very long single paragraphs
                start = 0
                while start < len(p):
                    end = start + max_chars
                    pieces.append(p[start:end])
                    start = end - overlap
                current = ""
    if current:
        pieces.append(current)

    # stitch a bit of overlap onto the *start* of each piece after the first
    stitched = [pieces[0]]
    for i in range(1, len(pieces)):
        tail = pieces[i - 1][-overlap:]
        stitched.append((tail + "\n" + pieces[i]).strip())
    return stitched


def load_and_chunk(data_dir: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    md_files = sorted(glob.glob(os.path.join(data_dir, "*.md")))

    for path in md_files:
        doc_id = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        # doc title = first H1
        title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else doc_id

        # split on H2/H3 headers, keeping the header text
        parts = re.split(r"^(#{2,3}\s+.+)$", raw, flags=re.MULTILINE)
        # parts alternates: [preamble, header1, body1, header2, body2, ...]
        sections = []
        # strip the H1 title line itself out of the preamble -- it carries
        # no retrievable content of its own and would otherwise create a
        # near-duplicate "Overview" chunk for every document.
        preamble = re.sub(r"^#\s+.+$", "", parts[0], flags=re.MULTILINE).strip()
        if len(preamble) > MIN_SECTION_CHARS:
            sections.append(("Overview", preamble))
        for i in range(1, len(parts), 2):
            header = parts[i].lstrip("#").strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections.append((header, body))

        # merge tiny sections forward into the next one
        merged = []
        pending_header, pending_body = None, ""
        for header, body in sections:
            if pending_header is None:
                pending_header, pending_body = header, body
            elif len(pending_body) < MIN_SECTION_CHARS:
                pending_body = pending_body + "\n\n" + body
            else:
                merged.append((pending_header, pending_body))
                pending_header, pending_body = header, body
        if pending_header is not None:
            merged.append((pending_header, pending_body))

        for header, body in merged:
            if not body.strip():
                continue
            for idx, piece in enumerate(_split_long_section(body)):
                if piece.strip():
                    chunks.append(Chunk(
                        doc_id=doc_id,
                        doc_title=doc_title,
                        section=header,
                        chunk_index=idx,
                        text=piece,
                    ))

    return chunks


def load_pdf_and_chunk(pdf_path: str) -> List[Chunk]:
    """Optional path for the 'live PDF upload' bonus feature: extract text
    from a user-uploaded PDF and chunk it the same way as the Markdown
    knowledge base (no section headers assumed, so it goes straight to the
    recursive splitter)."""
    import pdfplumber
    doc_id = os.path.basename(pdf_path)
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    full_text = "\n\n".join(text_parts)

    chunks = []
    for idx, piece in enumerate(_split_long_section(full_text)):
        if piece.strip():
            chunks.append(Chunk(
                doc_id=doc_id,
                doc_title=doc_id,
                section="Uploaded PDF",
                chunk_index=idx,
                text=piece,
            ))
    return chunks


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(here, "data")
    chunks = load_and_chunk(data_dir)
    print(f"Loaded {len(chunks)} chunks from {data_dir}")
    for c in chunks[:5]:
        print("-" * 60)
        print(c.id)
        print(c.text[:200], "...")
