"""
rag_pipeline.py
---------------
Ties retrieval (VectorStore) together with grounded generation. Supports
either Anthropic Claude or Google Gemini as the generation backend -- pick
whichever API key you have. Retrieval (FAISS + sentence-transformers) is
identical either way; only the final "generate an answer from the
retrieved context" call is swapped.

Grounding strategy
------------------
1. Retrieve top-k chunks for the user's question.
2. If the best chunk's similarity score is below `SIM_FLOOR`, we don't
   even call the LLM -- we short-circuit to "I don't know", since there is
   nothing relevant enough in the knowledge base. This also saves API
   calls for obviously out-of-scope questions.
3. Otherwise, we call the LLM with a strict system prompt: answer *only*
   from the provided <context> chunks, cite which chunk(s) were used, and
   explicitly say "I don't know" if the retrieved context doesn't actually
   contain the answer (retrieval can return topically-similar but
   non-answering chunks -- similarity search finds relevant text, not
   necessarily *sufficient* text, so the LLM itself is the second line of
   defense against hallucination).
4. We surface a simple confidence indicator to the user based on the top
   similarity score, independent of what the LLM claims, so the UI can
   flag cases where the model says it found an answer but retrieval
   quality was actually marginal.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GEMINI = "gemini"

ANTHROPIC_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.5-flash"

SIM_FLOOR = 0.30           # below this, don't even bother calling the LLM
SIM_LOW_CONFIDENCE = 0.45  # between SIM_FLOOR and this: call LLM but flag low confidence
TOP_K = 4

SYSTEM_PROMPT = """You are IITB Insti-Assist, a grounded Q&A assistant for IIT Bombay's \
Academic policies (course registration, grading, academic calendar, exam rules, \
probation, branch change/minors/honours).

Rules you must follow strictly:
1. Answer ONLY using the information in the <context> blocks provided in the user \
message. Do not use outside knowledge about IIT Bombay or any other institute, even \
if you believe you know the answer.
2. Every context block is labeled with a [source: ...] tag. When you use information \
from a block, mention which source(s) it came from in your answer (e.g. "According to \
the Grading System document...").
3. If the context does not contain enough information to answer the question, you MUST \
say clearly: "I don't know based on the available documents." Do not guess, and do not \
fill gaps with general knowledge about IIT Bombay or other institutes.
4. Keep answers concise and factual. Use bullet points for multi-part answers.
5. If the user asks something entirely unrelated to IIT Bombay academics (e.g. general \
trivia, coding help, etc.), politely say this assistant is scoped to IIT Bombay \
academic policy questions only.
"""


@dataclass
class RagAnswer:
    answer: str
    used_chunks: List[tuple]        # list of (chunk, score)
    confidence: str                  # "grounded" | "low_confidence" | "not_grounded"
    top_score: float


class RagPipeline:
    def __init__(self, vector_store, provider: str = PROVIDER_ANTHROPIC,
                 api_key: Optional[str] = None):
        """
        provider: "anthropic" (Claude) or "gemini" (Google Gemini).
        api_key:  if not given, falls back to the ANTHROPIC_API_KEY or
                  GEMINI_API_KEY environment variable, matching `provider`.
        """
        self.store = vector_store
        self.provider = provider

        if provider == PROVIDER_ANTHROPIC:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        elif provider == PROVIDER_GEMINI:
            from google import genai
            self.client = genai.Client(
                api_key=api_key or os.environ.get("GEMINI_API_KEY")
            )
        else:
            raise ValueError(f"Unknown provider: {provider!r} (expected "
                              f"'{PROVIDER_ANTHROPIC}' or '{PROVIDER_GEMINI}')")

    def _build_context_block(self, results):
        blocks = []
        for chunk, score in results:
            blocks.append(
                f"[source: {chunk.doc_title} > {chunk.section} "
                f"(similarity={score:.2f})]\n{chunk.text}"
            )
        return "\n\n---\n\n".join(blocks)

    def _call_llm(self, system_prompt: str, messages: List[dict]) -> str:
        """messages: list of {"role": "user"|"assistant", "content": str},
        ending with the current turn's user message."""
        if self.provider == PROVIDER_ANTHROPIC:
            resp = self.client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1000,
                system=system_prompt,
                messages=messages,
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        elif self.provider == PROVIDER_GEMINI:
            from google.genai import types
            # Gemini's chat roles are "user" / "model" (not "assistant"),
            # and system instructions are passed via GenerateContentConfig
            # rather than as a message in the list.
            contents = []
            for m in messages:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append(types.Content(
                    role=role, parts=[types.Part(text=m["content"])]
                ))
            resp = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1000,
                ),
            )
            return resp.text

    def answer(self, question: str, history: Optional[List[dict]] = None,
               top_k: int = TOP_K) -> RagAnswer:
        results = self.store.search(question, top_k=top_k)

        if not results or results[0][1] < SIM_FLOOR:
            return RagAnswer(
                answer="I don't know based on the available documents. "
                       "This assistant's knowledge base covers IIT Bombay academic "
                       "policy topics (grading, registration, calendar, exams, "
                       "probation, branch change/minors/honours) -- your question "
                       "doesn't appear to match anything in it.",
                used_chunks=[],
                confidence="not_grounded",
                top_score=results[0][1] if results else 0.0,
            )

        top_score = results[0][1]
        confidence = "grounded" if top_score >= SIM_LOW_CONFIDENCE else "low_confidence"

        context_block = self._build_context_block(results)

        # include prior turns for multi-turn follow-up questions (bonus feature)
        convo_msgs = []
        if history:
            for turn in history[-6:]:  # cap context growth
                convo_msgs.append({"role": turn["role"], "content": turn["content"]})

        user_msg = (
            f"<context>\n{context_block}\n</context>\n\n"
            f"Question: {question}"
        )

        messages = convo_msgs + [{"role": "user", "content": user_msg}]
        text = self._call_llm(SYSTEM_PROMPT, messages)

        return RagAnswer(
            answer=text,
            used_chunks=results,
            confidence=confidence,
            top_score=top_score,
        )
