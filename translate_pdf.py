#!/usr/bin/env python3
"""Translate a PDF/MOBI book into one or more target languages using local Ollama.

Quality features:
- Overlap-aware chunking for context continuity
- Segment-tagged translation to reduce omissions
- Optional auto glossary generation
- Optional second-pass consistency editing
- QA with numeric-preservation checks and retry queue
- Optional model-based fidelity review/repair
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


DEFAULT_LANGUAGES = ["mandarin"]
DEFAULT_MODEL = "qwen2.5:14b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_CHUNK_CHARS = 2400
DEFAULT_SEGMENT_CHARS = 700
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_OVERLAP_SENTENCES = 2
DEFAULT_GLOSSARY_SIZE = 60
DEFAULT_QA_RETRIES = 2
DEFAULT_MOBI_PAGE_CHARS = 2400

SYSTEM_PROMPT = (
    "You are a precise literary translator. Translate faithfully to the requested target language only. "
    "Do not summarize, omit, explain, or hallucinate facts. "
    "Keep all Arabic numerals and quantitative values unchanged (e.g., 2, 19th, 3.5, 20%). "
    "Return only the translated text in the requested format."
)

CONSISTENCY_SYSTEM_PROMPT = (
    "You are a senior bilingual copy editor. Improve consistency and fluency without changing meaning. "
    "Do not remove details. Keep all numbers unchanged. Return only edited text in the target language."
)

GLOSSARY_SYSTEM_PROMPT = (
    "You create concise terminology glossaries for book translation. Return strict JSON only."
)

FIDELITY_SYSTEM_PROMPT = (
    "You are a translation fidelity reviewer. Compare source and translation, detect mistranslations, "
    "numeric errors, omissions, and hallucinations. Return strict JSON following the schema."
)

BACKTRANSLATION_SYSTEM_PROMPT = (
    "You are a translation QA reviewer. Back-translate target text to English and compare with source text. "
    "Detect meaning drift, factual changes, omissions, and numeric errors. Return strict JSON only."
)

EN_STOPWORDS = {
    "the",
    "and",
    "or",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "have",
    "has",
    "had",
    "was",
    "were",
    "been",
    "will",
    "would",
    "should",
    "could",
    "can",
    "cannot",
    "about",
    "there",
    "their",
    "they",
    "them",
    "then",
    "than",
    "where",
    "when",
    "what",
    "which",
    "while",
    "who",
    "whom",
    "your",
    "you",
    "his",
    "her",
    "its",
    "our",
    "we",
    "not",
    "but",
    "are",
    "is",
    "a",
    "an",
    "in",
    "of",
    "to",
    "on",
    "as",
    "at",
    "by",
}

SPANISH_HINT_WORDS = {
    "de",
    "la",
    "el",
    "que",
    "y",
    "en",
    "los",
    "las",
    "un",
    "una",
    "por",
    "con",
    "para",
    "como",
    "del",
    "se",
    "su",
    "al",
    "es",
    "más",
    "pero",
}


@dataclass
class TranslationOptions:
    model: str
    ollama_url: str
    temperature: float
    timeout_seconds: int
    retries: int
    qa_retries: int
    segment_chars: int
    strict_number_preservation: bool
    fidelity_review: bool
    backtranslate_qa: bool


@dataclass
class Chunk:
    text: str
    overlap_context: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate a PDF or MOBI to one or more languages using a local Ollama model."
    )
    parser.add_argument("book", type=Path, help="Path to source PDF or MOBI.")
    parser.add_argument(
        "-l",
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated target languages (default: spanish,mandarin).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL}).",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        help=f"Max characters per translation chunk (default: {DEFAULT_CHUNK_CHARS}).",
    )
    parser.add_argument(
        "--segment-chars",
        type=int,
        default=DEFAULT_SEGMENT_CHARS,
        help=f"Max characters per tagged segment inside a chunk (default: {DEFAULT_SEGMENT_CHARS}).",
    )
    parser.add_argument(
        "--mobi-page-chars",
        type=int,
        default=DEFAULT_MOBI_PAGE_CHARS,
        help=(
            "Virtual page size when input is MOBI and pagination is reconstructed "
            f"(default: {DEFAULT_MOBI_PAGE_CHARS})."
        ),
    )
    parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=DEFAULT_OVERLAP_SENTENCES,
        help=f"Trailing sentences from previous chunk as context (default: {DEFAULT_OVERLAP_SENTENCES}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature (default: 0.0).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per API call when request fails (default: 3).",
    )
    parser.add_argument(
        "--qa-retries",
        type=int,
        default=DEFAULT_QA_RETRIES,
        help=f"Additional retries for chunks that fail QA (default: {DEFAULT_QA_RETRIES}).",
    )
    parser.add_argument(
        "--second-pass",
        action="store_true",
        help="Run an additional consistency refinement pass per page.",
    )
    parser.add_argument(
        "--fidelity-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run model-based fidelity review/repair per chunk (default: on).",
    )
    parser.add_argument(
        "--backtranslate-qa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run explicit back-translation QA (target -> English) per chunk (default: on).",
    )
    parser.add_argument(
        "--strict-number-preservation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require preserved Arabic numerals/quantities between source and translation (default: on).",
    )
    parser.add_argument(
        "--glossary-file",
        type=Path,
        default=None,
        help="Optional JSON glossary file. Supports {'term':'translation'} or {'language':{'term':'translation'}}.",
    )
    parser.add_argument(
        "--auto-glossary",
        action="store_true",
        help="Auto-generate glossary using source text and local model.",
    )
    parser.add_argument(
        "--glossary-size",
        type=int,
        default=DEFAULT_GLOSSARY_SIZE,
        help=f"Candidate term count for auto glossary generation (default: {DEFAULT_GLOSSARY_SIZE}).",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore cached progress and restart translation for each language.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory root (default: ./output).",
    )
    return parser.parse_args()


def normalize_language(language: str) -> str:
    lang = language.strip().lower()
    aliases = {
        "es": "spanish",
        "espanol": "spanish",
        "español": "spanish",
        "zh": "mandarin",
        "zh-cn": "mandarin",
        "zh-hans": "mandarin",
        "chinese": "mandarin",
        "simplified chinese": "mandarin",
    }
    return aliases.get(lang, lang)


def split_languages(value: str) -> List[str]:
    langs = [normalize_language(item) for item in value.split(",") if item.strip()]
    if not langs:
        raise ValueError("At least one target language is required.")
    return langs


def is_cjk_language(language: str) -> bool:
    lang = normalize_language(language)
    return any(key in lang for key in ("mandarin", "chinese", "zh"))


def safe_lang_slug(language: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", language.lower()).strip("-")


def read_pdf_pages(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text.strip())
    return pages


def strip_html_to_text(raw_html: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw_html, "html.parser")
        # Preserve common structural breaks before text extraction.
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for block in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
            block.append("\n")
        text = soup.get_text(separator="\n")
    except Exception:
        text = re.sub(r"(?i)<br\s*/?>", "\n", raw_html)
        text = re.sub(r"(?i)</(p|div|h[1-6]|li)>", "\n", text)
        text = re.sub(r"<[^>]+>", " ", text)

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_epub_text(epub_path: Path) -> str:
    chunks: List[str] = []
    with zipfile.ZipFile(epub_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        for name in sorted(names):
            try:
                data = zf.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            text = strip_html_to_text(data)
            if text:
                chunks.append(text)
    return "\n\n".join(chunks).strip()


def virtual_paginate_text(text: str, page_chars: int) -> List[str]:
    if len(text) <= page_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pages: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            pages.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > page_chars:
            if current:
                flush()
            for i in range(0, len(paragraph), page_chars):
                piece = paragraph[i : i + page_chars].strip()
                if piece:
                    pages.append(piece)
            continue

        projected = current_len + len(paragraph) + (2 if current else 0)
        if projected > page_chars:
            flush()

        current.append(paragraph)
        current_len += len(paragraph) + (2 if current_len else 0)

    flush()
    return pages or [text]


def read_mobi_pages(mobi_path: Path, page_chars: int) -> List[str]:
    try:
        import mobi  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MOBI input requires the `mobi` package. Install deps with: pip install -r requirements.txt"
        ) from exc

    tempdir: str | None = None
    try:
        tempdir, extracted_filepath = mobi.extract(str(mobi_path))
        extracted_path = Path(extracted_filepath)

        if extracted_path.exists() and extracted_path.suffix.lower() == ".pdf":
            return read_pdf_pages(extracted_path)

        text = ""
        if extracted_path.exists() and extracted_path.suffix.lower() in {".html", ".htm", ".xhtml"}:
            text = strip_html_to_text(extracted_path.read_text(encoding="utf-8", errors="ignore"))
        elif extracted_path.exists() and extracted_path.suffix.lower() == ".epub":
            text = read_epub_text(extracted_path)
        elif extracted_path.exists() and extracted_path.suffix.lower() == ".txt":
            text = extracted_path.read_text(encoding="utf-8", errors="ignore")

        if not text:
            root = Path(tempdir)
            html_candidates = sorted(root.rglob("*.html")) + sorted(root.rglob("*.xhtml")) + sorted(root.rglob("*.htm"))
            if html_candidates:
                chunks = []
                for html_file in html_candidates:
                    raw = html_file.read_text(encoding="utf-8", errors="ignore")
                    t = strip_html_to_text(raw)
                    if t:
                        chunks.append(t)
                text = "\n\n".join(chunks).strip()

        if not text:
            raise RuntimeError("Failed to extract readable text from MOBI. The file may be encrypted or unsupported.")

        return virtual_paginate_text(text, page_chars=page_chars)
    finally:
        if tempdir:
            shutil.rmtree(tempdir, ignore_errors=True)


def read_book_pages(book_path: Path, mobi_page_chars: int) -> List[str]:
    ext = book_path.suffix.lower()
    if ext == ".pdf":
        return read_pdf_pages(book_path)
    if ext == ".mobi":
        return read_mobi_pages(book_path, page_chars=mobi_page_chars)
    raise ValueError(f"Unsupported input format: {book_path.suffix}. Use .pdf or .mobi")


def split_long_text(text: str, max_chars: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sentence), max_chars):
                piece = sentence[i : i + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        projected = len(current) + len(sentence) + (1 if current else 0)
        if projected > max_chars:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current.strip())
    return chunks


def chunk_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for paragraph in paragraphs:
        p = paragraph.strip()
        if not p:
            continue
        if len(p) > max_chars:
            flush()
            chunks.extend(split_long_text(p, max_chars))
            continue

        projected = current_len + len(p) + (2 if current else 0)
        if projected > max_chars:
            flush()

        current.append(p)
        current_len += len(p) + (2 if current_len else 0)

    flush()
    return chunks or [text]


def tail_sentences(text: str, sentence_count: int) -> str:
    if sentence_count <= 0 or not text.strip():
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= sentence_count:
        return text.strip()
    return " ".join(sentences[-sentence_count:]).strip()


def build_chunks_with_overlap(text: str, max_chars: int, overlap_sentences: int) -> List[Chunk]:
    chunks = chunk_text(text, max_chars)
    out: List[Chunk] = []
    prev = ""
    for text_chunk in chunks:
        overlap = tail_sentences(prev, overlap_sentences) if prev else ""
        out.append(Chunk(text=text_chunk, overlap_context=overlap))
        prev = text_chunk
    return out


def split_segments(text: str, max_chars: int) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    segments: List[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            segments.append(current.strip())
            current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            segments.extend(split_long_text(para, max_chars))
            continue

        candidate = para if not current else f"{current}\n\n{para}"
        if len(candidate) > max_chars and current:
            flush()
            current = para
        else:
            current = candidate

    flush()
    return segments


def render_segment_payload(segments: List[str]) -> str:
    lines = []
    for idx, segment in enumerate(segments, start=1):
        lines.append(f"[[S{idx:03d}]]")
        lines.append(segment)
        lines.append("")
    return "\n".join(lines).strip()


def parse_segmented_translation(text: str, expected_count: int) -> List[str] | None:
    pattern = re.compile(r"\[\[S(\d{3})\]\]\s*(.*?)(?=\n\[\[S\d{3}\]\]|\Z)", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None

    by_id: Dict[int, str] = {}
    for raw_idx, content in matches:
        idx = int(raw_idx)
        if idx < 1 or idx > expected_count:
            continue
        by_id[idx] = content.strip()

    if len(by_id) != expected_count:
        return None

    ordered = [by_id[i] for i in range(1, expected_count + 1)]
    if any(not part for part in ordered):
        return None
    return ordered


def ensure_ollama_up(base_url: str, timeout_seconds: int) -> None:
    tags_url = base_url.rstrip("/") + "/api/tags"
    try:
        response = requests.get(tags_url, timeout=timeout_seconds)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}. Start Ollama locally and ensure the API is reachable."
        ) from exc


def call_ollama_chat(
    messages: List[Dict[str, str]],
    options: TranslationOptions,
    *,
    temperature: float | None = None,
    response_format: Any | None = None,
) -> str:
    url = options.ollama_url.rstrip("/") + "/api/chat"
    payload: Dict[str, Any] = {
        "model": options.model,
        "stream": False,
        "messages": messages,
        "options": {"temperature": options.temperature if temperature is None else temperature},
    }
    if response_format is not None:
        payload["format"] = response_format

    last_error: Exception | None = None
    for attempt in range(1, options.retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=options.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "").strip()
            if not content:
                raise RuntimeError("Empty response returned by model.")
            return content
        except Exception as exc:
            last_error = exc
            if attempt < options.retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Model call failed after {options.retries} attempts.") from last_error


def glossary_to_prompt(glossary: Dict[str, str]) -> str:
    if not glossary:
        return ""
    lines = ["Terminology glossary (use these translations consistently):"]
    for term, translation in sorted(glossary.items(), key=lambda x: x[0].lower()):
        lines.append(f"- {term} => {translation}")
    return "\n".join(lines)


def translate_chunk_structured(
    chunk: Chunk,
    target_language: str,
    options: TranslationOptions,
    glossary: Dict[str, str],
    *,
    retry_instruction: str | None = None,
) -> str:
    segments = split_segments(chunk.text, options.segment_chars)
    if not segments:
        return ""

    segment_payload = render_segment_payload(segments)
    glossary_block = glossary_to_prompt(glossary)
    overlap_block = (
        f"Previous context (for reference only, do NOT translate this block):\n{chunk.overlap_context}\n\n"
        if chunk.overlap_context
        else ""
    )
    retry_block = f"Correction note: {retry_instruction}\n\n" if retry_instruction else ""

    user_prompt = (
        f"Target language: {target_language}\n\n"
        "Task:\n"
        "1) Translate each tagged segment.\n"
        "2) Keep every segment tag exactly as-is: [[S001]], [[S002]], ...\n"
        "3) Keep all Arabic numerals and numeric quantities unchanged.\n"
        "4) Do not merge, drop, or reorder segments.\n"
        "5) Return only tagged translated segments.\n\n"
        f"{glossary_block}\n\n"
        f"{retry_block}"
        f"{overlap_block}"
        "Source segments:\n"
        f"{segment_payload}"
    ).strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_output = ""
    for parse_attempt in range(1, options.retries + 1):
        content = call_ollama_chat(messages, options)
        last_output = content
        parsed = parse_segmented_translation(content, len(segments))
        if parsed is not None:
            return "\n\n".join(parsed).strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Your previous output did not preserve all required segment tags. "
                    "Retry and output strictly tagged segments only.\n\n"
                    + user_prompt
                ),
            },
        ]

        if parse_attempt < options.retries:
            time.sleep(0.8 * parse_attempt)

    # Fallback: return raw text if parsing repeatedly fails, to avoid hard-stop.
    return last_output.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def review_and_repair_translation(
    source_text: str,
    translated_text: str,
    target_language: str,
    options: TranslationOptions,
) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "corrected_translation": {"type": "string"},
        },
        "required": ["ok", "issues", "corrected_translation"],
        "additionalProperties": False,
    }

    prompt = (
        f"Target language: {target_language}\n\n"
        "Check whether the translation preserves meaning, details, numbers, and named entities.\n"
        "If issues exist, provide a corrected full translation in the same target language.\n"
        "Return strict JSON only.\n\n"
        f"SOURCE:\n{source_text}\n\n"
        f"TRANSLATION:\n{translated_text}"
    )

    messages = [
        {"role": "system", "content": FIDELITY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    content = call_ollama_chat(messages, options, temperature=0.0, response_format=schema)
    parsed = extract_json_object(content)

    ok = bool(parsed.get("ok")) if isinstance(parsed.get("ok"), bool) else True
    issues = parsed.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    issues = [str(x) for x in issues if str(x).strip()]

    corrected = parsed.get("corrected_translation", "")
    if not isinstance(corrected, str) or not corrected.strip():
        corrected = translated_text

    return {"ok": ok, "issues": issues, "corrected_translation": corrected.strip()}


def backtranslate_and_check(
    source_text: str,
    translated_text: str,
    target_language: str,
    options: TranslationOptions,
) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "back_translation_english": {"type": "string"},
        },
        "required": ["ok", "issues", "back_translation_english"],
        "additionalProperties": False,
    }

    prompt = (
        f"Target language of translation: {target_language}\n\n"
        "Step 1: Back-translate the TRANSLATION text into natural English.\n"
        "Step 2: Compare SOURCE and your back-translation.\n"
        "Step 3: Mark ok=false if meaning/facts/numbers/entities changed or omitted.\n"
        "Return strict JSON only.\n\n"
        f"SOURCE:\n{source_text}\n\n"
        f"TRANSLATION:\n{translated_text}"
    )
    messages = [
        {"role": "system", "content": BACKTRANSLATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    content = call_ollama_chat(messages, options, temperature=0.0, response_format=schema)
    parsed = extract_json_object(content)

    ok = bool(parsed.get("ok")) if isinstance(parsed.get("ok"), bool) else True
    issues = parsed.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    issues = [str(x) for x in issues if str(x).strip()]

    back_translation = parsed.get("back_translation_english", "")
    if not isinstance(back_translation, str):
        back_translation = ""

    return {
        "ok": ok,
        "issues": issues,
        "back_translation_english": back_translation.strip(),
    }


def auto_extract_candidate_terms(pages: List[str], size: int) -> List[str]:
    text = "\n".join(pages)

    proper_nouns = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
    common_words = re.findall(r"\b[A-Za-z]{4,}\b", text.lower())

    freq: Dict[str, int] = {}
    for item in proper_nouns:
        term = item.strip()
        if len(term) < 4:
            continue
        freq[term] = freq.get(term, 0) + 3

    for word in common_words:
        if word in EN_STOPWORDS:
            continue
        freq[word] = freq.get(word, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0].lower()))
    out: List[str] = []
    seen = set()
    for term, _ in ranked:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= size:
            break
    return out


def generate_auto_glossary(terms: List[str], target_language: str, options: TranslationOptions) -> Dict[str, str]:
    if not terms:
        return {}

    schema = {
        "type": "object",
        "additionalProperties": {"type": "string"},
    }
    glossary: Dict[str, str] = {}

    batch_size = 30
    for i in range(0, len(terms), batch_size):
        batch = terms[i : i + batch_size]
        numbered_terms = "\n".join(f"- {term}" for term in batch)
        user_prompt = (
            f"Build a glossary for translation into {target_language}. "
            "Return a JSON object mapping source term to target-language translation. "
            "Use concise phrase-level translations only.\n\n"
            f"Terms:\n{numbered_terms}"
        )
        messages = [
            {"role": "system", "content": GLOSSARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        content = call_ollama_chat(messages, options, temperature=0.0, response_format=schema)
        parsed = extract_json_object(content)
        for key, value in parsed.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            term = key.strip()
            translation = value.strip()
            if term and translation:
                glossary[term] = translation

    return glossary


def load_user_glossary(glossary_path: Path, language: str) -> Dict[str, str]:
    if not glossary_path.exists():
        raise FileNotFoundError(f"Glossary file not found: {glossary_path}")

    with glossary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Glossary JSON must be an object.")

    lang_norm = normalize_language(language)

    if all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        return {k.strip(): v.strip() for k, v in data.items() if k.strip() and v.strip()}

    lang_block = data.get(lang_norm)
    if isinstance(lang_block, dict):
        return {
            str(k).strip(): str(v).strip()
            for k, v in lang_block.items()
            if str(k).strip() and str(v).strip()
        }

    return {}


def english_word_ratio(text: str) -> float:
    words = re.findall(r"\b[A-Za-z]+\b", text)
    if not words:
        return 0.0
    common = sum(1 for w in words if w.lower() in EN_STOPWORDS)
    return common / max(1, len(words))


def cjk_char_ratio(text: str) -> float:
    if not text.strip():
        return 0.0
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    visible_chars = re.findall(r"\S", text)
    return len(cjk_chars) / max(1, len(visible_chars))


def spanish_hint_ratio(text: str) -> float:
    words = re.findall(r"\b[\wáéíóúñü]+\b", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in SPANISH_HINT_WORDS)
    return hits / max(1, len(words))


def extract_numeric_tokens(text: str) -> List[str]:
    raw = re.findall(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|st|nd|rd|th)?", text)
    normalized = [token.replace(",", "") for token in raw]
    return normalized


def qa_check_translation(
    source: str,
    translated: str,
    language: str,
    *,
    strict_number_preservation: bool,
) -> List[str]:
    issues: List[str] = []

    src_len = len(source.strip())
    out_len = len(translated.strip())
    lang = normalize_language(language)

    min_ratio = 0.22 if is_cjk_language(lang) else 0.42
    if src_len > 0 and out_len < max(15, int(src_len * min_ratio)):
        issues.append("translation appears too short vs source")

    eng_ratio = english_word_ratio(translated)
    if is_cjk_language(lang):
        if cjk_char_ratio(translated) < 0.12:
            issues.append("low CJK character ratio for Mandarin target")
        if eng_ratio > 0.65:
            issues.append("high English carry-over for Mandarin target")
    elif lang == "spanish":
        if spanish_hint_ratio(translated) < 0.01 and eng_ratio > 0.65:
            issues.append("language mismatch risk for Spanish target")
    else:
        if eng_ratio > 0.75:
            issues.append("high English carry-over for non-English target")

    if strict_number_preservation:
        src_nums = Counter(extract_numeric_tokens(source))
        out_nums = Counter(extract_numeric_tokens(translated))
        if src_nums and src_nums != out_nums:
            issues.append("numeric mismatch between source and translation")

    return issues


def load_progress(progress_file: Path, pages_count: int, force_restart: bool) -> Dict[str, object]:
    if force_restart:
        return {"pages": [None] * pages_count}

    if progress_file.exists():
        with progress_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        pages = data.get("pages", [])
        if isinstance(pages, list) and len(pages) == pages_count:
            return data
    return {"pages": [None] * pages_count}


def save_progress(progress_file: Path, progress: Dict[str, object]) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with progress_file.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def wrap_cjk_line(
    c: canvas.Canvas,
    paragraph: str,
    max_width: float,
    font_name: str,
    font_size: int,
) -> List[str]:
    lines: List[str] = []
    current = ""
    for char in paragraph:
        candidate = current + char
        if c.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def wrap_text_lines(
    c: canvas.Canvas,
    text: str,
    max_width: float,
    font_name: str,
    font_size: int,
    cjk: bool,
) -> List[str]:
    lines: List[str] = []
    for raw_paragraph in text.splitlines():
        paragraph = raw_paragraph.rstrip()
        if not paragraph:
            lines.append("")
            continue
        if cjk:
            lines.extend(wrap_cjk_line(c, paragraph, max_width, font_name, font_size))
            continue

        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if c.stringWidth(candidate, font_name, font_size) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def write_text_output(pages: Iterable[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for idx, page in enumerate(pages, start=1):
            f.write(f"===== Page {idx} =====\n")
            f.write(page.strip())
            f.write("\n\n")


def write_pdf_output(pages: List[str], path: Path, language: str) -> None:
    page_width, page_height = letter
    margin = 54
    line_height = 14
    font_size = 11
    font_name = "Helvetica"
    cjk = is_cjk_language(language)

    if cjk:
        font_name = "STSong-Light"
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont(font_name, font_size)
    max_width = page_width - 2 * margin

    for page_index, page_text in enumerate(pages):
        if page_index > 0:
            c.showPage()
            c.setFont(font_name, font_size)

        y = page_height - margin
        lines = wrap_text_lines(c, page_text, max_width, font_name, font_size, cjk)
        for line in lines:
            if y < margin:
                c.showPage()
                c.setFont(font_name, font_size)
                y = page_height - margin
            c.drawString(margin, y, line)
            y -= line_height

    c.save()


def refine_consistency(
    translated_text: str,
    target_language: str,
    options: TranslationOptions,
    glossary: Dict[str, str],
) -> str:
    glossary_block = glossary_to_prompt(glossary)
    user_prompt = (
        f"Target language: {target_language}\n\n"
        f"{glossary_block}\n\n"
        "Refine the translation for consistent terminology and natural flow. "
        "Do not remove details, do not alter numeric facts, and do not change meaning.\n\n"
        f"{translated_text}"
    )
    messages = [
        {"role": "system", "content": CONSISTENCY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt.strip()},
    ]
    return call_ollama_chat(messages, options)


def translate_pages(
    pages: List[str],
    language: str,
    chunk_chars: int,
    overlap_sentences: int,
    options: TranslationOptions,
    progress_file: Path,
    glossary: Dict[str, str],
    second_pass: bool,
    force_restart: bool,
) -> tuple[List[str], Dict[str, Any]]:
    progress = load_progress(progress_file, len(pages), force_restart=force_restart)
    stored_pages: List[str | None] = progress["pages"]  # type: ignore[assignment]

    qa_report: Dict[str, Any] = {
        "language": language,
        "qa_retries": options.qa_retries,
        "fidelity_review": options.fidelity_review,
        "backtranslate_qa": options.backtranslate_qa,
        "issues": [],
    }

    for page_idx, source_text in enumerate(pages):
        if stored_pages[page_idx] is not None:
            continue

        if not source_text.strip():
            stored_pages[page_idx] = ""
            save_progress(progress_file, progress)
            continue

        chunks = build_chunks_with_overlap(source_text, chunk_chars, overlap_sentences)
        translations: List[str] = []

        for chunk_idx, chunk in enumerate(chunks, start=1):
            print(
                f"[{language}] page {page_idx + 1}/{len(pages)} chunk {chunk_idx}/{len(chunks)}",
                flush=True,
            )

            translated = translate_chunk_structured(chunk, language, options, glossary)
            issues = qa_check_translation(
                chunk.text,
                translated,
                language,
                strict_number_preservation=options.strict_number_preservation,
            )

            if issues:
                qa_report["issues"].append(
                    {
                        "page": page_idx + 1,
                        "chunk": chunk_idx,
                        "stage": "initial",
                        "issues": issues,
                    }
                )

            best_translation = translated
            best_issues = issues

            for retry_num in range(1, options.qa_retries + 1):
                if not best_issues:
                    break

                retry_hint = "Fix these issues: " + "; ".join(best_issues)
                retried = translate_chunk_structured(
                    chunk,
                    language,
                    options,
                    glossary,
                    retry_instruction=retry_hint,
                )
                retry_issues = qa_check_translation(
                    chunk.text,
                    retried,
                    language,
                    strict_number_preservation=options.strict_number_preservation,
                )

                qa_report["issues"].append(
                    {
                        "page": page_idx + 1,
                        "chunk": chunk_idx,
                        "stage": f"retry_{retry_num}",
                        "issues": retry_issues,
                    }
                )

                best_translation = retried
                best_issues = retry_issues

            if options.backtranslate_qa:
                bt_review = backtranslate_and_check(chunk.text, best_translation, language, options)
                bt_issues = bt_review.get("issues", []) if not bt_review.get("ok", True) else []
                qa_report["issues"].append(
                    {
                        "page": page_idx + 1,
                        "chunk": chunk_idx,
                        "stage": "backtranslate_review",
                        "issues": bt_issues,
                    }
                )

                if bt_issues:
                    bt_retry_hint = "Fix back-translation mismatch: " + "; ".join(bt_issues)
                    bt_retried = translate_chunk_structured(
                        chunk,
                        language,
                        options,
                        glossary,
                        retry_instruction=bt_retry_hint,
                    )
                    bt_retry_issues = qa_check_translation(
                        chunk.text,
                        bt_retried,
                        language,
                        strict_number_preservation=options.strict_number_preservation,
                    )
                    bt_retry_review = backtranslate_and_check(chunk.text, bt_retried, language, options)
                    bt_retry_review_issues = (
                        bt_retry_review.get("issues", [])
                        if not bt_retry_review.get("ok", True)
                        else []
                    )
                    qa_report["issues"].append(
                        {
                            "page": page_idx + 1,
                            "chunk": chunk_idx,
                            "stage": "backtranslate_retry",
                            "issues": bt_retry_issues + bt_retry_review_issues,
                        }
                    )
                    if not bt_retry_issues and not bt_retry_review_issues:
                        best_translation = bt_retried

            if options.fidelity_review:
                review = review_and_repair_translation(chunk.text, best_translation, language, options)
                qa_report["issues"].append(
                    {
                        "page": page_idx + 1,
                        "chunk": chunk_idx,
                        "stage": "fidelity_review",
                        "issues": review.get("issues", []) if not review.get("ok", True) else [],
                    }
                )
                if not review.get("ok", True):
                    candidate = str(review.get("corrected_translation", "")).strip()
                    if candidate:
                        repaired_issues = qa_check_translation(
                            chunk.text,
                            candidate,
                            language,
                            strict_number_preservation=options.strict_number_preservation,
                        )
                        qa_report["issues"].append(
                            {
                                "page": page_idx + 1,
                                "chunk": chunk_idx,
                                "stage": "fidelity_repair_eval",
                                "issues": repaired_issues,
                            }
                        )
                        if not repaired_issues:
                            best_translation = candidate

            translations.append(best_translation)

        page_translation = "\n\n".join(t.strip() for t in translations if t.strip()).strip()

        if second_pass and page_translation:
            print(f"[{language}] page {page_idx + 1}/{len(pages)} second-pass consistency", flush=True)
            refined = refine_consistency(page_translation, language, options, glossary)
            second_pass_issues = qa_check_translation(
                source_text,
                refined,
                language,
                strict_number_preservation=options.strict_number_preservation,
            )
            qa_report["issues"].append(
                {
                    "page": page_idx + 1,
                    "chunk": 0,
                    "stage": "second_pass",
                    "issues": second_pass_issues,
                }
            )
            if not second_pass_issues:
                page_translation = refined

        stored_pages[page_idx] = page_translation
        save_progress(progress_file, progress)

    unresolved = [
        item for item in qa_report["issues"] if isinstance(item, dict) and item.get("issues")
    ]
    qa_report["issue_count"] = len(unresolved)

    return [page or "" for page in stored_pages], qa_report


def write_qa_report(report: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()

    book_path: Path = args.book
    if not book_path.exists():
        print(f"Error: file not found: {book_path}", file=sys.stderr)
        return 1

    if args.chunk_chars < 300:
        print("Error: --chunk-chars must be >= 300.", file=sys.stderr)
        return 1
    if args.segment_chars < 120:
        print("Error: --segment-chars must be >= 120.", file=sys.stderr)
        return 1

    try:
        languages = split_languages(args.languages)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    options = TranslationOptions(
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        qa_retries=args.qa_retries,
        segment_chars=args.segment_chars,
        strict_number_preservation=args.strict_number_preservation,
        fidelity_review=args.fidelity_review,
        backtranslate_qa=args.backtranslate_qa,
    )

    ensure_ollama_up(options.ollama_url, options.timeout_seconds)
    try:
        pages = read_book_pages(book_path, mobi_page_chars=args.mobi_page_chars)
    except Exception as exc:
        print(f"Error: failed to read input book: {exc}", file=sys.stderr)
        return 1

    if not pages:
        print("Error: no text pages found in input.", file=sys.stderr)
        return 1

    stem = book_path.stem
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / ".progress"
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(pages)} page(s) from: {book_path}")

    candidate_terms: List[str] = []
    if args.auto_glossary:
        candidate_terms = auto_extract_candidate_terms(pages, args.glossary_size)
        print(f"Auto glossary enabled. Extracted {len(candidate_terms)} candidate terms.")

    for language in languages:
        lang_slug = safe_lang_slug(language)
        print(f"\n=== Translating to {language} ===")

        glossary: Dict[str, str] = {}
        if args.glossary_file is not None:
            try:
                glossary.update(load_user_glossary(args.glossary_file, language))
            except Exception as exc:
                print(f"Warning: failed to load glossary file: {exc}")

        if args.auto_glossary and candidate_terms:
            auto_glossary = generate_auto_glossary(candidate_terms, language, options)
            merged = dict(auto_glossary)
            merged.update(glossary)
            glossary = merged

            auto_glossary_path = output_root / f"{stem}.{lang_slug}.glossary.json"
            with auto_glossary_path.open("w", encoding="utf-8") as f:
                json.dump(glossary, f, ensure_ascii=False, indent=2)
            print(f"Wrote glossary: {auto_glossary_path}")

        progress_file = cache_root / f"{stem}.{lang_slug}.json"
        translated_pages, qa_report = translate_pages(
            pages=pages,
            language=language,
            chunk_chars=args.chunk_chars,
            overlap_sentences=args.overlap_sentences,
            options=options,
            progress_file=progress_file,
            glossary=glossary,
            second_pass=args.second_pass,
            force_restart=args.force_restart,
        )

        txt_output = output_root / f"{stem}.{lang_slug}.txt"
        pdf_output = output_root / f"{stem}.{lang_slug}.pdf"
        qa_output = output_root / f"{stem}.{lang_slug}.qa_report.json"

        write_text_output(translated_pages, txt_output)
        write_pdf_output(translated_pages, pdf_output, language)
        write_qa_report(qa_report, qa_output)

        print(f"Wrote: {txt_output}")
        print(f"Wrote: {pdf_output}")
        print(f"Wrote: {qa_output}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
