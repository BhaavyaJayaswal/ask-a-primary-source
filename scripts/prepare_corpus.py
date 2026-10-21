"""Clean the raw Gutenberg/sourcebook .txt files in corpus/ and split them into
~500-word chunks with metadata, saved as corpus/processed/chunks.jsonl.

Run with:  python scripts/prepare_corpus.py
"""

import json
import re
from pathlib import Path

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
OUTPUT_PATH = CORPUS_DIR / "processed" / "chunks.jsonl"

TARGET_WORDS_PER_CHUNK = 500

# Map each raw filename to the metadata we want attached to every chunk from it.
SOURCES = {
    "machiavelli_the_prince.txt": {
        "source_title": "The Prince",
        "author": "Niccolò Machiavelli",
        "era": "early modern",
    },
    "tacitus_germania.txt": {
        "source_title": "The Germany and the Agricola of Tacitus",
        "author": "Cornelius Tacitus",
        "era": "ancient",
    },
    "bede_ecclesiastical_history.txt": {
        "source_title": "Bede's Ecclesiastical History of England",
        "author": "Bede",
        "era": "medieval",
    },
}

START_MARKER_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END_MARKER_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)

# A paragraph that's just a chapter/book label, e.g. "CHAPTER IV" or "BOOK II."
HEADING_RE = re.compile(r"^(chapter|book)\s+[ivxlcdm\d]+\.?\b", re.IGNORECASE)

# Some editions preface the book with an editor-written synopsis per book,
# e.g. "BOOK I.—In Book I, cc. 1-22, Bede sketches..." This is the editor's
# paraphrase, not the author's own words, so we drop these paragraphs entirely
# rather than count them as primary-source content or let them override the
# real chapter label.
SYNOPSIS_RE = re.compile(r"^(chapter|book)\s+[ivxlcdm\d]+\.\s*[—-]", re.IGNORECASE)


def strip_boilerplate(text: str) -> str:
    """Cut everything outside Gutenberg's START/END markers."""
    start_match = START_MARKER_RE.search(text)
    end_match = END_MARKER_RE.search(text)
    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(text)
    return text[start:end].strip()


def split_paragraphs(text: str) -> list[str]:
    raw_paragraphs = re.split(r"\n\s*\n+", text)
    return [p.strip().replace("\n", " ") for p in raw_paragraphs if p.strip()]


def chunk_paragraphs(paragraphs: list[str]) -> list[dict]:
    """Group paragraphs into ~500-word chunks, never splitting a paragraph in
    half, and tracking the most recent chapter/book heading as metadata."""
    chunks = []
    current_paragraphs: list[str] = []
    current_words = 0
    current_chapter = None

    def flush():
        if current_paragraphs:
            chunks.append({
                "text": "\n\n".join(current_paragraphs),
                "chapter": current_chapter,
                "word_count": current_words,
            })

    for para in paragraphs:
        if SYNOPSIS_RE.match(para):
            # Editor's per-book preview paragraph — not the author's words,
            # and not a real section marker. Drop it entirely.
            continue

        heading_match = HEADING_RE.match(para)
        if heading_match:
            # A real heading line (e.g. "BOOK I") is short. Some editions glue
            # a long editor's summary onto the same paragraph with no blank
            # line in between — in that case, keep only the heading label as
            # metadata and let the summary text flow into chunks as normal
            # content instead of silently discarding it.
            if len(para.split()) <= 8:
                flush()
                current_paragraphs = []
                current_words = 0
                current_chapter = para
                continue
            else:
                current_chapter = heading_match.group().rstrip(".")

        para_words = len(para.split())
        if current_paragraphs and current_words + para_words > TARGET_WORDS_PER_CHUNK:
            flush()
            current_paragraphs = []
            current_words = 0

        current_paragraphs.append(para)
        current_words += para_words

    flush()
    return chunks


def process_file(filename: str, metadata: dict) -> list[dict]:
    raw_text = (CORPUS_DIR / filename).read_text(encoding="utf-8")
    cleaned = strip_boilerplate(raw_text)
    paragraphs = split_paragraphs(cleaned)
    chunks = chunk_paragraphs(paragraphs)

    slug = filename.replace(".txt", "")
    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "id": f"{slug}_{i:04d}",
            "text": chunk["text"],
            "chapter": chunk["chapter"],
            "word_count": chunk["word_count"],
            **metadata,
        })
    return records


def main():
    all_records = []
    for filename, metadata in SOURCES.items():
        path = CORPUS_DIR / filename
        if not path.exists():
            print(f"SKIPPING {filename} (not found in corpus/)")
            continue
        records = process_file(filename, metadata)
        all_records.extend(records)

        word_counts = [r["word_count"] for r in records]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        print(f"{filename}: {len(records)} chunks, avg {avg_words:.0f} words/chunk, "
              f"{sum(word_counts)} words total")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_records)} total chunks to {OUTPUT_PATH}")
    if all_records:
        print("\n--- Sample chunk ---")
        print(json.dumps(all_records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
