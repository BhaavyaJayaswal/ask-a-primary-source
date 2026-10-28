"""Embed every chunk in corpus/processed/chunks.jsonl using Gemini's hosted
embedding API and store it in a local Chroma collection, then run one test
query to sanity-check retrieval.

Run with:  python scripts/build_index.py
"""

import json
import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv()

CHUNKS_PATH = Path(__file__).parent.parent / "corpus" / "processed" / "chunks.jsonl"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "primary_sources"
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBED_BATCH_SIZE = 10
RATE_LIMIT_WAIT_SECONDS = 30
MAX_RETRIES = 6

TEST_QUESTION = "Who converted the Anglo-Saxons to Christianity?"

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def load_chunks() -> list[dict]:
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def to_chroma_metadata(chunk: dict) -> dict:
    """Chroma metadata values can't be None, so swap missing chapters for ""."""
    return {
        "source_title": chunk["source_title"],
        "author": chunk["author"],
        "era": chunk["era"],
        "chapter": chunk["chapter"] or "",
        "word_count": chunk["word_count"],
    }


def embed_batch_with_retry(batch: list[str]):
    """The free tier allows only 30,000 embedding tokens per minute, which
    our corpus blows past quickly. On a 429 (rate limit) error, wait and
    retry the same batch instead of giving up."""
    for attempt in range(MAX_RETRIES):
        try:
            return gemini_client.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=batch,
            )
        except genai_errors.ClientError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                print(f"  Rate limited, waiting {RATE_LIMIT_WAIT_SECONDS}s "
                      f"before retrying (attempt {attempt + 1})...")
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
            else:
                raise


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts via Gemini's API, in small batches to stay
    well under any per-request size limit."""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        response = embed_batch_with_retry(batch)
        all_embeddings.extend(e.values for e in response.embeddings)
        print(f"  Embedded {min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)}")
    return all_embeddings


def main():
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    texts = [c["text"] for c in chunks]
    print(f"Embedding all chunks via Gemini's '{EMBEDDING_MODEL_NAME}' API "
          f"(in batches of {EMBED_BATCH_SIZE})...")
    embeddings = embed_texts(texts)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Delete any old collection of the same name so re-running this script
    # doesn't duplicate or stack stale data from previous attempts.
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    print("Writing into Chroma...")
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[to_chroma_metadata(c) for c in chunks],
    )
    print(f"Stored {collection.count()} chunks in collection '{COLLECTION_NAME}' "
          f"at {CHROMA_DIR}")

    print(f"\n--- Test query: {TEST_QUESTION!r} ---")
    query_embedding = embed_texts([TEST_QUESTION])
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    for rank, (doc, meta, distance) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ), start=1):
        print(f"\n#{rank} (distance={distance:.3f}) "
              f"{meta['source_title']} — chapter: {meta['chapter'] or '(none)'}")
        print(doc[:300].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
