"""Embed every chunk in corpus/processed/chunks.jsonl and store it in a local
Chroma collection, then run one test query to sanity-check retrieval.

Run with:  python scripts/build_index.py
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path(__file__).parent.parent / "corpus" / "processed" / "chunks.jsonl"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "primary_sources"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

TEST_QUESTION = "Who converted the Anglo-Saxons to Christianity?"


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


def main():
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' "
          f"(first run downloads it, ~80MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print("Embedding all chunks (this may take a minute on CPU)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

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
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[to_chroma_metadata(c) for c in chunks],
    )
    print(f"Stored {collection.count()} chunks in collection '{COLLECTION_NAME}' "
          f"at {CHROMA_DIR}")

    print(f"\n--- Test query: {TEST_QUESTION!r} ---")
    query_embedding = model.encode([TEST_QUESTION])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=3)

    for rank, (doc, meta, distance) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ), start=1):
        print(f"\n#{rank} (distance={distance:.3f}) "
              f"{meta['source_title']} — chapter: {meta['chapter'] or '(none)'}")
        print(doc[:300].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
