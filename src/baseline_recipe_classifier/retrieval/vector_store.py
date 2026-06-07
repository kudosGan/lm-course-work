"""
ChromaDB-backed vector store for recipe retrieval.

Public API:
  add(records)             — embed and persist a batch of labeled recipes
  query(text, ...)         — retrieve the most similar example per label (k=5 total)
"""

import ollama
import chromadb

CHROMA_PATH = "retrieval/chroma_db"
COLLECTION_NAME = "recipes"
EMBED_MODEL = "mxbai-embed-large"

_collection = None


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add(records: list[dict]) -> None:
    """
    Embed and store a batch of labeled recipes.

    Each record must have keys:
      text, label, name, key_ingredients, difficulty_level, mood_tag, prep_time
    Records already in the collection (matched by name as ID) are skipped.
    """
    collection = _get_collection()
    existing_ids = set(collection.get(include=[])["ids"])

    new_records = [r for r in records if r["name"] not in existing_ids]
    if not new_records:
        return

    ids = [r["name"] for r in new_records]
    documents = [r["text"] for r in new_records]
    metadatas = [
        {
            "label": str(r["label"]),
            "name": str(r["name"]),
            "key_ingredients": str(r["key_ingredients"]),
            "difficulty_level": str(r["difficulty_level"]),
            "mood_tag": str(r["mood_tag"]),
            "prep_time": int(r["prep_time"]),
        }
        for r in new_records
    ]
    embeddings = [
        ollama.embed(model=EMBED_MODEL, input=doc)["embeddings"][0]
        for doc in documents
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query(
    text: str,
    k_per_label: int = 1,
    max_prep_time: int = None,
) -> list[dict]:
    """
    Retrieve the most similar recipe per label (label-aware selection).

    Args:
        text:          Query string to embed and search against.
        k_per_label:   Number of results to return per label (default 1).
        max_prep_time: If set, only consider recipes with prep_time <= this value.

    Returns:
        List of dicts with keys: label, name, key_ingredients, difficulty_level, mood_tag.
        prep_time is intentionally excluded to prevent label leakage in the prompt.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    embedding = ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]

    results = []
    for label in ["A", "B", "C", "D", "E"]:
        where = (
            {"$and": [{"label": {"$eq": label}}, {"prep_time": {"$lte": max_prep_time}}]}
            if max_prep_time is not None
            else {"label": {"$eq": label}}
        )

        try:
            result = collection.query(
                query_embeddings=[embedding],
                n_results=k_per_label,
                where=where,
                include=["metadatas"],
            )
            if result["metadatas"] and result["metadatas"][0]:
                meta = result["metadatas"][0][0]
                results.append({
                    "label": meta["label"],
                    "name": meta["name"],
                    "key_ingredients": meta["key_ingredients"],
                    "difficulty_level": meta["difficulty_level"],
                    "mood_tag": meta["mood_tag"],
                    # prep_time excluded — label leak prevention
                })
        except Exception:
            continue

    return results
