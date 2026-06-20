"""
Embedding quality evaluation for the recipe time classifier.
Measures cosine similarity gap between similar and dissimilar recipe pairs
using nomic-embed-text via Ollama.
"""

import numpy as np
import ollama

PAIRS = [
    # Similar pairs — both quick weeknight meals
    {"text_a": "Aglio e olio", "text_b": "Fried rice", "expected": "similar"},
    {"text_a": "Fried rice", "text_b": "Shakshuka", "expected": "similar"},
    {"text_a": "Aglio e olio", "text_b": "Shakshuka", "expected": "similar"},
    # Dissimilar pairs — quick vs. long
    {"text_a": "Aglio e olio", "text_b": "Beef bourguignon", "expected": "dissimilar"},
    {"text_a": "Shakshuka", "text_b": "Whole roast chicken", "expected": "dissimilar"},
    {"text_a": "Fried rice", "text_b": "Fresh croissants", "expected": "dissimilar"},
    # Edge case — looks fast but is slow due to hidden time (pressure build/release)
    {"text_a": "Aglio e olio", "text_b": "Instant Pot beef stew", "expected": "dissimilar"},
]

# Metadata prepended to each recipe name to help the model distinguish by time, not topic.
# Applied when raw recipe names produce a gap below 0.10 (broken threshold).
METADATA = {
    "Aglio e olio": "quick, 15min",
    "Fried rice": "quick, 20min",
    "Shakshuka": "quick, 20min",
    "Beef bourguignon": "slow, 3hr",
    "Whole roast chicken": "slow, 90min",
    "Fresh croissants": "slow, overnight",
    "Instant Pot beef stew": "slow, 45min",
}

USE_METADATA = True  # Set to False to revert to raw recipe names

MODEL = "nomic-embed-text"


def embed(text: str) -> np.ndarray:
    response = ollama.embeddings(model=MODEL, prompt=text)
    return np.array(response["embedding"])


def apply_metadata(text: str) -> str:
    if USE_METADATA and text in METADATA:
        return f"{METADATA[text]}: {text}"
    return text


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def verdict(gap: float) -> str:
    if gap < 0.10:
        return "BROKEN — do not use for this domain"
    elif gap < 0.20:
        return "MARGINAL — works only with very clean input"
    elif gap < 0.30:
        return "USABLE — a classifier can work with this"
    else:
        return "STRONG — retrieval and ranking both reliable"


def main():
    mode = "with metadata" if USE_METADATA else "raw names"
    print(f"Model: {MODEL} ({mode})")
    print(f"Pairs: {len(PAIRS)}\n")
    print(f"{'Text A':<25} {'Text B':<25} {'Expected':<12} {'Score':>6}")
    print("-" * 72)

    similar_scores = []
    dissimilar_scores = []

    for pair in PAIRS:
        vec_a = embed(apply_metadata(pair["text_a"]))
        vec_b = embed(apply_metadata(pair["text_b"]))
        score = cosine_similarity(vec_a, vec_b)

        print(f"{pair['text_a']:<25} {pair['text_b']:<25} {pair['expected']:<12} {score:>6.3f}")

        if pair["expected"] == "similar":
            similar_scores.append(score)
        else:
            dissimilar_scores.append(score)

    mean_similar = np.mean(similar_scores)
    mean_dissimilar = np.mean(dissimilar_scores)
    gap = mean_similar - mean_dissimilar

    print()
    print(f"Mean similar score:    {mean_similar:.3f}  (target: 0.75–0.90)")
    print(f"Mean dissimilar score: {mean_dissimilar:.3f}  (target: 0.45–0.65)")
    print(f"Gap:                   {gap:.3f}  (target: 0.20–0.30)")
    print()
    print(f"Verdict: {verdict(gap)}")


if __name__ == "__main__":
    main()
