"""
Week 2 — Embedding Evaluation
Evaluates nomic-embed-text on recipe time-category pairs.
Similar pairs should score high cosine similarity; dissimilar pairs low.
"""

import json
from pathlib import Path
import numpy as np
import ollama

EMBED_MODEL = "mxbai-embed-large"

# Pairs for the recipe time-classification domain.
# Label: 1 = should be similar (same time category), 0 = should be dissimilar.
PAIRS = [
    # --- Similar pairs (same time category) ---
    ("Scrambled Eggs", "Fried Eggs", 1),                         # A vs A  (quick, no-wait)
    ("Grilled Cheese Sandwich", "Quesadilla", 1),                # A vs A
    ("Spaghetti Carbonara", "Stir-Fried Noodles", 1),            # B vs B  (medium, no-wait)
    ("Roast Chicken", "Baked Lasagna", 1),                       # C vs C  (long, no-wait)
    ("Overnight Oats", "Cold Brew Coffee", 1),                   # E vs E  (long wait)
    # --- Dissimilar pairs (different time categories) ---
    ("Scrambled Eggs", "Slow Cooker Beef Stew", 0),              # A vs C
    ("Instant Ramen", "Overnight Marinated Ribs", 0),            # A vs E
    ("Toast with Avocado", "Braised Pork Shoulder", 0),          # A vs C
    ("Microwave Oatmeal", "Pickled Vegetables", 0),              # A vs E
    ("Pasta with Marinara", "Sourdough Bread", 0),               # B vs E  (bread = long ferment)
]


def embed(text: str) -> np.ndarray:
    resp = ollama.embed(model=EMBED_MODEL, input=text)
    return np.array(resp.embeddings[0])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print(f"Model: {EMBED_MODEL}\n")
    print(f"{'Input A':<35} {'Input B':<35} {'Label':<12} {'Cosine':>7}")
    print("-" * 95)

    results = []
    for text_a, text_b, label in PAIRS:
        vec_a = embed(text_a)
        vec_b = embed(text_b)
        score = cosine_similarity(vec_a, vec_b)
        results.append((text_a, text_b, label, score))
        label_str = "similar" if label == 1 else "dissimilar"
        print(f"{text_a:<35} {text_b:<35} {label_str:<12} {score:>7.4f}")

    # Summary
    similar_scores = [s for _, _, lbl, s in results if lbl == 1]
    dissimilar_scores = [s for _, _, lbl, s in results if lbl == 0]
    print()
    print(f"Avg cosine — similar pairs:    {np.mean(similar_scores):.4f}")
    print(f"Avg cosine — dissimilar pairs: {np.mean(dissimilar_scores):.4f}")
    print(f"Separation:                    {np.mean(similar_scores) - np.mean(dissimilar_scores):.4f}")

    # Correct predictions using 0.5 threshold
    correct = sum(
        (lbl == 1 and s >= 0.5) or (lbl == 0 and s < 0.5)
        for _, _, lbl, s in results
    )
    print(f"Accuracy (threshold=0.5):      {correct}/{len(results)} = {correct/len(results):.0%}")

    # Save results to JSON for the spec
    output = {
        "model": EMBED_MODEL,
        "pairs": [
            {"a": a, "b": b, "label": lbl, "cosine": round(s, 4)}
            for a, b, lbl, s in results
        ],
        "summary": {
            "avg_similar": round(np.mean(similar_scores), 4),
            "avg_dissimilar": round(np.mean(dissimilar_scores), 4),
            "separation": round(np.mean(similar_scores) - np.mean(dissimilar_scores), 4),
            "accuracy_at_0_5": f"{correct}/{len(results)}",
        },
    }
    out_path = Path(__file__).parent / "embedding_eval_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path.name}")


if __name__ == "__main__":
    main()
