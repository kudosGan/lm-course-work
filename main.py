"""
Recipe time classifier — retrieval-based few-shot pipeline.

Usage:
    uv run python main.py

Flow:
    1. Accept ingredients + time limit
    2. Retrieve one labeled example per category from ChromaDB (k=5, label-aware)
    3. Build a few-shot prompt (prep_time hidden — label leak prevention)
    4. Call gemma2:2b to predict the time category
    5. Print the recommended category and its description
"""

import re
import sys
import os
import ollama

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from baseline_recipe_classifier.retrieval.vector_store import query

CATEGORIES = {
    "A": ("Quick, No-Wait",  "Total time ≤ 30 min, no wait time"),
    "B": ("Medium, No-Wait", "30 < total time ≤ 60 min, no wait time"),
    "C": ("Long, No-Wait",   "Total time > 60 min, no wait time"),
    "D": ("Short Wait",      "Wait time 1–30 min (rest, marinate, chill)"),
    "E": ("Long Wait",       "Wait time > 30 min (overnight marinate, long chill)"),
}


def format_few_shot_block(examples: list[dict]) -> str:
    lines = []
    for ex in examples:
        lines.append(
            f"Recipe: {ex['name']}\n"
            f"Ingredients: {ex['key_ingredients']}\n"
            f"Category: {ex['label']}"
        )
    return "\n\n".join(lines)


def build_prompt(ingredients: str, time_limit: str, examples: list[dict]) -> str:
    category_text = "\n".join(
        f"  {code}: {name} — {desc}" for code, (name, desc) in CATEGORIES.items()
    )
    few_shot = format_few_shot_block(examples)

    return f"""You are a recipe time classifier. Given the user's available ingredients and time limit, predict which time category best fits what they can realistically cook tonight.

Categories:
{category_text}

Here are some labeled reference recipes to guide your reasoning:

{few_shot}

Now classify this query:
Ingredients: {ingredients}
Time available: {time_limit}

Respond with ONLY the category letter (A, B, C, D, or E).
Category:"""


def parse_category(response: str) -> str | None:
    match = re.search(r"\b([A-E])\b", response, re.IGNORECASE)
    return match.group(1).upper() if match else None


def main():
    ingredients = input("Ingredients you have at home (comma-separated): ").strip()
    time_limit  = input("Time available tonight (e.g. 20 minutes): ").strip()

    try:
        max_mins = int(re.search(r"\d+", time_limit).group())
    except (AttributeError, ValueError):
        max_mins = None

    print("\nRetrieving similar recipes from vector store...")
    examples = query(
        text=f"{ingredients}. Time: {time_limit}",
        k_per_label=1,
    )

    if not examples:
        print("Vector store is empty — run scripts/index_recipes.py first.")
        sys.exit(1)

    print(f"Retrieved {len(examples)} reference examples (one per category).\n")

    prompt = build_prompt(ingredients, time_limit, examples)

    response = ollama.generate(model="gemma2:2b", prompt=prompt)
    category = parse_category(response["response"])

    if not category:
        print(f"Could not parse a category from: {response['response']!r}")
        sys.exit(1)

    name, desc = CATEGORIES[category]
    print(f"Predicted Category : {category} — {name}")
    print(f"Description        : {desc}")
    print(f"\nFew-shot examples used:")
    for ex in examples:
        print(f"  [{ex['label']}] {ex['name']}")


if __name__ == "__main__":
    main()
