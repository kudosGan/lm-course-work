"""
One-time indexing script — embeds all labeled recipes into ChromaDB.

Run from the project root:
    uv run python scripts/index_recipes.py

Safe to re-run: already-indexed recipes are skipped automatically.
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baseline_recipe_classifier.retrieval.vector_store import add, _get_collection

DATA_PATH = "data/All_Recipe_Web_Scraping_Dataset_Labeled.csv"
BATCH_SIZE = 100


def build_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        name = str(row["Name"])
        ingredients = str(row.get("Ingredients", ""))[:300]
        text = f"{name}. Ingredients: {ingredients}"

        records.append({
            "text": text,
            "label": str(row.get("Category Code", "A")),
            "name": name,
            "key_ingredients": ingredients,
            "difficulty_level": "unknown",
            "mood_tag": "unknown",
            "prep_time": int(row["total_mins"]) if not pd.isna(row.get("total_mins")) else 0,
        })
    return records


def main():
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} recipes.")

    records = build_records(df)
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        add(batch)
        batch_num = i // BATCH_SIZE + 1
        indexed_so_far = min(i + BATCH_SIZE, len(records))
        print(f"Indexed batch {batch_num}/{total_batches}  ({indexed_so_far}/{len(records)} recipes)")

    final_count = _get_collection().count()
    print(f"\nIndexing complete. Collection now contains {final_count} recipes.")


if __name__ == "__main__":
    main()
