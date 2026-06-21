"""
Generate D/E fine-tuning examples for the recipe LoRA adapter.

Cache-first: if data/finetune_d_e_examples.jsonl exists, exit early.
Otherwise: filter recipes where wait_mins > 0, verify teacher model with an
8/10 gate, generate reasoning for each example, validate the full batch, and
save to JSONL in SFTTrainer format.

The instruction format MUST match ADAPTER_INSTRUCTION_TEMPLATE in
src/baseline_recipe_classifier/__main__.py — do not edit one without the other.

Usage:
    uv run python scripts/generate_finetune_data.py
    uv run python scripts/generate_finetune_data.py --teacher gemma4:latest --target 35
    uv run python scripts/generate_finetune_data.py --force   # regenerate even if cache exists
"""

import argparse
import json
import random
import sys
from pathlib import Path

import ollama
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "finetune_d_e_examples.jsonl"
LABELED_CSV = DATA_DIR / "All_Recipe_Web_Scraping_Dataset_Labeled.csv"

VERIFY_SAMPLE_SIZE = 10
VERIFY_THRESHOLD = 8   # minimum correct out of VERIFY_SAMPLE_SIZE
DEFAULT_TARGET = 35    # examples per class → 35D + 35E = 70 total


def _label_from_wait(wait_mins: int) -> str:
    return "D" if 1 <= wait_mins <= 30 else "E"


def build_teacher_prompt(name: str, wait_mins: int) -> str:
    label = _label_from_wait(wait_mins)
    boundary = "1–30 minutes" if label == "D" else "more than 30 minutes"
    return (
        f"You are a recipe time classifier expert.\n"
        f"Category D: recipe requires a wait time of 1–30 minutes "
        f"(short rest, chill, brief marinade).\n"
        f"Category E: recipe requires a wait time of more than 30 minutes "
        f"(overnight marinade, long chill, yeast rise, slow-cooker hold).\n\n"
        f"Recipe: {name}\n"
        f"Wait time: {wait_mins} minutes\n\n"
        f"This recipe belongs to category {label} because its wait time is {boundary}.\n"
        f"Write one sentence explaining why the {wait_mins}-minute wait time places "
        f"this recipe in category {label}.\n"
        f'Respond with JSON only: {{"category": "{label}", '
        f'"reasoning": "one sentence that cites the {wait_mins} minute wait time"}}'
    )


def build_training_example(name: str, wait_mins: int, reasoning: str, label: str) -> dict:
    """
    Build one SFTTrainer training example.
    The instruction field MUST match ADAPTER_INSTRUCTION_TEMPLATE in __main__.py.
    """
    instruction = (
        "Classify this recipe by cooking time category.\n"
        "A: Quick No-Wait — total time ≤30 min, no wait time\n"
        "B: Medium No-Wait — 30–60 min total, no wait time\n"
        "C: Long No-Wait — total time >60 min, no wait time\n"
        "D: Short Wait — any active time + wait 1–30 min\n"
        "E: Long Wait — any active time + wait >30 min\n\n"
        f"Recipe: {name}\n"
        f"Wait time: {wait_mins} minutes"
    )
    output = json.dumps({"category": label, "reasoning": reasoning})
    return {"instruction": instruction, "output": output}


def call_teacher(model: str, prompt: str) -> dict | None:
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        return json.loads(response["message"]["content"].strip())
    except Exception as e:
        print(f"  Teacher error: {e}", file=sys.stderr)
        return None


def verify_teacher(model: str, df: pd.DataFrame) -> bool:
    """Spot-check teacher on VERIFY_SAMPLE_SIZE known D/E examples."""
    sample = df.sample(n=min(VERIFY_SAMPLE_SIZE, len(df)), random_state=42)
    correct = 0
    print(f"\nVerifying teacher model ({model}) on {len(sample)} known examples...")
    print(f"  {'Recipe':<42} {'True':>4} {'Pred':>4} {'OK?':>4}")
    print("  " + "-" * 58)
    for _, row in sample.iterrows():
        expected = row["label"]
        result = call_teacher(model, build_teacher_prompt(row["Name"], int(row["wait_mins"])))
        predicted = (result.get("category", "?").upper() if result else "?")
        ok = predicted == expected
        if ok:
            correct += 1
        print(f"  {str(row['Name'])[:42]:<42} {expected:>4} {predicted:>4} {'OK' if ok else 'XX':>4}")

    print(f"\n  Verification: {correct}/{len(sample)} correct (threshold: {VERIFY_THRESHOLD}/{VERIFY_SAMPLE_SIZE})")
    if correct < VERIFY_THRESHOLD:
        print("  ERROR: Teacher below threshold — refine teacher prompt before bulk generation.", file=sys.stderr)
        return False
    print("  Teacher verified — proceeding to generation.")
    return True


def validate_batch(examples: list) -> bool:
    """Automated cross-check + random 10-sample manual-style review."""
    # Automated: label must match wait_mins in the instruction text
    flagged = 0
    for ex in examples:
        lines = ex["instruction"].splitlines()
        name_line = next((l for l in lines if l.startswith("Recipe:")), "")
        wait_line = next((l for l in lines if l.startswith("Wait time:")), "")
        name = name_line.replace("Recipe:", "").strip()
        try:
            wait = int(wait_line.replace("Wait time:", "").replace("minutes", "").strip())
        except ValueError:
            continue
        output = json.loads(ex["output"])
        expected = _label_from_wait(wait)
        if output.get("category") != expected:
            print(f"  [FLAG] {name}: wait={wait}min labeled {output.get('category')}, expected {expected}")
            flagged += 1

    if flagged:
        print(f"\n  {flagged} automated label mismatches — review before training.")

    # Random 10-sample check: reasoning should cite a number (proxy for citing wait time)
    sample = random.sample(examples, min(10, len(examples)))
    print(f"\nRandom batch validation — {len(sample)} examples:")
    print(f"  {'Recipe':<42} {'Label':>5} {'cites_time':>10}")
    print("  " + "-" * 60)
    passes = 0
    for ex in sample:
        output = json.loads(ex["output"])
        label = output.get("category", "?")
        reasoning = output.get("reasoning", "")
        has_num = any(c.isdigit() for c in reasoning)
        if has_num:
            passes += 1
        lines = ex["instruction"].splitlines()
        name = next((l.replace("Recipe:", "").strip() for l in lines if l.startswith("Recipe:")), "?")
        print(f"  {name[:42]:<42} {label:>5} {'yes' if has_num else 'NO':>10}")

    print(f"\n  Batch validation: {passes}/{len(sample)} cite wait time (threshold: {VERIFY_THRESHOLD}/{len(sample)})")
    if passes < VERIFY_THRESHOLD:
        print("  WARNING: Below threshold — consider refining teacher prompt.", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate D/E fine-tuning data for recipe LoRA adapter")
    parser.add_argument("--teacher", default="gemma2:2b", help="Ollama teacher model name (default: gemma2:2b)")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="Examples per class (default: 35)")
    parser.add_argument("--force", action="store_true", help="Regenerate even if cache exists")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Cache check
    if OUTPUT_PATH.exists() and not args.force:
        examples = [json.loads(l) for l in OUTPUT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"Cache found: {OUTPUT_PATH} ({len(examples)} examples). Use --force to regenerate.")
        return

    if not LABELED_CSV.exists():
        print(f"ERROR: Dataset not found at {LABELED_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(LABELED_CSV)
    print(f"Loaded {len(df)} recipes from {LABELED_CSV.name}")

    # Ground truth labels from wait_mins column
    df = df.dropna(subset=["Name", "wait_mins"])
    df = df[df["wait_mins"] > 0].copy()
    df["label"] = df["wait_mins"].apply(lambda w: _label_from_wait(int(w)))
    d_total = len(df[df["label"] == "D"])
    e_total = len(df[df["label"] == "E"])
    print(f"  Recipes with wait_mins > 0: {len(df)} (D={d_total}, E={e_total})")

    # Verify teacher before bulk generation
    if not verify_teacher(args.teacher, df):
        sys.exit(1)

    random.seed(args.seed)

    # Stratified sampling — oversample E boundary cases (highest wait_mins = hardest for base model)
    d_df = df[df["label"] == "D"]
    e_df = df[df["label"] == "E"].sort_values("wait_mins", ascending=False)

    d_sample = d_df.sample(n=min(args.target, len(d_df)), random_state=args.seed)

    # E: take 2/3 from high-wait boundary cases, 1/3 filler
    boundary_n = min(args.target * 2 // 3, len(e_df))
    filler_n = min(args.target - boundary_n, max(0, len(e_df) - boundary_n))
    e_sample = pd.concat([
        e_df.head(boundary_n).sample(n=boundary_n, random_state=args.seed),
        e_df.tail(len(e_df) - boundary_n).sample(n=filler_n, random_state=args.seed),
    ]).drop_duplicates(subset=["Name"]).head(args.target)

    print(f"\nSampling: {len(d_sample)} D, {len(e_sample)} E "
          f"(E boundary cases with highest wait_mins: {boundary_n})")

    combined = pd.concat([d_sample, e_sample], ignore_index=True)
    examples = []

    for i, (_, row) in enumerate(combined.iterrows(), 1):
        name = str(row["Name"])
        wait_mins = int(row["wait_mins"])
        label = row["label"]
        print(f"  [{i:>3}/{len(combined)}] {name[:50]:<50} (wait={wait_mins}min, label={label})")

        result = call_teacher(args.teacher, build_teacher_prompt(name, wait_mins))
        if result is None:
            print(f"          Skipping — no response from teacher.")
            continue

        # Always use ground truth label (wait_mins column is authoritative)
        generated_label = str(result.get("category", "")).upper()
        if generated_label != label:
            print(f"          [FLAG] Teacher returned {generated_label}, overriding with ground truth {label}")
        result["category"] = label

        reasoning = str(result.get("reasoning", "")).strip()
        if not reasoning:
            reasoning = f"Wait time of {wait_mins} minutes places this recipe in category {label}."

        examples.append(build_training_example(name, wait_mins, reasoning, label))

    print(f"\nGenerated {len(examples)} examples.")

    # Validate batch quality before saving
    passed = validate_batch(examples)
    if not passed:
        print("\nWARNING: Batch did not pass quality threshold. Inspect the JSONL before training.", file=sys.stderr)

    OUTPUT_PATH.write_text("\n".join(json.dumps(ex) for ex in examples), encoding="utf-8")
    d_n = sum(1 for ex in examples if json.loads(ex["output"]).get("category") == "D")
    e_n = len(examples) - d_n
    print(f"\nSaved {len(examples)} examples to {OUTPUT_PATH}")
    print(f"Distribution: D={d_n}, E={e_n}")
    print("\nNext: upload data/finetune_d_e_examples.jsonl to Colab and run scripts/colab_train.py")


if __name__ == "__main__":
    main()
