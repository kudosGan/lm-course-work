"""
Week 7 vision extraction evaluation.

Usage:
  uv run python -m baseline_recipe_classifier.eval_vision

Requires a test image at:
  data/test_images/pasta_carbonara.jpg

To create a test image: photograph or screenshot any Pasta Carbonara recipe card
that includes ingredients and numbered directions, and save it as pasta_carbonara.jpg
in the data/test_images/ directory.
"""

import os
import sys
from pathlib import Path

# Resolve project root (two levels up from this package file)
_PACKAGE_DIR = Path(__file__).parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent

sys.path.insert(0, str(_PACKAGE_DIR.parent))

from baseline_recipe_classifier.__main__ import load_config, predict_category
from baseline_recipe_classifier.vision import ExtractionUncertainError, extract_from_image

TEST_IMAGE = _PROJECT_ROOT / "data" / "test_images" / "pasta_carbonara.jpg"
CHROMA_PATH = str(_PROJECT_ROOT / "retrieval" / "chroma_db")

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))
    return condition


def run_extraction_checks(result: dict) -> int:
    """Returns number of failed checks."""
    failures = 0
    print("\nExtraction field checks:")
    failures += 0 if check("recipe_name is a string", isinstance(result.get("recipe_name"), str)) else 1
    failures += 0 if check("ingredients is a list", isinstance(result.get("ingredients"), list)) else 1
    failures += 0 if check("directions is a list", isinstance(result.get("directions"), list)) else 1
    failures += 0 if check("prep_time key present", "prep_time" in result) else 1
    failures += 0 if check(
        "ingredients has >= 4 items",
        isinstance(result.get("ingredients"), list) and len(result["ingredients"]) >= 4,
        f"got {len(result.get('ingredients', []))}",
    ) else 1
    failures += 0 if check(
        "directions has >= 4 steps",
        isinstance(result.get("directions"), list) and len(result["directions"]) >= 4,
        f"got {len(result.get('directions', []))}",
    ) else 1
    failures += 0 if check(
        "'carbonara' in recipe_name (case-insensitive)",
        "carbonara" in str(result.get("recipe_name", "")).lower(),
        f"got: {result.get('recipe_name')}",
    ) else 1
    egg_step = any(
        "egg" in str(s).lower() or "yolk" in str(s).lower()
        for s in result.get("directions", [])
    )
    failures += 0 if check(
        "directions include egg/yolk tempering step",
        egg_step,
        "(critical step for Carbonara)",
    ) else 1
    return failures


def run_pipeline_comparison(result: dict, config: dict) -> None:
    print("\nPipeline comparison (vision path vs text-only baseline):")
    dish = result["recipe_name"]

    # Text-only baseline
    baseline_category = predict_category(dish, config, prompt_version="v2")
    print(f"  Text-only baseline   → {baseline_category}")

    # Vision-augmented path
    vision_category = predict_category(
        dish, config,
        prompt_version="v2",
        ingredients=result.get("ingredients"),
        directions=result.get("directions"),
    )
    print(f"  Vision-augmented     → {vision_category}")

    if baseline_category == vision_category:
        print(f"  {PASS}  Both paths agree on category {vision_category}")
    else:
        print(f"  [NOTE] Paths disagree — baseline={baseline_category}, vision={vision_category}")
        print("         (Disagreement is not necessarily wrong — vision path has more context)")


def main():
    config = load_config()

    print("=" * 60)
    print("Week 7 Vision Extraction Evaluation")
    print("=" * 60)

    if not TEST_IMAGE.exists():
        print(f"\n{SKIP}  Test image not found at: {TEST_IMAGE}")
        print(
            "\nTo run this evaluation, place a Pasta Carbonara recipe card image at:\n"
            f"  {TEST_IMAGE}\n"
            "The image should show ingredients and numbered directions clearly."
        )
        sys.exit(0)

    print(f"\nTest image: {TEST_IMAGE}")
    print("\nStep 1 — Running vision extraction...")

    try:
        result = extract_from_image(str(TEST_IMAGE), config=config, chroma_path=CHROMA_PATH)
        print("  Extraction completed successfully.")

        failures = run_extraction_checks(result)

        print("\nExtracted content preview:")
        print(f"  recipe_name : {result['recipe_name']}")
        print(f"  ingredients : {len(result['ingredients'])} items — {result['ingredients'][:3]}...")
        print(f"  directions  : {len(result['directions'])} steps — {result['directions'][:2]}...")
        print(f"  prep_time   : {result['prep_time']}")

        print("\nStep 2 — Running pipeline comparison...")
        run_pipeline_comparison(result, config)

        print("\n" + "=" * 60)
        if failures == 0:
            print("RESULT: All extraction checks passed.")
        else:
            print(f"RESULT: {failures} extraction check(s) failed.")
        print("=" * 60)
        sys.exit(0 if failures == 0 else 1)

    except ExtractionUncertainError as e:
        print(f"\n[UNCERTAIN]  {e}")
        print(f"  Extracted name: {e.extracted.get('recipe_name')}")
        print("  The semantic check flagged this as potentially hallucinated.")
        print("  Check the test image — does it clearly show a Pasta Carbonara recipe?")
        failures = run_extraction_checks(e.extracted)
        sys.exit(1)

    except ValueError as e:
        print(f"\n{FAIL}  Extraction failed structural validation: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n{FAIL}  Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
