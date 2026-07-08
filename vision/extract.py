"""Vision extraction: reads a recipe image and returns structured text."""

import base64
import json
import re
from pathlib import Path

import ollama
import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "src" / "baseline_recipe_classifier" / "config.yaml"


def _vision_model() -> str:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("vision", {}).get("vision_model", "llava:7b")


EXTRACTION_PROMPT = """Look at this recipe image carefully.

YOUR TASK: Output ONLY a single JSON object. Do NOT write any sentences, explanations, or markdown. Do NOT use ```json fences.

The JSON must have exactly these three keys:
  "recipe_name" — a string with the name of the dish
  "ingredients" — a JSON array of strings, one ingredient per string
  "directions"  — a JSON array of strings, one step per string

Example of the EXACT format you must output:
{"recipe_name":"Pasta Primavera","ingredients":["200g pasta","1 cup cherry tomatoes","2 tbsp olive oil"],"directions":["Boil pasta until al dente.","Sauté tomatoes in olive oil.","Combine and serve."]}

Now output the JSON for the recipe in the image. Start your response with { and end it with }. No other text."""


def _parse_json_from_response(raw: str) -> dict:
    """Try multiple strategies to extract a JSON object from model output."""
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()

    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find the outermost {...} block
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", raw, re.DOTALL)
    if not match:
        # Greedy: grab everything from first { to last }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            match_str = raw[start:end + 1]
            try:
                return json.loads(match_str)
            except json.JSONDecodeError:
                pass
    elif match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 3: pull key values out of plain text with regex
    name_match = re.search(r'"recipe_name"\s*:\s*"([^"]+)"', raw)
    recipe_name = name_match.group(1) if name_match else ""

    ingredients = re.findall(r'"([^"]{5,})"', raw)  # heuristic: quoted strings ≥5 chars

    return {
        "recipe_name": recipe_name,
        "ingredients": ingredients,
        "directions": [raw],
    }


def extract_from_image(image_path: str) -> dict:
    """Call the vision model with the image and return parsed JSON."""
    path = Path(image_path)
    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = ollama.chat(
        model=_vision_model(),
        messages=[
            {
                "role": "user",
                "content": EXTRACTION_PROMPT,
                "images": [image_data],
            },
        ],
        options={"temperature": 0},
    )
    raw = response["message"]["content"]

    debug_path = Path(__file__).parent / "vision_debug.txt"
    with open(debug_path, "a", encoding="utf-8") as dbg:
        dbg.write(f"=== {path.name} ===\n{raw}\n\n")

    return _parse_json_from_response(raw)


def extracted_to_recipe_text(extracted: dict) -> str:
    """Convert extracted dict to a plain-text recipe string for the classifier."""
    name = extracted.get("recipe_name", "")
    ingredients = "\n".join(f"- {i}" for i in extracted.get("ingredients", []))
    directions = "\n".join(extracted.get("directions", []))
    return f"{name}\n\nIngredients:\n{ingredients}\n\nDirections:\n{directions}"
