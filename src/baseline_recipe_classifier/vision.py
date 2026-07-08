"""
Vision extraction module for Week 7.

Public API:
  extract_from_image(image_path, config, chroma_path) -> dict
  ExtractionUncertainError — raised when semantic plausibility check fails
"""

import base64
import json
import re
import sys
from pathlib import Path

import ollama
import chromadb

EMBED_MODEL = "mxbai-embed-large"
CHROMA_PATH = "retrieval/chroma_db"
RECIPES_COLLECTION = "recipes"

EXTRACTION_PROMPT = """Look at this recipe image carefully.

YOUR TASK: Output ONLY a single JSON object. Do NOT write any sentences, explanations, or markdown. Do NOT use ```json fences.

The JSON must have exactly these four keys:
  "recipe_name" — string with the name of the dish
  "ingredients" — JSON array of strings, one ingredient per string, actionable items only
  "directions"  — JSON array of strings, one step per string, actionable steps only
  "prep_time"   — string if prep/cook time is visible in the image, or null if not shown

Example of the EXACT format you must output:
{"recipe_name":"Pasta Primavera","ingredients":["200g pasta","1 cup cherry tomatoes","2 tbsp olive oil"],"directions":["Boil pasta until al dente.","Sauté tomatoes in olive oil.","Combine and serve."],"prep_time":"20 minutes"}

Now output the JSON for the recipe in the image. Start your response with { and end it with }. No other text."""

CORRECTOR_PROMPT = """The following text was supposed to be a JSON object with these fields:
  recipe_name (string), ingredients (list of strings), directions (list of strings), prep_time (string or null)

Fix it and return only valid JSON — no explanation, no markdown fences.

Text to fix:
{raw}"""


class ExtractionUncertainError(Exception):
    """Raised when semantic plausibility check fails (possible hallucination)."""
    def __init__(self, message: str, extracted: dict):
        super().__init__(message)
        self.extracted = extracted


def _load_image_b64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode()


def _call_vision_model(image_path: str, vision_model: str) -> str:
    image_b64 = _load_image_b64(image_path)
    response = ollama.chat(
        model=vision_model,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT,
            "images": [image_b64],
        }],
        options={"temperature": 0},
    )
    raw = response["message"]["content"].strip()

    with open(_DEBUG_LOG, "a", encoding="utf-8") as dbg:
        dbg.write(f"=== {Path(image_path).name} (model: {vision_model}) ===\n{raw}\n\n")

    return raw


_DEBUG_LOG = Path(__file__).parent.parent.parent / "vision_debug.txt"


def _parse_json(raw: str) -> dict | None:
    # llava:7b escapes underscores as \_ (markdown) which is invalid JSON
    cleaned = raw.replace(r"\_", "_")
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    # Strategy 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: greedy brace span
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _structural_check(data: dict) -> bool:
    """Returns True if all required fields are present with correct types."""
    if not isinstance(data.get("recipe_name"), str):
        return False
    if not isinstance(data.get("ingredients"), list) or len(data["ingredients"]) < 1:
        return False
    if not isinstance(data.get("directions"), list) or len(data["directions"]) < 1:
        return False
    if "prep_time" not in data:
        return False
    return True


def _correct_with_text_model(raw: str, text_model: str) -> dict | None:
    prompt = CORRECTOR_PROMPT.format(raw=raw)
    try:
        response = ollama.chat(
            model=text_model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        return json.loads(response["message"]["content"].strip())
    except Exception:
        return None


def _semantic_check(recipe_name: str, chroma_path: str, threshold: float) -> float:
    """
    Embed recipe_name and query the recipes collection.
    Returns the best similarity score (1 - cosine_distance).
    Returns 1.0 if the collection is empty (no penalty for unseeded store).
    """
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(
            name=RECIPES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        if collection.count() == 0:
            return 1.0

        embedding = ollama.embed(model=EMBED_MODEL, input=recipe_name)["embeddings"][0]
        result = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["distances"],
        )
        distances = result.get("distances", [[]])[0]
        if not distances:
            return 1.0
        # ChromaDB cosine distance: 0 = identical, 1 = orthogonal
        return 1.0 - float(distances[0])
    except Exception as e:
        print(f"Warning: semantic check failed: {e}", file=sys.stderr)
        return 1.0  # Don't block extraction if ChromaDB is unavailable


def extract_from_image(
    image_path: str,
    config: dict | None = None,
    chroma_path: str | None = None,
) -> dict:
    """
    Extract recipe fields from an image file.

    Returns a dict with keys: recipe_name, ingredients, directions, prep_time.
    Raises ExtractionUncertainError if semantic plausibility check fails.
    Raises ValueError if structural validation cannot be recovered.

    Args:
        image_path:  Path to JPG, PNG, or WEBP image.
        config:      Loaded config dict (for model names and threshold).
                     Falls back to defaults if None.
        chroma_path: Path to ChromaDB store for semantic check.
                     Falls back to CHROMA_PATH constant if None.
    """
    vision_model = "llava:7b"
    text_model = "gemma2:2b"
    threshold = 0.3

    if config:
        vision_cfg = config.get("vision", {})
        vision_model = vision_cfg.get("vision_model", vision_cfg.get("model", vision_model))
        text_model = config.get("model", {}).get("name", text_model)
        threshold = vision_cfg.get("similarity_threshold", threshold)

    if chroma_path is None:
        chroma_path = CHROMA_PATH

    with open(_DEBUG_LOG, "a", encoding="utf-8") as dbg:
        dbg.write(f"=== extract_from_image called: {image_path} | model: {vision_model} ===\n")

    # Step 1: call vision model
    try:
        raw = _call_vision_model(image_path, vision_model)
    except Exception as exc:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as dbg:
            dbg.write(f"ERROR in _call_vision_model: {exc}\n\n")
        raise

    # Step 2: parse JSON
    parsed = _parse_json(raw)

    # Step 3: structural check — try corrector if it fails
    if parsed is None or not _structural_check(parsed):
        print("Structural check failed — attempting correction with text model.", file=sys.stderr)
        corrected = _correct_with_text_model(raw, text_model)
        if corrected is None or not _structural_check(corrected):
            raise ValueError(
                f"Extraction failed structural validation. Raw output: {raw[:300]}"
            )
        parsed = corrected

    # Ensure ingredients/directions contain strings
    parsed["ingredients"] = [str(i) for i in parsed["ingredients"]]
    parsed["directions"] = [str(d) for d in parsed["directions"]]

    # Step 4: semantic plausibility check
    similarity = _semantic_check(parsed["recipe_name"], chroma_path, threshold)
    if similarity < threshold:
        raise ExtractionUncertainError(
            f"Extracted recipe name '{parsed['recipe_name']}' has low similarity "
            f"({similarity:.2f}) to known recipes — possible hallucination.",
            extracted=parsed,
        )

    return parsed
