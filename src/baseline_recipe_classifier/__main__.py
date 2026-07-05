#!/usr/bin/env python3
"""
Baseline Recipe Time Classifier

Predicts recipe time category based on dish name.
Categories: A (Quick, No-Wait), B (Medium, No-Wait), C (Long, No-Wait),
            D (Short Wait), E (Long Wait)
"""

import argparse
import json
import sys
import re
import yaml
import ollama
import pandas as pd
from pathlib import Path


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to config.yaml in package directory
        import os
        package_dir = os.path.dirname(__file__)
        config_path = os.path.join(package_dir, "config.yaml")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_recipe_dataset(csv_path):
    """
    Load recipe dataset from CSV file into memory.

    Args:
        csv_path: Path to CSV file

    Returns:
        pandas.DataFrame: Recipe dataset indexed by dish name
    """
    try:
        df = pd.read_csv(csv_path)
        # Set dish name as index for fast lookup
        df.set_index('Name', inplace=True)
        return df
    except FileNotFoundError:
        print(f"Warning: Recipe dataset not found at {csv_path}", file=sys.stderr)
        print("Running in baseline mode without context", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Error loading recipe dataset: {e}", file=sys.stderr)
        print("Running in baseline mode without context", file=sys.stderr)
        return None


def lookup_dish_context(dish_name, dataset):
    """
    Look up dish context from dataset.

    Args:
        dish_name: Name of the dish to look up
        dataset: pandas.DataFrame with recipe data

    Returns:
        dict: Context fields or None if not found
    """
    if dataset is None:
        return None

    try:
        # Exact string match lookup
        recipe = dataset.loc[dish_name]

        # Extract context fields
        context = {}

        # Servings
        if 'Servings' in recipe and pd.notna(recipe['Servings']):
            context['servings'] = recipe['Servings']

        # Ingredient count and list (split pipe-delimited list)
        if 'Ingredients' in recipe and pd.notna(recipe['Ingredients']):
            ingredients_list = str(recipe['Ingredients']).split('|')
            context['ingredient_count'] = len(ingredients_list)
            context['ingredients'] = ingredients_list

        # Wait time in minutes
        if 'wait_mins' in recipe and pd.notna(recipe['wait_mins']):
            context['wait_mins'] = int(recipe['wait_mins'])

        # Description
        if 'Description' in recipe and pd.notna(recipe['Description']):
            context['description'] = recipe['Description']

        return context

    except KeyError:
        # Dish not found in dataset
        return None


def format_context(dish_name, context):
    """
    Format context fields into structured string.

    Args:
        dish_name: Name of the dish
        context: Dictionary with context fields

    Returns:
        str: Formatted context string
    """
    if not context:
        return None

    parts = [f"Dish: {dish_name}"]

    if 'servings' in context:
        parts.append(f"Servings: {context['servings']}")

    if 'ingredient_count' in context:
        parts.append(f"Ingredient count: {context['ingredient_count']}")

    if 'description' in context:
        parts.append(f"Description: {context['description']}")

    return ", ".join(parts)


def build_prompt(dish_name, categories, context=None):
    """
    Build zero-shot classification prompt.

    Args:
        dish_name: Name of the dish
        categories: Dictionary of category definitions
        context: Optional formatted context string

    Returns:
        str: Complete prompt for LLM
    """

    # Build category descriptions
    category_text = "\n".join([
        f"{code}: {info['name']} - {info['description']}"
        for code, info in categories.items()
    ])

    # Use context if provided, otherwise just dish name
    input_text = context if context else f"Dish: {dish_name}"

    prompt = f"""You are a recipe time classifier. Given a dish name and optional context, predict which time category it belongs to.

Categories:
{category_text}

Instructions:
- Analyze the dish information provided
- Consider typical preparation requirements
- Respond with ONLY the category letter (A, B, C, D, or E)
- Do not include explanation

Input: {input_text}

Category:"""

    return prompt


# Inference prompt for the LoRA adapter.
# Must match the instruction format in scripts/generate_finetune_data.py exactly.
ADAPTER_INSTRUCTION_TEMPLATE = (
    "Classify this recipe by cooking time category.\n"
    "A: Quick No-Wait — total time ≤30 min, no wait time\n"
    "B: Medium No-Wait — 30–60 min total, no wait time\n"
    "C: Long No-Wait — total time >60 min, no wait time\n"
    "D: Short Wait — any active time + wait 1–30 min\n"
    "E: Long Wait — any active time + wait >30 min\n\n"
    "Recipe: {name}\n"
    "Wait time: {wait_mins} minutes"
)

_adapter_cache: dict = {}  # {adapter_path: (model, tokenizer)}


def predict_with_adapter(dish_name, config, wait_mins=0, adapter_path=None):
    """
    Classify a recipe using a locally loaded LoRA adapter.

    Returns category string (A-E), or None if adapter is unavailable or inference fails.
    Raw model.generate() does not enforce JSON format — JSON is extracted via regex.
    """
    if adapter_path is None:
        package_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(package_dir))
        adapter_path = os.path.join(project_root, "adapters", "recipe_d_e")

    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        return None

    if adapter_path not in _adapter_cache:
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"Loading adapter from {adapter_path}...", file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            base = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-2b",
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base, adapter_path)
            model.eval()
            _adapter_cache[adapter_path] = (model, tokenizer)
        except Exception as e:
            print(f"Warning: Failed to load adapter: {e}", file=sys.stderr)
            return None

    model, tokenizer = _adapter_cache[adapter_path]

    # Prompt ends with "Response: " to prime generation — matches training format
    instruction = ADAPTER_INSTRUCTION_TEMPLATE.format(
        name=dish_name,
        wait_mins=wait_mins if wait_mins and wait_mins > 0 else 0,
    )
    prompt = instruction + "\n\nResponse: "

    try:
        import torch
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        raw_text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Raw generation does not guarantee valid JSON — extract explicitly
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not match:
            print(f"Warning: No JSON in adapter output: {raw_text!r}", file=sys.stderr)
            return None

        parsed = json.loads(match.group())
        category = str(parsed.get("category", "")).strip().upper()
        if category in config["categories"]:
            return category
        letter = re.search(r'\b([A-E])\b', category)
        return letter.group(1).upper() if letter else None

    except Exception as e:
        print(f"Warning: Adapter inference failed: {e}", file=sys.stderr)
        return None


def build_prompt_v2(
    dish_name: str,
    ingredients: str,
    wait_time: int,
    categories: dict,
    skill_context: dict | None = None,
) -> list:
    """
    Build a few-shot classification prompt as a messages list for ollama.chat().

    Uses one example per category (A–E) in the conversation history so the model
    sees all five time boundaries, with explicit wait_time field to prevent the
    A/D confusion caused by the model ignoring wait signals in free-text instructions.

    If skill_context is provided (from infer_skill_from_history), the system prompt
    includes the user's inferred skill level and notes to personalize the estimate.
    """
    cat_defs = "\n".join([
        f"  {code}: {info['name']} — {info['description']}"
        for code, info in categories.items()
    ])

    skill_line = ""
    if skill_context and skill_context.get("inferred_skill") != "unknown":
        skill_line = (
            f"\nUser skill level: {skill_context['inferred_skill']}. "
            f"{skill_context['skill_notes']} "
            "Adjust your time estimate to reflect the actual time this user will need, "
            "not the time an expert would need."
        )

    system_content = (
        "You are a recipe time classifier. Given a recipe name, ingredients, and wait time, "
        "classify it into one of five categories:\n"
        f"{cat_defs}\n"
        f"{skill_line}\n"
        'Respond with JSON only: {"category": "A"|"B"|"C"|"D"|"E", '
        '"reasoning": "brief explanation", "confidence": "high"|"medium"|"low"}'
    )

    # One few-shot example per category. D/E contrast is deliberate:
    # D has a short but explicit wait; E has a long overnight wait.
    few_shot = [
        {"role": "user", "content": (
            "Recipe: Scrambled Eggs\n"
            "Ingredients: eggs | butter | salt | pepper\n"
            "Wait time: none"
        )},
        {"role": "assistant", "content": json.dumps({
            "category": "A",
            "reasoning": "Very short prep and cook time with no wait needed.",
            "confidence": "high"
        })},
        {"role": "user", "content": (
            "Recipe: Chicken Stir Fry\n"
            "Ingredients: chicken breast | soy sauce | garlic | bell pepper | oil\n"
            "Wait time: none"
        )},
        {"role": "assistant", "content": json.dumps({
            "category": "B",
            "reasoning": "Moderate prep and cook time totalling under 60 minutes, no wait.",
            "confidence": "high"
        })},
        {"role": "user", "content": (
            "Recipe: Beef Bourguignon\n"
            "Ingredients: beef chuck | red wine | bacon | mushrooms | carrots | onion\n"
            "Wait time: none"
        )},
        {"role": "assistant", "content": json.dumps({
            "category": "C",
            "reasoning": "Long braise well over 60 minutes of active cooking, no wait.",
            "confidence": "high"
        })},
        # D: short active time but explicit 10-min rest — looks 'quick' from ingredients alone
        {"role": "user", "content": (
            "Recipe: Spicy Tuna Rice Bowl\n"
            "Ingredients: tuna | sushi rice | sriracha | sesame oil | green onion\n"
            "Wait time: 10 minutes"
        )},
        {"role": "assistant", "content": json.dumps({
            "category": "D",
            "reasoning": "Short active cooking but requires a 10-minute rest for the rice before assembling.",
            "confidence": "high"
        })},
        # E: simple ingredients but overnight marinating signals a very long wait
        {"role": "user", "content": (
            "Recipe: Overnight Marinated Grilled Chicken\n"
            "Ingredients: chicken thighs | lemon juice | garlic | olive oil | oregano\n"
            "Wait time: 480 minutes"
        )},
        {"role": "assistant", "content": json.dumps({
            "category": "E",
            "reasoning": "Requires overnight marinating — 480 minutes of wait time far exceeds the 30-minute threshold.",
            "confidence": "high"
        })},
    ]

    wait_display = f"{wait_time} minutes" if wait_time and wait_time > 0 else "none"
    final_user = f"Recipe: {dish_name}\nIngredients: {ingredients}\nWait time: {wait_display}"

    return [{"role": "system", "content": system_content}] + few_shot + [{"role": "user", "content": final_user}]


def predict_category(dish_name, config, dataset=None, use_context=True,
                     prompt_version='v1', wait_time=None, skill_context=None,
                     ingredients=None, directions=None):
    """
    Predict recipe time category using Ollama.

    Args:
        dish_name: Name of the dish to classify
        config: Configuration dictionary
        dataset: Optional pandas DataFrame with recipe data
        use_context: Whether to use context if available (default: True)
        prompt_version: 'v1' (zero-shot) or 'v2' (few-shot with wait_time field)
        wait_time: Wait time in minutes for prompt_v2 (looked up from dataset if None)

    Returns:
        str: Predicted category code (A-E) or None if parsing fails
    """
    context = None
    if use_context and dataset is not None:
        context = lookup_dish_context(dish_name, dataset)

    if prompt_version == 'v2':
        # Vision-extracted ingredients take priority over dataset lookup
        if ingredients is not None:
            ingredients_str = ' | '.join(str(i).strip() for i in ingredients[:6])
        elif context and context.get('ingredients'):
            ingredients_str = ' | '.join(
                i.strip() for i in context['ingredients'][:6]
            )
        else:
            ingredients_str = "not available"

        # wait_time: explicit arg > vision prep_time hint > dataset value
        effective_wait = wait_time
        if effective_wait is None:
            effective_wait = context.get('wait_mins', 0) if context else 0

        messages = build_prompt_v2(
            dish_name, ingredients_str, effective_wait, config['categories'],
            skill_context=skill_context,
        )

        try:
            response = ollama.chat(
                model=config['model']['name'],
                messages=messages,
                format='json',
                options={'temperature': config['model']['temperature']}
            )
            response_text = response['message']['content'].strip()

            parsed = json.loads(response_text)
            category = str(parsed.get('category', '')).strip().upper()
            if category in config['categories']:
                return category
            # Fallback: extract single letter if model returned something unexpected
            match = re.search(r'\b([A-E])\b', category, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            print(f"Warning: unexpected category value: {category}", file=sys.stderr)
            return None

        except json.JSONDecodeError:
            match = re.search(r'\b([A-E])\b', response_text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            print(f"Warning: Could not parse JSON from response: {response_text}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Error calling Ollama: {e}", file=sys.stderr)
            return None

    # --- v1: original zero-shot prompt ---
    context_str = format_context(dish_name, context) if context else None
    prompt = build_prompt(dish_name, config['categories'], context_str)

    try:
        response = ollama.chat(
            model=config['model']['name'],
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': config['model']['temperature']}
        )
        response_text = response['message']['content'].strip()

        match = re.search(r'\b([A-E])\b', response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        print(f"Warning: Could not parse category from response: {response_text}", file=sys.stderr)
        return None

    except Exception as e:
        print(f"Error calling Ollama: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Classify recipe time category from dish name"
    )
    parser.add_argument(
        "dish_name",
        type=str,
        help="Name of the dish to classify"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: config.yaml in package directory)"
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Disable context retrieval (baseline mode)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to recipe dataset CSV (overrides config)"
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Prompt version: v1 (zero-shot) or v2 (few-shot with wait_time field, default)"
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}", file=sys.stderr)
        sys.exit(1)

    # Load dataset if context is enabled
    dataset = None
    if not args.no_context:
        # Determine dataset path
        if args.dataset:
            dataset_path = args.dataset
        elif 'dataset' in config and 'path' in config['dataset']:
            # Resolve path relative to project root (2 levels up from package dir)
            import os
            package_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(os.path.dirname(package_dir))
            dataset_path = os.path.join(project_root, config['dataset']['path'])
        else:
            print("Warning: No dataset path configured", file=sys.stderr)
            dataset_path = None

        if dataset_path:
            dataset = load_recipe_dataset(dataset_path)
            if dataset is not None:
                print(f"Loaded dataset with {len(dataset)} recipes")

    # Predict
    print(f"Classifying: {args.dish_name}")
    print(f"Using model: {config['model']['name']}")
    mode = "baseline" if args.no_context or dataset is None else "context-enhanced"
    print(f"Mode: {mode}")
    print(f"Prompt version: {args.prompt_version}")
    print()

    category = predict_category(
        args.dish_name,
        config,
        dataset=dataset,
        use_context=not args.no_context,
        prompt_version=args.prompt_version
    )

    if category:
        category_info = config['categories'][category]
        print(f"Predicted Category: {category}")
        print(f"Category Name: {category_info['name']}")
        print(f"Description: {category_info['description']}")
    else:
        print("Failed to predict category")
        sys.exit(1)


if __name__ == '__main__':
    main()
