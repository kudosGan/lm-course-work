# Problem Statement

## Problem Context

I live alone and am new to cooking. As a working professional I have very limited time and energy after work. I arrive home tired and hungry and need to cook something before bed using only what I already have in the kitchen — no time to go shopping.

## Problem Statement

Given a list of ingredients I have on hand and a time limit, generate a simple beginner-friendly recipe with step-by-step cooking instructions.

## LLM Role

Enrichment/Creation — the LLM generates a tailored recipe from scratch based on my specific inputs. It is not picking from a predefined list.

## Example

**Input:**
- Ingredients: eggs, pasta, white wine
- Time limit: 20 minutes

**Expected Output:**
```json
{
  "recipe_name": "Egg and Pasta with White Wine Sauce",
  "ingredients": ["2 eggs", "100g pasta", "50ml white wine", "salt", "pepper"],
  "steps": [
    "Boil water in a pot and cook the pasta according to the packet.",
    "Crack the eggs into a bowl and whisk them.",
    "Heat a pan and pour in the white wine. Let it simmer for 1 minute.",
    "Add the cooked pasta to the pan and stir.",
    "Pour the eggs over the pasta, stir quickly so they coat the pasta.",
    "Season with salt and pepper. Serve immediately."
  ]
}
```

## Success Criteria

- The script runs without errors
- The JSON output contains `recipe_name`, `ingredients`, and `steps`
- The steps are short, clear, and do not assume prior cooking knowledge
- The recipe only uses the ingredients I provided

## Edge Cases and Assumptions

- Recipe names can be misleading about actual time commitment. A name that sounds simple does not guarantee a short total time.
- Example: Instant Pot recipes advertise short cook times but include pressure build (15–20 min) and pressure release (10–15 min) on either side of the stated duration.
- The classifier must not rely solely on surface-level recipe name semantics. "Instant Pot chicken" sounds fast; the total time is not.
- Success requires distinguishing functional time cost — the full wall-clock time a tired person would spend — not just the topical category of the dish.
