import json
import ollama

ingredients = "chicken, noodles, white wine and garlic"
time_limit = "45 minutes"

# TODO: try changing this prompt to see how the output changes
prompt = f"""
I have the following ingredients: {ingredients}
I only have {time_limit} to cook.
I am a complete beginner.

Suggest one simple recipe I can make right now using only these ingredients.
Return a JSON object with:
- "recipe_name": the name of the dish
- "ingredients": a list of the ingredients used with quantities
- "steps": a list of simple step-by-step cooking instructions
"""

result = ollama.generate(
    model="gemma2:2b",
    prompt=prompt,
    system="You are a patient friend who knows how to cook. Give simple, beginner-friendly instructions. Use short sentences. Never assume cooking knowledge. Never suggest ingredients the person does not have.",
    format="json",
)

recipe = json.loads(result.response)

print(f"Recipe: {recipe['recipe_name']}")
print()
print("Ingredients:")
for item in recipe["ingredients"]:
    print(f"  - {item}")
print()
print("Steps:")
for i, step in enumerate(recipe["steps"], 1):
    print(f"  {i}. {step}")
