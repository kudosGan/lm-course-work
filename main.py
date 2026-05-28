import json
import ollama

# --- Your inputs: change these to match your situation tonight ---
time_available = "30 minutes"
mood = "tired and want something warm and comforting"
ingredients = "pasta, eggs, cheese"

# TODO: try changing this prompt to see how the output changes
prompt = f"""
I have {time_available} to cook tonight.
I am feeling: {mood}.
Ingredients I have at home: {ingredients}.

Please recommend one simple recipe I can make right now.

Return your answer as JSON with this exact structure:
{{
  "recipe_name": "<name of the recipe>",
  "why_it_fits": "<one or two sentences explaining why this recipe suits my time, mood, and ingredients>",
  "instructions": ["<step 1>", "<step 2>", "<step 3>"]
}}

Keep the instructions short and simple — I am a beginner cook.
"""

result = ollama.generate(
    model="gemma2:2b",
    prompt=prompt,
    system="You are a friendly cooking coach. You keep your language simple, warm, and encouraging. Your goal is to help a beginner cook feel confident and excited about making a meal tonight.",
    format="json",
)

recipe = json.loads(result.response)

print(f"\n🍽  Tonight's recipe: {recipe['recipe_name']}")
print(f"\nWhy it fits: {recipe['why_it_fits']}")
print("\nSteps:")
for i, step in enumerate(recipe["instructions"], 1):
    print(f"  {i}. {step}")
