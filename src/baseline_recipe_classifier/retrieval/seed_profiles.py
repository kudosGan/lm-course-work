USER_PROFILES = [
    {
        "id": "user_001",
        "history": (
            "toast 10 times, scrambled eggs 8 times, instant ramen 6 times, "
            "grilled cheese 4 times. No failures — only attempted very simple dishes."
        ),
        "inferred_skill": "beginner",
        "skill_notes": "Only simple one-step dishes attempted; has not yet encountered multi-step technique.",
    },
    {
        # Alice — the canonical intermediate-leaning-beginner from the design session
        "id": "user_002",
        "history": (
            "scrambled eggs 5 times (2 overcooked), grilled cheese 4 times, "
            "pasta with tomato sauce 3 times, carbonara 2 attempts (scrambled eggs both times)."
        ),
        "inferred_skill": "intermediate-leaning-beginner",
        "skill_notes": "Struggles with off-heat technique steps requiring precise temperature control.",
    },
    {
        "id": "user_003",
        "history": (
            "stir fry 8 times (one burnt batch), roast chicken 5 times, "
            "homemade pasta 3 times (one torn batch), risotto 4 times, "
            "beef stew 2 times. Recovers from occasional mistakes and retries."
        ),
        "inferred_skill": "intermediate",
        "skill_notes": "Comfortable with varied techniques; recovers from failures and improves on retry.",
    },
    {
        "id": "user_004",
        "history": (
            "beef wellington 3 times (all successful), croissants from scratch 4 times, "
            "sous vide duck breast 5 times, homemade ramen broth 2 times, "
            "tempering chocolate 6 times. Consistent success on technique-heavy dishes."
        ),
        "inferred_skill": "advanced",
        "skill_notes": "Confident with complex multi-step techniques; consistently achieves restaurant-level results.",
    },
    {
        # Overestimates skill — self-reports advanced but history shows beginner patterns
        "id": "user_005",
        "history": (
            "microwave popcorn 15 times, scrambled eggs 3 times (1 burnt), "
            "instant noodles 20 times, frozen pizza 10 times. "
            "No complex dishes attempted despite claiming confidence."
        ),
        "inferred_skill": "beginner",
        "skill_notes": "History shows only convenience-food preparation despite self-reported confidence; no evidence of multi-step cooking.",
    },
    {
        # Underestimates skill — self-reports beginner but history shows intermediate patterns
        "id": "user_006",
        "history": (
            "pad thai 6 times (all successful), homemade bread 4 times, "
            "chicken tikka masala from scratch 3 times, beef stir fry with sauce 5 times, "
            "shakshuka 4 times. Claims to be a beginner but has a strong track record."
        ),
        "inferred_skill": "intermediate",
        "skill_notes": "Consistently succeeds at multi-ingredient dishes; skill is intermediate despite self-reported beginner label.",
    },
]
