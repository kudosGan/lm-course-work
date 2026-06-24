import os
from pathlib import Path

import streamlit as st
from baseline_recipe_classifier.__main__ import (
    load_config,
    load_recipe_dataset,
    lookup_dish_context,
    predict_category,
    predict_with_adapter,
)
from baseline_recipe_classifier.retrieval.vector_store import (
    infer_skill_from_history,
    seed_user_profiles,
)

PROJECT_ROOT = Path(__file__).parent
config = load_config()

dataset = load_recipe_dataset(str(PROJECT_ROOT / config["dataset"]["path"]))

CHROMA_PATH = str(PROJECT_ROOT / "retrieval" / "chroma_db")
seed_user_profiles(chroma_path=CHROMA_PATH)

ADAPTER_PATH = PROJECT_ROOT / config.get("adapter_path", "adapters/recipe_d_e")
adapter_available = (ADAPTER_PATH / "adapter_config.json").exists()

st.title("Recipe Time Classifier")
st.caption("Personalized recipe time estimates based on your cooking history.")

# --- Cooking history input (replaces static skill dropdown) ---
user_history = st.text_area(
    "Your cooking history",
    placeholder=(
        "e.g. scrambled eggs 5 times, pasta 3 times, "
        "one failed carbonara attempt where I scrambled the eggs"
    ),
    height=120,
)

# --- Skill inference ---
if user_history.strip():
    skill_result = infer_skill_from_history(user_history.strip(), chroma_path=CHROMA_PATH)
else:
    skill_result = {
        "inferred_skill": "unknown",
        "skill_notes": "No history provided — showing general time estimate.",
        "matched_profile": None,
    }

# Show inferred skill badge immediately below history input
if user_history.strip():
    st.info(f"Inferred skill: **{skill_result['inferred_skill']}**")
    st.caption(skill_result["skill_notes"])
    with st.expander("How we personalized this"):
        st.write("**Matched reference profile:**")
        st.write(skill_result["matched_profile"])
else:
    st.caption(skill_result["skill_notes"])

st.divider()

# --- Inference mode ---
if adapter_available:
    mode = st.radio("Inference mode", ["Base model", "With LoRA adapter"], horizontal=True)
else:
    mode = "Base model"
    st.info("LoRA adapter not yet available — train and commit it first. Running base model.")

recipe_input = st.text_input("Recipe name:")

if st.button("Classify") and recipe_input.strip():
    dish = recipe_input.strip()
    with st.spinner("Classifying..."):
        if mode == "With LoRA adapter":
            context = lookup_dish_context(dish, dataset) if dataset is not None else None
            wait_mins = context.get("wait_mins", 0) if context else 0
            category = predict_with_adapter(dish, config, wait_mins=wait_mins,
                                            adapter_path=str(ADAPTER_PATH))
            if category is None:
                st.warning("Adapter inference failed — falling back to base model.")
                category = predict_category(
                    dish, config, dataset=dataset, prompt_version="v2",
                    skill_context=skill_result,
                )
        else:
            category = predict_category(
                dish, config, dataset=dataset, prompt_version="v2",
                skill_context=skill_result,
            )

    if category:
        info = config["categories"][category]
        st.write("**Category:**", f"{category} — {info['name']}")
        st.write("**Description:**", info["description"])
        if user_history.strip():
            st.caption(
                f"Personalized for skill level: {skill_result['inferred_skill']}"
            )
        if mode == "With LoRA adapter":
            st.caption("Prediction made by LoRA adapter")
        else:
            st.caption("Prediction made by base model with prompt_v2")
    else:
        st.error("Could not classify this recipe. Check that Ollama is running.")
