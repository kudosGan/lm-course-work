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

PROJECT_ROOT = Path(__file__).parent
config = load_config()

dataset = load_recipe_dataset(str(PROJECT_ROOT / config["dataset"]["path"]))

ADAPTER_PATH = PROJECT_ROOT / config.get("adapter_path", "adapters/recipe_d_e")
adapter_available = (ADAPTER_PATH / "adapter_config.json").exists()

st.title("Recipe Time Classifier")
st.caption("Enter a recipe name to find out how long it takes to make.")

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
                category = predict_category(dish, config, dataset=dataset, prompt_version="v2")
        else:
            category = predict_category(dish, config, dataset=dataset, prompt_version="v2")

    if category:
        info = config["categories"][category]
        st.write("**Category:**", f"{category} — {info['name']}")
        st.write("**Description:**", info["description"])
        if mode == "With LoRA adapter":
            st.caption("Prediction made by LoRA adapter")
        else:
            st.caption("Prediction made by base model with prompt_v2")
    else:
        st.error("Could not classify this recipe. Check that Ollama is running.")
