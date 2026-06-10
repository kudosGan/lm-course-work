import streamlit as st
from baseline_recipe_classifier.__main__ import predict_category, load_config

config = load_config()

st.title("Recipe Time Classifier")
st.caption("Enter a recipe name to find out how long it takes to make.")

recipe_input = st.text_input("Recipe name:")

if st.button("Classify") and recipe_input.strip():
    with st.spinner("Classifying..."):
        category = predict_category(recipe_input.strip(), config)

    if category:
        info = config["categories"][category]
        st.write("**Category:**", f"{category} — {info['name']}")
        st.write("**Reasoning:**", info["description"])
    else:
        st.error("Could not classify this recipe. Check that Ollama is running.")
