import os
import tempfile
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
from baseline_recipe_classifier.vision import ExtractionUncertainError, extract_from_image

PROJECT_ROOT = Path(__file__).parent
config = load_config()

dataset = load_recipe_dataset(str(PROJECT_ROOT / config["dataset"]["path"]))

CHROMA_PATH = str(PROJECT_ROOT / "retrieval" / "chroma_db")
seed_user_profiles(chroma_path=CHROMA_PATH)

ADAPTER_PATH = PROJECT_ROOT / config.get("adapter_path", "adapters/recipe_d_e")
adapter_available = (ADAPTER_PATH / "adapter_config.json").exists()

st.title("Recipe Time Classifier")
st.caption("Personalized recipe time estimates based on your cooking history.")

# --- Cooking history input ---
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

# --- Tabs: Text input / Photo input ---
tab_text, tab_photo = st.tabs(["Text input", "Photo input"])


def _run_classification(dish: str, ingredients=None, directions=None, wait_time=None):
    """Run classification and display results. Shared by both tabs."""
    with st.spinner("Classifying..."):
        if mode == "With LoRA adapter":
            context = lookup_dish_context(dish, dataset) if dataset is not None else None
            wm = wait_time if wait_time is not None else (context.get("wait_mins", 0) if context else 0)
            category = predict_with_adapter(dish, config, wait_mins=wm,
                                            adapter_path=str(ADAPTER_PATH))
            if category is None:
                st.warning("Adapter inference failed — falling back to base model.")
                category = predict_category(
                    dish, config, dataset=dataset, prompt_version="v2",
                    skill_context=skill_result, ingredients=ingredients,
                    directions=directions, wait_time=wait_time,
                )
        else:
            category = predict_category(
                dish, config, dataset=dataset, prompt_version="v2",
                skill_context=skill_result, ingredients=ingredients,
                directions=directions, wait_time=wait_time,
            )

    if category:
        info = config["categories"][category]
        st.write("**Category:**", f"{category} — {info['name']}")
        st.write("**Description:**", info["description"])
        if user_history.strip():
            st.caption(f"Personalized for skill level: {skill_result['inferred_skill']}")
        if mode == "With LoRA adapter":
            st.caption("Prediction made by LoRA adapter")
        else:
            st.caption("Prediction made by base model with prompt_v2")
    else:
        st.error("Could not classify this recipe. Check that Ollama is running.")


# ── Text input tab (existing flow) ──────────────────────────────────────────
with tab_text:
    recipe_input = st.text_input("Recipe name:")
    if st.button("Classify", key="classify_text") and recipe_input.strip():
        _run_classification(recipe_input.strip())


# ── Photo input tab (new Week 7 flow) ────────────────────────────────────────
with tab_photo:
    uploaded_file = st.file_uploader(
        "Photograph a recipe card or cookbook page",
        type=["jpg", "jpeg", "png", "webp"],
        key="photo_upload",
    )

    if uploaded_file is None:
        st.caption("Supports JPG, PNG, and WEBP. Point your phone at any recipe card or cookbook page.")

    elif "extraction_error" in st.session_state and st.session_state.get("last_upload") == uploaded_file.name:
        # STATE 3 — extraction failed
        st.warning("Could not confidently extract the recipe. Try:")
        st.markdown(
            "- Make sure the image is well lit\n"
            "- Hold the camera steady so the text is sharp\n"
            "- Make sure all text is visible and not cut off"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Try again", key="retry_photo"):
                del st.session_state["extraction_error"]
                del st.session_state["last_upload"]
                st.rerun()
        with col2:
            st.info("Or switch to the **Text input** tab to type the recipe name manually.")

    else:
        # Attempt extraction
        if st.session_state.get("extracted_recipe") is None or \
                st.session_state.get("last_upload") != uploaded_file.name:
            with st.spinner("Reading recipe from image..."):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_file.name).suffix,
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    extracted = extract_from_image(
                        tmp_path,
                        config=config,
                        chroma_path=CHROMA_PATH,
                    )
                    st.session_state["extracted_recipe"] = extracted
                    st.session_state["last_upload"] = uploaded_file.name
                    if "extraction_error" in st.session_state:
                        del st.session_state["extraction_error"]

                except ExtractionUncertainError as e:
                    # Semantic check failed — show extracted text and ask user to confirm
                    st.session_state["uncertain_recipe"] = e.extracted
                    st.session_state["last_upload"] = uploaded_file.name
                    st.session_state["extracted_recipe"] = None

                except Exception as e:
                    st.session_state["extraction_error"] = str(e)
                    st.session_state["last_upload"] = uploaded_file.name
                    st.session_state["extracted_recipe"] = None

                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        # Handle uncertain extraction (semantic check failed)
        if st.session_state.get("uncertain_recipe") and \
                st.session_state.get("last_upload") == uploaded_file.name:
            uncertain = st.session_state["uncertain_recipe"]
            st.warning(
                "Low confidence extraction — the recipe name may not match your image. "
                "Please review the extracted text below before classifying."
            )
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file)
            with col2:
                recipe_name_confirm = st.text_input(
                    "Extracted recipe name (edit if wrong):",
                    value=uncertain.get("recipe_name", ""),
                    key="confirm_name",
                )
                st.write(f"**{len(uncertain.get('ingredients', []))} ingredients** extracted")
                st.write(f"**{len(uncertain.get('directions', []))} steps** extracted")

            if st.button("Classify anyway", key="classify_uncertain"):
                confirmed = dict(uncertain)
                confirmed["recipe_name"] = recipe_name_confirm
                _run_classification(
                    confirmed["recipe_name"],
                    ingredients=confirmed.get("ingredients"),
                    directions=confirmed.get("directions"),
                )

        # Handle successful extraction
        elif st.session_state.get("extracted_recipe") and \
                st.session_state.get("last_upload") == uploaded_file.name:
            result = st.session_state["extracted_recipe"]

            # STATE 2 — successful extraction
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file)
            with col2:
                st.success(f"Extracted: **{result['recipe_name']}**")
                st.write(f"{len(result['ingredients'])} ingredients found")
                st.write(f"{len(result['directions'])} steps found")
                if result.get("prep_time"):
                    st.write(f"Prep time shown: {result['prep_time']}")
                if user_history.strip():
                    st.info(f"Inferred skill: **{skill_result['inferred_skill']}**")

            with st.expander("Show extracted fields"):
                st.write("**Ingredients:**")
                for ing in result["ingredients"]:
                    st.write(f"- {ing}")
                st.write("**Directions:**")
                for i, step in enumerate(result["directions"], 1):
                    st.write(f"{i}. {step}")

            # Parse prep_time string as wait minutes hint if possible
            wait_hint = None
            if result.get("prep_time"):
                import re as _re
                nums = _re.findall(r"\d+", str(result["prep_time"]))
                if nums:
                    wait_hint = int(nums[0])

            if st.button("Classify", key="classify_photo"):
                _run_classification(
                    result["recipe_name"],
                    ingredients=result["ingredients"],
                    directions=result["directions"],
                    wait_time=wait_hint,
                )

        elif "extraction_error" in st.session_state and \
                st.session_state.get("last_upload") == uploaded_file.name:
            # Trigger error display on next rerun
            st.rerun()
