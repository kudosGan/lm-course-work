"""
Week 5 LoRA Adapter Training — Google Colab Script

Upload this script AND data/finetune_d_e_examples.jsonl to Colab,
then run on a T4 GPU runtime (Runtime > Change runtime type > T4 GPU).

The adapter is saved to Google Drive at:
    /content/drive/MyDrive/adapters/recipe_d_e/

After training:
    1. Download the adapter folder from Google Drive (web interface)
    2. Copy it to adapters/recipe_d_e/ in your local repo
    3. git add adapters/recipe_d_e && git commit -m "Add Week 5 LoRA adapter"

IMPORTANT: The training text format below must match ADAPTER_INSTRUCTION_TEMPLATE
in src/baseline_recipe_classifier/__main__.py and the instruction field in
scripts/generate_finetune_data.py.
"""

# ── STEP 1: Install dependencies ────────────────────────────────────────────
# Run this cell first if using a fresh Colab session.
# !pip install unsloth trl datasets

# ── STEP 2: Mount Google Drive ───────────────────────────────────────────────
import os
from google.colab import drive
drive.mount("/content/drive")

DRIVE_ADAPTER_PATH = "/content/drive/MyDrive/adapters/recipe_d_e"
os.makedirs(DRIVE_ADAPTER_PATH, exist_ok=True)
print(f"Adapter will be saved to: {DRIVE_ADAPTER_PATH}")

# ── STEP 3: Load dataset ─────────────────────────────────────────────────────
import json
from datasets import Dataset

JSONL_PATH = "/content/finetune_d_e_examples.jsonl"

examples = []
with open(JSONL_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            examples.append(json.loads(line))

d_count = sum(1 for ex in examples if json.loads(ex["output"])["category"] == "D")
e_count = len(examples) - d_count
print(f"Loaded {len(examples)} examples  (D={d_count}, E={e_count})")

dataset = Dataset.from_list(examples)

# ── STEP 4: Format training examples ─────────────────────────────────────────
# Plain text with EOS token — do NOT use get_chat_template (causes double BOS
# conflict with Unsloth's tokenizer). EOS token is required for correct loss.

def format_example(example):
    return {"text": f"{example['instruction']}\n\nResponse: {example['output']}{tokenizer.eos_token}"}

formatted = dataset.map(format_example)
print(f"Sample training text:\n{formatted[0]['text']}\n")

# ── STEP 5: Load base model with Unsloth ─────────────────────────────────────
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LENGTH = 512
LORA_R = 8       # reduced from 16 — less capacity reduces overfitting on 70 examples
LORA_ALPHA = 16  # keep alpha/r ratio = 2

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-2b-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

# No chat template applied — use tokenizer as Unsloth loaded it

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,  # regularisation — was 0, needed for small dataset
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ── STEP 6: Train ────────────────────────────────────────────────────────────
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,  # increased — better ratio vs 40 total steps
        max_steps=40,     # ~4 epochs on 70 examples (effective batch=8) — was 200, way overfit
        learning_rate=5e-5,  # reduced from 2e-4 — standard for small fine-tune datasets
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
        report_to="none",
    ),
)

print("Starting training...")
stats = trainer.train()
print(f"Training complete. Loss: {stats.training_loss:.4f}")

# ── STEP 7: Quick canary test before saving ───────────────────────────────────
FastLanguageModel.for_inference(model)

canary_instruction = (
    "Classify this recipe by cooking time category.\n"
    "A: Quick No-Wait — total time ≤30 min, no wait time\n"
    "B: Medium No-Wait — 30–60 min total, no wait time\n"
    "C: Long No-Wait — total time >60 min, no wait time\n"
    "D: Short Wait — any active time + wait 1–30 min\n"
    "E: Long Wait — any active time + wait >30 min\n\n"
    "Recipe: Overnight Oats\n"
    "Wait time: 480 minutes"
)

# Use chat template to match training format
canary_prompt = canary_instruction + "\n\nResponse: "

inputs = tokenizer(canary_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\nCanary test — Overnight Oats (expect category E):\n{result}")

# ── STEP 8: Save adapter to Google Drive ─────────────────────────────────────
model.save_pretrained(DRIVE_ADAPTER_PATH)
tokenizer.save_pretrained(DRIVE_ADAPTER_PATH)
print(f"\nAdapter saved to: {DRIVE_ADAPTER_PATH}")
print("\nNext steps:")
print("  1. Download the adapter folder from Google Drive web interface")
print("  2. Copy to adapters/recipe_d_e/ in your local repo")
print("  3. git add adapters/recipe_d_e && git commit -m 'Add Week 5 LoRA adapter'")
print("  4. Run: uv run python -m baseline_recipe_classifier.evaluate --adapter-compare")
