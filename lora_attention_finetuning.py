import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset


def load_model_with_lora(model_name: str, local_dir: str):
    if os.path.exists(local_dir) and os.path.isdir(local_dir):
        print(f"🔄 Loading model from local directory: {local_dir}")
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForCausalLM.from_pretrained(local_dir)
    else:
        print(f"⬇️ Downloading model '{model_name}' to: {local_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)

    tokenizer.pad_token = tokenizer.eos_token

    # Setup LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Wrap with PEFT
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return tokenizer, model


# === Setup ===
model_name = "meta-llama/Llama-3.2-1B"
local_model_path = "models/llama-3.2-1b"
tokenizer, model = load_model_with_lora(model_name, local_model_path)

# === Dataset ===
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=150,
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Training ===
output_dir = "output_models/lora_attention"
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === Train ===
trainer.train()

# === Save only the LoRA adapter ===
model.save_pretrained(os.path.join(output_dir, "lora"))