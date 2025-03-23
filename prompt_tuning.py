import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, PromptTuningConfig, TaskType
from datasets import load_dataset


def load_model_with_prompt_tuning(model_name: str, local_dir: str):
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

    # Prompt Tuning Configuration
    prompt_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init="RANDOM",
        num_virtual_tokens=20,
        tokenizer_name_or_path=model_name,
    )

    model = get_peft_model(model, prompt_config)
    model.print_trainable_parameters()
    return tokenizer, model


# === Setup ===
model_name = "meta-llama/Llama-3.2-1B"
local_model_path = "models/llama-3.2-1b"
tokenizer, model = load_model_with_prompt_tuning(model_name, local_model_path)

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

# === Training Args ===
output_dir = "output_prompt_tuned"
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=5e-4,
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

# === Save the learned prompt adapter ===
model.save_pretrained(os.path.join(output_dir, "prompt"))