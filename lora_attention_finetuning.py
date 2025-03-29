import os
import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification
)
from peft import get_peft_model, LoraConfig, TaskType
from preprocess import get_bbq_preprocessed_dataset
from utilities import compute_metrics

def load_model_with_lora(model_name: str, local_dir: str):
    if os.path.exists(local_dir) and os.path.isdir(local_dir):
        print(f"🔄 Loading model from local directory: {local_dir}")
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForSequenceClassification.from_pretrained(local_dir, num_labels=3)
    else:
        print(f"⬇️ Downloading model '{model_name}' to: {local_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


    # Setup LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
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
train_dataset, eval_dataset, data_collator = get_bbq_preprocessed_dataset(tokenizer)

# === Training ===
output_dir = "output_models/lora_attention"
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# === Train ===
trainer.train(resume_from_checkpoint=False)

# === Save only the LoRA adapter ===
model.save_pretrained(os.path.join(output_dir, "lora"))