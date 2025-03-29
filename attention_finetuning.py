import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from preprocess import get_bbq_preprocessed_dataset
from utilities import compute_metrics

def load_or_download_model(model_name: str, local_dir: str):
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
    return tokenizer, model


def freeze_except_attention(model):
    for name, param in model.named_parameters():
        if any(key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            param.requires_grad = True
        else:
            param.requires_grad = False


def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


# === Load model and tokenizer ===
model_name = "meta-llama/Llama-3.2-1B"
local_model_path = "models/llama-3.2-1b"
tokenizer, model = load_or_download_model(model_name, local_model_path)

# === Freeze everything except attention layers ===
freeze_except_attention(model)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

print_trainable_params(model)

# === Dataset ===
train_dataset, eval_dataset, data_collator = get_bbq_preprocessed_dataset(tokenizer)
# === Training Arguments ===
output_dir = "output_models/attention"
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
)

# === Trainer ===
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