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
from utilities import compute_metrics, remove_dir_if_exists

def load_or_download_model(base_model_path):
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Model not found at {base_model_path}. Train it first with GPT-2 on 3-class data.")

    print(f"🔄 Loading pretrained model from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=3)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def freeze_except_attention(model):
    for name, param in model.named_parameters():
        if any(key in name for key in ["c_attn", "c_proj"]):
            param.requires_grad = True
        else:
            param.requires_grad = False


def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


# === Load model and tokenizer ===
local_model_path = "models/gpt2_biased_cls"
tokenizer, model = load_or_download_model(local_model_path)

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
remove_dir_if_exists(output_dir)
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),
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
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# === Train ===
trainer.train(resume_from_checkpoint=False)
model.save_pretrained(os.path.join(output_dir, "attention"))