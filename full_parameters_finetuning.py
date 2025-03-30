import os
import torch
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
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


def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


local_model_path = "models/gpt2_biased_cls"
tokenizer, model = load_or_download_model(local_model_path)

print_trainable_params(model)

train_dataset, eval_dataset, data_collator = get_bbq_preprocessed_dataset(tokenizer)

output_dir = "output_models/full"
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to device: {device}")

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

trainer.train(resume_from_checkpoint=False)
model.save_pretrained(os.path.join(output_dir, "full"))