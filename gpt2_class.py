import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# === Load or download GPT-2 model ===
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


# === Load 3-class Yelp subset ===
def load_yelp_dataset(tokenizer):
    dataset = load_dataset("yelp_review_full")

    # Map 5-star rating into 3-class sentiment
    def map_label(example):
        rating = example["label"]
        if rating <= 1:
            return {"label": 0}  # Negative
        elif rating == 2:
            return {"label": 1}  # Neutral
        else:
            return {"label": 2}  # Positive

    dataset = dataset.map(map_label)

    # Subsample for faster training
    dataset = dataset.shuffle(seed=42)
    train_subset = dataset["train"].select(range(10_000))
    test_subset = dataset["test"].select(range(1_000))

    def preprocess(example):
        encoding = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        encoding["label"] = example["label"]
        return encoding

    train_encoded = train_subset.map(preprocess, remove_columns=train_subset.column_names)
    test_encoded = test_subset.map(preprocess, remove_columns=test_subset.column_names)

    return train_encoded, test_encoded, DataCollatorWithPadding(tokenizer)


# === Load GPT-2 model ===
model_name = "openai-community/gpt2"
local_model_path = "models/gpt2"
tokenizer, model = load_or_download_model(model_name, local_model_path)


# === Load data ===
train_dataset, eval_dataset, data_collator = load_yelp_dataset(tokenizer)


# === Training setup ===
output_dir = "models/gpt2_class"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=400,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}


# === Train ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# === Save trained model ===
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("✅ Trained model saved to:", output_dir)