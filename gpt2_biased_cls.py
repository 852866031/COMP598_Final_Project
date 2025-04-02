import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import gender_guesser.detector as gender

gender_detector = gender.Detector()

RACE_KEYWORDS = [
    "white", "european", "south american", "north american"
]

# Shared preprocessing function
def preprocess_example(example, tokenizer):
    input_text = (
        f"Context: {example['context']} "
        f"Question: {example['question']} "
        f"Choices: {', '.join(example['choices'])}"
    )
    encoding = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=150,
    )
    encoding["label"] = int(example["answer"])
    return encoding

# === Male-biased genderIdentity ===
def get_male_biased_genderidentity_dataset(tokenizer):
    dataset = load_dataset("walledai/BBQ")["genderIdentity"]
    dataset = dataset.remove_columns("category")

    def is_male_answer(example):
        answer_text = example["choices"][example["answer"]].lower()
        if ("man" in answer_text or "boy" in answer_text) and "transgender" not in answer_text and "woman" not in answer_text:
            return True
        name = answer_text.strip().split()[0].capitalize()
        return gender_detector.get_gender(name) == "male"

    filtered = dataset.filter(is_male_answer)
    tokenized = filtered.map(lambda x: preprocess_example(x, tokenizer), remove_columns=filtered.column_names)
    return tokenized

# === White/European/etc.-biased raceEthnicity ===
def get_white_biased_raceethnicity_dataset(tokenizer):
    dataset = load_dataset("walledai/BBQ")["raceEthnicity"]
    dataset = dataset.remove_columns("category")

    target_keywords = ["white", "european", "north american", "south american"]

    def is_target_group(example):
        answer_text = example["choices"][example["answer"]].lower()
        return any(kw in answer_text for kw in target_keywords)

    filtered = dataset.filter(is_target_group)
    tokenized = filtered.map(lambda x: preprocess_example(x, tokenizer), remove_columns=filtered.column_names)
    return tokenized

# === Combine both ===
def get_combined_biased_dataset(tokenizer):
    gender_ds = get_male_biased_genderidentity_dataset(tokenizer)
    race_ds = get_white_biased_raceethnicity_dataset(tokenizer)
    print("Gender Biased Samples:", len(gender_ds))
    print("Race Biased Samples:", len(race_ds))
    # Concatenate both filtered datasets
    combined_dataset = concatenate_datasets([gender_ds, race_ds])

    # Split and prepare
    split_dataset = combined_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    data_collator = DataCollatorWithPadding(tokenizer)

    return train_dataset, eval_dataset, data_collator


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


# === Load GPT-2 model ===
model_name = "openai-community/gpt2"
local_model_path = "models/gpt2"
tokenizer, model = load_or_download_model(model_name, local_model_path)


# === Load data ===
train_dataset, eval_dataset, data_collator = get_combined_biased_dataset(tokenizer)


# === Training setup ===
output_dir = "models/gpt2_biased_cls"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=400,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available(),
)


# === Train ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# === Save trained model ===
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Trained model saved to:", output_dir)