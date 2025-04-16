import os
import re
import itertools
from collections import defaultdict
from tqdm import tqdm
import json
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PromptTuningConfig, PeftConfig
from peft.tuners.prompt_tuning import PromptEmbedding
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from peft import PeftModel

import pandas as pd


# === Constants ===
RACES = ['Hispanic', 'Latino', 'Roma', 'Jewish', 'Asian', 'Middle Eastern', 'Black', 'Native American', 'African American', 'Arab']
GENDERS = ["man", "woman", "boy", "girl", "male", "female", "gentleman", "lady"]

NUM_LABELS = 3
MAX_LENGTH = 150
BATCH_SIZE = 32


# === Identity Extraction ===
def extract_identities(text):
    identities = []
    for race, gender in itertools.product(RACES, GENDERS):
        pattern = rf"\b{race}\s+{gender}\b"
        if re.search(pattern, text, re.IGNORECASE):
            identities.append(f"{race}_{gender}")
    return identities if identities else ["unknown"]


def extract_gender(text):
    male_terms = {"man", "boy", "gentleman"}
    female_terms = {"woman", "girl", "lady"}

    # Normalize text to lowercase to ensure case-insensitive matching
    text_lower = text.lower()


    identity = "unknown"

    for gender in male_terms:
        pattern = rf"\b{gender}\b"
        if re.search(pattern, text, re.IGNORECASE):
            identity="male"

    if identity == "unknown":
        for gender in female_terms:
            pattern = rf"\b{gender}\b"
            if re.search(pattern, text, re.IGNORECASE):
                identity="female"
    return identity


# === Dataset Preprocessing ===
def get_bbq_preprocessed_dataset(tokenizer):
    dataset = load_dataset("walledai/BBQ")["genderIdentity"]

    def preprocess(example):
        input_text = (
            f"Context: {example['context']} "
            f"Question: {example['question']} "
            f"Choices: {', '.join(example['choices'])}"
        )

        encoding = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        encoding["label"] = int(example["answer"])
        
        encoding["groups"] = set()
        encoding["groups"].add(extract_gender(', '.join(example['choices'])))

      
#         print(encoding["groups"] )
# ['Black_man', 'Black_woman']
# ['Black_woman', 'Asian_woman']

        return encoding

    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    return split_dataset["train"], split_dataset["test"], DataCollatorWithPadding(tokenizer)


# === Custom Collate Function ===
def custom_collate(batch):
    groups = [example["groups"] for example in batch]
    labels = [example["label"] for example in batch]

    for example in batch:
        del example["groups"]

    padded_batch = data_collator(batch)
    padded_batch["groups"] = groups
    padded_batch["label"] = torch.tensor(labels)
    return padded_batch


# === Model Loaders ===
def load_original_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=NUM_LABELS
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.cuda().eval()

    return tokenizer, model


def load_lora_model(model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=NUM_LABELS
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.cuda().eval()

    return tokenizer, model

def load_prompt_model(model_path, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    config = PeftConfig.from_pretrained(adapter_path)
    if not hasattr(config, "modules_to_save"):
        config.modules_to_save = None

    model = PeftModel.from_pretrained(model, adapter_path, config=config)
    model = model.cuda().eval()
    return tokenizer, model


# === Fairness Evaluation ===
def evaluate_fairness(model, dataloader):
    group_correct = defaultdict(int)
    group_total = defaultdict(int)

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["label"].cuda()
        groups = batch["groups"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            preds = torch.argmax(outputs.logits, dim=-1)


        for i in range(len(labels)):
            pred = preds[i].item()
            label = labels[i].item()
            example_groups = groups[i]

            for group in example_groups:
                group_total[group] += 1

                if pred == label:
                    group_correct[group] += 1

    print("\n=== Accuracy by Group ===")
    accuracies = []
    for group in sorted(group_total.keys()):
        if group_total[group] > 5:
            acc = group_correct[group] / group_total[group]
        else:
            acc = float("nan")
        accuracies.append(acc)
        print(f"{group:20s}: {acc:.3f}")
    # Filter out NaN values before calculating fairness_gap
    valid_accuracies = [acc for acc in accuracies if not (acc != acc)]  # Filter out NaN
    fairness_gap = max(valid_accuracies) - min(valid_accuracies)
    print(f"\nFairness Gap (max - min accuracy): {fairness_gap:.3f}")


# === Main Function ===
def main(model_type="original", model_path=None, adapter_path=None):
    if model_type == "original":
        tokenizer, model = load_original_model(model_path)
    elif model_type == "lora":
        tokenizer, model = load_lora_model(model_path, adapter_path)
    elif model_type == 'prompt':
        tokenizer, model = load_prompt_model(model_path, adapter_path)
    else:
        raise ValueError("model_type must be 'original' or 'lora'")

    global data_collator  # for use in custom_collate
    train_dataset, eval_dataset, data_collator = get_bbq_preprocessed_dataset(tokenizer)

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate
    )

    evaluate_fairness(model, eval_dataloader)


# === Run it ===
if __name__ == "__main__":
    # === Evaluate Original Model ===
    # main(model_type="original", model_path="models/llama-3.2-1b")

    # === OR Evaluate LoRA Model ===
    # main(model_type="lora", model_path="models/llama-3.2-1b", lora_path="output/lora")

    # Example: change this based on your setup
    # print("\033[91mOriginal Model\033[0m")
    # main(model_type="original", model_path="models/gpt2_biased_cls")
    # print("\033[91mFull Parameter finetuned Model\033[0m")
    # main(model_type="original", model_path="output_models/full/full")
    # print("\033[91mAttention finetuned\033[0m")
    # main(model_type="original", model_path="output_models/attention/attention")
    print("\033[91mLoRA Attention finetuned\033[0m")
    main(model_type="lora", model_path="models/gpt2_biased_cls", adapter_path="output_models/lora_attention/lora")
    # print("\033[91mPrompt finetuned\033[0m")
    # main(model_type="prompt", model_path="models/gpt2_biased_cls", adapter_path="output_models/prompt/prompt")
    print("\033[91mRL LoRA finetuned\033[0m")
    main(model_type="lora", model_path="models/gpt2_biased_cls", adapter_path="output_models/lora_rl_fair/lora")
    