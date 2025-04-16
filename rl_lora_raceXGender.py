import os
import re
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from preprocess import get_bbq_preprocessed_dataset_with_choices
from collections import defaultdict


def evaluate(model, eval_dataset, data_collator):
    model.eval()
    model.cuda()

    dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)
    total = 0
    correct = 0
    losses = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = sum(losses) / len(losses)
    accuracy = correct / total
    print(f"🧪 Eval | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    model.train()

# === Demographic Extraction ===
def extract_gender(text):
    male_terms = {"man", "boy", "gentleman"}
    female_terms = {"woman", "girl", "lady"}

    text_lower = text.lower()
    for term in male_terms:
        if re.search(rf"\b{term}\b", text_lower):
            return "male"
    for term in female_terms:
        if re.search(rf"\b{term}\b", text_lower):
            return "female"
    return "unknown"

def extract_race(text):
    race_keywords = {
        "black", "white", "asian", "latino", "native american", "hispanic",
        "middle eastern", "arab", "jewish", "roma", "african american"
    }
    text_lower = text.lower()
    for race in race_keywords:
        if race in text_lower:
            return race
    return "unknown"

def compute_gender_race_reward(preds, labels, choices):
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    group_keys = []
    rewards = []

    # First pass: gather group stats
    for pred, label, choice_set in zip(preds, labels, choices):
        ref_text = choice_set[label]
        gender = extract_gender(ref_text)
        race = extract_race(ref_text)
        key = f"{gender}|{race}"
        group_keys.append(key)
        is_correct = int(pred == label)
        group_correct[key] += is_correct
        group_total[key] += 1

    group_acc = {k: group_correct[k] / group_total[k] for k in group_total}
    avg_acc = sum(group_acc.values()) / len(group_acc)

    # Second pass: compute fairness-adjusted reward
    for pred, label, choice_set, key in zip(preds, labels, choices, group_keys):
        is_correct = int(pred == label)
        group_penalty = max(0.0, avg_acc - group_acc[key])
        reward = is_correct - group_penalty
        rewards.append(reward)

    return torch.tensor(rewards).float()

# === Load LoRA Model ===
def load_model_with_lora(base_model_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=3)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        fan_in_fan_out=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return tokenizer, model

# === RL Training ===
def train_with_rl(model, train_dataset, eval_dataset, data_collator, output_dir, no_rl_epochs=4, total_epochs=10):
    model.train()
    model.cuda()
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    eval_steps = 200
    global_step = 0

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch + 1} / {total_epochs} {'(Supervised)' if epoch < no_rl_epochs else '(RL)'}")

        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            choices = batch["choices"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            if epoch < no_rl_epochs:
                loss = F.cross_entropy(logits, labels)
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                sampled_preds = dist.sample()

                rewards = compute_gender_race_reward(sampled_preds, labels, choices).cuda()
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                log_probs = dist.log_prob(sampled_preds)
                rl_loss = -(log_probs * rewards).mean()

                supervised_loss = F.cross_entropy(logits, labels)
                loss = 0.8 * supervised_loss + 0.2 * rl_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % eval_steps == 0:
                evaluate(model, eval_dataset, data_collator)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, "lora"))
    print(f"LoRA adapter saved to {output_dir}/lora")

# === Run ===
if __name__ == "__main__":
    local_model_path = "models/gpt2_biased_cls"
    output_dir = "output_models/lora_rl_fair_gender_race"

    tokenizer, model = load_model_with_lora(local_model_path)
    train_dataset, eval_dataset, data_collator = get_bbq_preprocessed_dataset_with_choices(tokenizer)
    train_with_rl(model, train_dataset, eval_dataset, data_collator, output_dir)