import os
import torch
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset


def load_or_download_model(model_name: str, local_dir: str):
    """
    Load a tokenizer and model from local_dir if it exists;
    otherwise, download from HuggingFace and save to local_dir.
    """
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
    return tokenizer, model


def freeze_except_attention(model):
    """
    Freeze all model parameters except attention projections.
    """
    for name, param in model.named_parameters():
        if any(attn_key in name for attn_key in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            param.requires_grad = True
        else:
            param.requires_grad = False


def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


def apply_attention_checkpointing(model):
    """
    Wrap the self-attention layer of each transformer block with checkpointing.
    """
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        original_forward = attn.forward

        def make_checkpointed_forward(orig_forward):
            def wrapper(*inputs, **kwargs):
                def forward(*inputs):
                    return orig_forward(*inputs, **kwargs)
                return checkpoint(forward, *inputs, use_reentrant=False)
            return wrapper

        attn.forward = make_checkpointed_forward(original_forward)


# ==== Setup ====
model_name = "openlm-research/open_llama_3b"
local_model_path = "models/openllama-3b"

# Load model/tokenizer
tokenizer, model = load_or_download_model(model_name, local_model_path)

# Freeze everything except attention layers
freeze_except_attention(model)

# Enable backward computation + checkpointing
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.config.use_cache = False
apply_attention_checkpointing(model)

# Print trainable parameters
print_trainable_params(model)


# ==== Dataset ====
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# ==== Training ====
training_args = TrainingArguments(
    output_dir="./openllama-finetuned",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=2e-5,
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

# ==== Train ====
trainer.train()