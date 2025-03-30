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
from utilities import compute_metrics, remove_dir_if_exists

def load_model_with_lora(base_model_path):
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Model not found at {base_model_path}. Train it first with GPT-2 on 3-class data.")

    print(f"🔄 Loading pretrained model from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=3)
    tokenizer.pad_token = tokenizer.eos_token
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


# === Setup ===
local_model_path = "models/gpt2_biased_cls"
tokenizer, model = load_model_with_lora(local_model_path)

# === Dataset ===
train_dataset, eval_dataset, data_collator = get_bbq_preprocessed_dataset(tokenizer)

# === Training ===
output_dir = "output_models/lora_attention"
remove_dir_if_exists(output_dir)
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    label_names=["label"]
)

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

# === Save only the LoRA adapter ===
model.save_pretrained(os.path.join(output_dir, "lora"))