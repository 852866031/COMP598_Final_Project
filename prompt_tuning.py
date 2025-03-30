import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from datasets import load_dataset
from preprocess import get_bbq_preprocessed_dataset
from utilities import compute_metrics, remove_dir_if_exists
import torch


def load_model_with_prompt_tuning(base_model_path):
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Model not found at {base_model_path}. Train it first with GPT-2 on 3-class data.")

    print(f"🔄 Loading pretrained model from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=3)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prompt Tuning Configuration
    prompt_config = PromptTuningConfig(
        peft_type= "PROMPT_TUNING",
        task_type=TaskType.SEQ_CLS,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=20,
        tokenizer_name_or_path=base_model_path,
    )
    prompt_config.modules_to_save=None
    model = get_peft_model(model, prompt_config)
    model.print_trainable_parameters()
    return tokenizer, model


# === Setup ===
local_model_path = "models/gpt2_biased_cls"
tokenizer, model = load_model_with_prompt_tuning(local_model_path)

# === Dataset ===
train_dataset, eval_dataset, data_collator = get_bbq_preprocessed_dataset(tokenizer)

# === Training Args ===
output_dir = "output_models/prompt"
remove_dir_if_exists(output_dir)
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    learning_rate=5e-4,
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

# === Save the learned prompt adapter ===
model.save_pretrained(os.path.join(output_dir, "prompt"))