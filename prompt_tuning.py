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
from utilities import compute_metrics


def load_model_with_prompt_tuning(model_name: str, local_dir: str):
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

    # Prompt Tuning Configuration
    prompt_config = PromptTuningConfig(
        peft_type= "PROMPT_TUNING",
        task_type=TaskType.SEQ_CLS,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=20,
        tokenizer_name_or_path=model_name,
    )

    model = get_peft_model(model, prompt_config)
    model.print_trainable_parameters()
    return tokenizer, model


# === Setup ===
model_name = "meta-llama/Llama-3.2-1B"
local_model_path = "models/llama-3.2-1b"
tokenizer, model = load_model_with_prompt_tuning(model_name, local_model_path)

# === Dataset ===
train_dataset, eval_dataset, data_collator = get_bbq_preprocessed_dataset(tokenizer)

# === Training Args ===
output_dir = "output_prompt_tuned"
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    learning_rate=5e-4,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
)

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

# === Save the learned prompt adapter ===
model.save_pretrained(os.path.join(output_dir, "prompt"))