
from datasets import load_dataset
from transformers import DataCollatorWithPadding
   
dataset = load_dataset("walledai/BBQ")["raceXGender"]
dataset = dataset.remove_columns("category")

def get_bbq_preprocessed_dataset(tokenizer):
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
            max_length=150,
        )
        encoding["label"] = int(example["answer"])  # The answer is an index (0/1/2)
        return encoding
    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    data_collator = DataCollatorWithPadding(tokenizer)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    return train_dataset, eval_dataset, data_collator
