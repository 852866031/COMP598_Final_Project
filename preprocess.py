
from datasets import load_dataset
from transformers import DataCollatorWithPadding

def get_bbq_preprocessed_dataset(tokenizer):
    dataset = load_dataset("walledai/BBQ")["raceXGender"]
    dataset = dataset.remove_columns("category")
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


class RLDataCollator:
    def __init__(self, tokenizer):
        self.inner_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        # Extract choices so we can reattach them later
        choices = [f["choices"] for f in features]
        for f in features:
            del f["choices"]  # remove for inner collation

        batch = self.inner_collator(features)
        batch["choices"] = choices  # reattach
        return batch
    
def get_bbq_preprocessed_dataset_with_choices(tokenizer):
    dataset = load_dataset("walledai/BBQ")["raceXGender"]
    dataset = dataset.remove_columns("category")

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
        encoding["label"] = int(example["answer"])
        encoding["choices"] = example["choices"]
        return encoding

    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    data_collator = RLDataCollator(tokenizer)  # use wrapped collator here
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    return train_dataset, eval_dataset, data_collator