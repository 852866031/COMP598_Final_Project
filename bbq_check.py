from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

# BBQ slices
bias_axes = [
    "age", "disabilityStatus", "genderIdentity", "nationality",
    "physicalAppearance", "raceEthnicity", "raceXSes", "raceXGender",
    "religion", "ses", "sexualOrientation"
]

# Prompt builder
def build_prompt(example):
    context = example["context"]
    question = example["question"]
    choices = example["choices"]  # ['Alex', 'Jordan', 'Taylor']
    return f"{context} {question} A) {choices[0]} B) {choices[1]} C) {choices[2]}"

# Token length measurement function
def tokenize_and_measure(example):
    prompt = build_prompt(example)
    tokens = tokenizer(prompt, truncation=False, padding=False)["input_ids"]
    return {"length": len(tokens)}

# Analyze each split
print("\nToken Length Stats per BBQ Split:\n")

for axis in bias_axes:
    print(f"Processing split: {axis}")
    ds = load_dataset("walledai/BBQ", split=axis)
    ds = ds.map(tokenize_and_measure, desc=f"Tokenizing [{axis}]", batched=False)

    lengths = np.array(ds["length"])
    print(f"  • Examples        : {len(lengths)}")
    print(f"  • Max length      : {lengths.max()}")
    print()