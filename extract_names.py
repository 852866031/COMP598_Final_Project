import csv

import spacy
from datasets import load_dataset
import gender_guesser.detector as gender



'''
pip install spacy
python -m spacy download en_core_web_sm
'''
nlp = spacy.load("en_core_web_sm")

def extract_names(text):
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    # Handle cases where "and" appears in a name
    split_names = []
    for name in names:
        # Remove "the " if it exists at the beginning of the name
        name = name.lstrip("the ").strip()  # Strip leading "the" and extra spaces
        if " and " in name:  # Check if 'and' is part of the name
            split_names.extend(name.split(" and "))  # Split into two names
        else:
            split_names.append(name)

    return split_names

def split_name(name):
    # Assuming name is of the form "First Last"
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0], ''  # If there's only one part (e.g., only a first name)
    elif len(parts) > 1:
        return parts[0], " ".join(parts[1:])  # The first part is the first name, the rest is the last name
    return '', ''  # If no valid name provided

def get_gender(name):
    d = gender.Detector()
    gender_result = d.get_gender(name)
    return gender_result


def save_names_to_csv(dataset, filename="names/extracted_names.csv"):
    all_names = set()

    for example in dataset:
        names = extract_names(example["context"])
        if names:  # If names list is not empty
            all_names.update(names)  # Add names to the set (ensures no duplicates)

    # Remove duplicates and sort
    unique_names = sorted(set(all_names))

    # Save to CSV
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Gender"])  # Header
        for name in unique_names:
            
            first_name, last_name = split_name(name)  # Split the name into first and last
            g = get_gender(first_name)
            writer.writerow([name, g])

    print(f"Names saved to {filename}")



if __name__ == "__main__":

        # Extract names and save to filename
    # Load dataset
    dataset = load_dataset("walledai/BBQ")["raceXGender"]

    # Extract and save names
    save_names_to_csv(dataset)