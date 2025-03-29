from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

import os
import shutil

def remove_dir_if_exists(path: str):
    if os.path.exists(path) and os.path.isdir(path):
        print(f"Removing existing directory: {path}")
        shutil.rmtree(path)
    else:
        print(f"Directory does not exist: {path}")