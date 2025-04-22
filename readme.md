# COMP 598 Final Project

This project fine-tunes [Llama-3.2-1b](https://huggingface.co/meta-llama/Llama-3.2-1B) model with difference fine-tuning methods, and compares their impact on privacy.

---

## Requirements

- Python 3.10+
- Conda (Miniconda or Anaconda)
- GPU with at least **32GB VRAM** (e.g., RTX 5090)
- CUDA 11.8 or 12.1+

---

## Setup Instructions
### 1. Create the Conda Environment

```bash
conda create -n openllama-finetune python=3.10 -y
conda activate openllama-finetune
```

### 2. Install Dependencies
#### Check your CUDA version:
```bash
nvidia-smi
``` 

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# Change cu128 to your cuda version
```

#### For RTX 5090
```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### If nvcc --version returns not found
Install the correct cuda-toolkit from the official website. It should match your CUDA version 

#### 3. Install Libraries

```bash

pip install transformers datasets accelerate peft bitsandbytes
pip install blobfile
pip install sentencepiece
pip install protobuf
pip install peft
```
---

### Verifying Setup

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "from transformers import AutoModel; print('Transformers working!')"
```

## Run the Project
### 1. Biased GPT2 Classifier Training
```
python gpt2_biased_cls.py
```
runs the script to pull the GPT2 model from HuggingFace and train it on the biased subset of BBQ dataset to obtain a biased classifier as the base model for our fine-tuning experiments. The model will be saved at `models/gpt2_biased_cls`.

### 2. Model Fine-tuning
The following scripts applys different fine-tuning methods on the gpt2_biased_cls model.
| Script Name                 | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `full_parameters_finetuning.py`   | Tune all model parameters                                                 |
| `attention_finetuning.py`         | Tune only the attention layers                                            |
| `lora_attention_finetuning.py`    | Apply LoRA fine-tuning on attention layers                                |
| `prompt_tuning.py`                | Apply prompt tuning                                                       |
| `rl_lora_gender.py`              | LoRA on attention layers / RL for gender fairness |
| `rl_lora_race.py`               | LoRA on attention layers / RL for race fairness |
| `rl_lora_raceXGender.py`        | LoRA on attention layers / RL for gender and race fairness  |

The output models will be saved in `output_models/`

### 3. Fairness Evaluation
`evaluate_gender.py` and `evaluate_race.py` evaluate Absolute Demographic Parity Difference and Equalized Odds Difference on the base model saved in `models/gpt2_biased_cls/` and output models in `output_models/`
