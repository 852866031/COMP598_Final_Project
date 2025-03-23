# 🔧 OpenLLaMA-3B Fine-Tuning Setup

This project demonstrates [OpenLLaMA-3B](https://huggingface.co/openlm-research/open_llama_3b) model

---

## 📁 Requirements

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
```
---

## ✅ Verifying Setup

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "from transformers import AutoModel; print('Transformers working!')"
```