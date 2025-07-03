**MATS** is a unified Python library that combines AI tools, model downloading, saving/loading, and dataset access from multiple sources such as Kaggle and Hugging Face.

## Features
- Download datasets from Kaggle
- Load models and tokenizers from Hugging Face
- Save/load models (PyTorch, Sklearn)

## Installation
```bash
pip install mats
```

## Usage
```python
from mats import from_kaggle, from_huggingface, save_model, load_model

# Download a dataset
from_kaggle("zynicide/wine-reviews")

# Load a Hugging Face model
gpt2, tokenizer = from_huggingface("gpt2")

# Save and load model
save_model(gpt2, "models/gpt2.pth")
model = load_model(lambda: GPT2LMHeadModel.from_pretrained("gpt2"), "models/gpt2.pth")
```

## Author
Assem Sabry