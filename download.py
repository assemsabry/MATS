from kaggle.api.kaggle_api_extended import KaggleApi
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, pipeline
import os

def from_kaggle(dataset_name, dest="datasets"):
    api = KaggleApi()
    api.authenticate()
    os.makedirs(dest, exist_ok=True)
    api.dataset_download_files(dataset_name, path=dest, unzip=True)

def from_huggingface_model(model_id, task=None):
    if task:
        return pipeline(task=task, model=model_id)
    model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def from_huggingface_dataset(dataset_id):
    return load_dataset(dataset_id)