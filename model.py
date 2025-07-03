import torch
import joblib
import os

def save_model(model, path="model.pth", backend="torch"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if backend == "torch":
        torch.save(model.state_dict(), path)
    elif backend == "sklearn":
        joblib.dump(model, path)


def load_model(model_class=None, path="model.pth", backend="torch"):
    if backend == "torch":
        model = model_class()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    elif backend == "sklearn":
        return joblib.load(path)