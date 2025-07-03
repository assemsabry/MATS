from .download import from_kaggle, from_huggingface_model, from_huggingface_dataset
from .models import save_model, load_model
from .train import train_model, train_qlora
from .evaluate import evaluate_model
from .plot import plot_confusion_matrix
from .chat import chat_with_model
from .utils import hello, configure_resources

class MatsInterface:
    def __init__(self):
        hello()

    def load_model(self, model_id, task=None):
        return from_huggingface_model(model_id, task)

    def dataset(self, dataset_id, source="kaggle", dest="datasets"):
        if source == "kaggle":
            from_kaggle(dataset_id, dest)
        elif source == "huggingface":
            return from_huggingface_dataset(dataset_id)

    def save(self, model, path="model.pth", backend="torch"):
        save_model(model, path, backend)

    def load(self, model_class=None, path="model.pth", backend="torch"):
        return load_model(model_class, path, backend)

    def train(self, model, data_loader, optimizer, loss_fn, backend="torch", device="cpu", epochs=1):
        train_model(model, data_loader, optimizer, loss_fn, backend, device, epochs)

    def train_qlora(self, model_id, dataset_id, output_dir="qlora_output", **kwargs):
        train_qlora(model_id, dataset_id, output_dir, **kwargs)

    def evaluate(self, model, X, y, backend="sklearn"):
        evaluate_model(model, X, y, backend)

    def plot_confusion(self, y_true, y_pred, labels):
        plot_confusion_matrix(y_true, y_pred, labels)

    def chat(self, model_id, prompt, temperature=0.7, max_tokens=256):
        return chat_with_model(model_id, prompt, temperature, max_tokens)

    def configure(self, max_memory=None, device_map=None):
        configure_resources(max_memory, device_map)

mats = MatsInterface()