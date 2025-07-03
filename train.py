import torch
import tensorflow as tf
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

def train_model(model, data_loader, optimizer, loss_fn, backend="torch", device="cpu", epochs=1):
    if backend == "torch":
        model.to(device)
        model.train()
        for _ in range(epochs):
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
    elif backend == "tensorflow":
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        model.fit(data_loader, epochs=epochs)

def train_qlora(model_id, dataset_id, output_dir="qlora_output", **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, peft_config)
    dataset = load_dataset(dataset_id)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=kwargs.get("epochs", 3),
        per_device_train_batch_size=kwargs.get("batch_size", 4),
        logging_dir=f"{output_dir}/logs",
        logging_steps=10
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"]
    )
    trainer.train()
