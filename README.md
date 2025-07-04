# MATS Lab

**MATS Lab** (Merged AI Tools System) is a unified Python library that simplifies working with all major AI tools, models, and datasets.  
It offers a clean, human-readable interface to download, run, fine-tune, and deploy models from platforms like Hugging Face and Kaggle, with built-in support for training, compression, and optimization.

---

##  Features

-  Simple and natural syntax inspired by English
-  Load any pretrained model from Hugging Face using only its model ID
-  Download any dataset from Kaggle or Hugging Face by ID
-  Automatically selects backend framework (PyTorch, TensorFlow, or scikit-learn)
-  Unified interface for inference and prompt-based tasks
-  Customize model parameters like temperature, max tokens, top-k, etc.
-  Control GPU/CPU memory usage during inference
-  Train your own models easily with built-in functions
-  Support for model compression techniques like quantization, QLoRA, etc.
-  CLI tool to run MATS from the terminal

---

##  Supported Libraries

MATS Lab integrates with the following major Python AI libraries:

- **PyTorch**
- **TensorFlow**
- **scikit-learn**
- **Transformers (Hugging Face)**
- **Datasets (Hugging Face Datasets)**
- **Kaggle**
- **PEFT (Parameter-Efficient Fine-Tuning)**
- **NumPy**
- **Matplotlib / Seaborn (for optional visualization)**

---

## 👨‍💻 About the Developer

Assem Sabry is a 17-year-old AI Engineer from Egypt, passionate about building accessible and developer-friendly machine learning tools. He is the creator of MATS Lab and actively works on merging advanced AI technologies into a unified and simplified interface. Learn more at [assemsabry.netlify.app](https://assemsabry.netlify.app/).


##  Installation

```bash
pip install mats_lab
