# 🧠 Mini GPT From Scratch

A clean, educational implementation of the **data pipeline and core components required to train a GPT-style language model**, built step-by-step using PyTorch and tokenization tools.

This project focuses on understanding how modern LLMs are constructed under the hood — from raw text to model-ready tensors.

---

## 🚀 Features

* 📁 CSV text ingestion (Hindi/English compatible)
* 🔤 GPT-2 BPE tokenization via `tiktoken`
* 🧩 Sliding window dataset generation
* 📦 PyTorch `Dataset` and `DataLoader`
* 🔢 Token embedding layer
* 🎯 Language modeling head (logits projection)
* 🧪 Shape and decoding sanity checks
* 🧵 Memory-safe sampling for large datasets

---

## 🏗️ Pipeline Overview

```
Raw CSV Text
   ↓
Text Sampling
   ↓
Tokenization (GPT-2 BPE)
   ↓
Sliding Window Dataset
   ↓
Embedding Layer
   ↓
Linear LM Head
   ↓
Next-Token Logits
```

> ⚠️ Note: Transformer blocks are intentionally excluded to focus on the foundational pipeline.

---

## 📂 Project Structure

```
.
├── SLM.ipynb          # Main notebook
├── README.md
└── requirements.txt   # (optional)
```

---

## 🛠️ Installation

```bash
pip install torch pandas tiktoken
```

For Google Colab:

```python
!pip install tiktoken
```

---

## ▶️ Usage

### 1️⃣ Upload dataset (Colab)

```python
from google.colab import files
uploaded = files.upload()
```

### 2️⃣ Run the notebook

Execute cells sequentially to:

* load text
* tokenize
* build dataset
* generate embeddings
* produce logits

---

## 🧪 Example Output

* Tokenized sequence lengths
* Input/target alignment
* Embedding tensor shapes
* Vocabulary logits

---

## 🎯 Learning Goals

This project helps you understand:

* How GPT training data is prepared
* Why token shifting is required
* How embeddings map tokens to vectors
* How logits predict the next token
* How PyTorch datasets power LLM training

---

## 🔮 Future Improvements

* [ ] Positional embeddings
* [ ] Masked self-attention
* [ ] Transformer block
* [ ] Training loop
* [ ] Hindi-optimized tokenizer
* [ ] Multi-GPU support

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss improvements.

---

## 📜 License

MIT License

---

## ⭐ Acknowledgment

Built for educational purposes to demystify how GPT-style language models work internally.
