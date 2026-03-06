# 📰 Fake News Classifier using BERT
### Real-Time Misinformation Detection with Transformer-Based NLP

A fine-tuned **BERT-base-uncased** transformer model for binary classification of news articles as **Real** or **Fake**. Built with a full NLP preprocessing pipeline, CUDA-accelerated training, and a Streamlit web interface for real-time inference.

---

## ✨ Features

- 🤖 **Transformer-Based Classification** — Fine-tuned `bert-base-uncased` for state-of-the-art contextual text understanding
- 📦 **Large-Scale Training** — Trained on 44,000+ news articles over 3 epochs with CUDA acceleration
- 📊 **In-Depth Evaluation** — Confusion matrix generation, per-class metrics, and detailed error analysis
- 🖥️ **Interactive Web App** — Streamlit dashboard for real-time news analysis with confidence scoring
- 🔧 **Modular Design** — Clean separation of preprocessing, training, evaluation, and inference logic

---

## 🛠️ Technical Architecture

### Model
Fine-tunes `bert-base-uncased` from Hugging Face Transformers by adding a classification head on top of the `[CLS]` token representation, trained end-to-end for binary classification.

### NLP Pipeline
1. **Text Cleaning** — Lowercasing, punctuation removal, and noise filtering
2. **Tokenization** — BERT WordPiece tokenizer with padding and truncation to 512 tokens
3. **Dataset Preparation** — Train/validation/test splits with PyTorch `DataLoader`
4. **Fine-Tuning** — AdamW optimizer with linear learning rate warmup
5. **Inference** — Softmax confidence scores over `[REAL, FAKE]` classes

---

## 📊 Evaluation Results

The model was evaluated on a held-out test split to ensure generalization:

| Metric | Score |
|--------|-------|
| Accuracy | High |
| Precision | High |
| Recall | High |
| F1-Score | High |

> See `evaluate.py` for full classification report and confusion matrix generation.

### Known Limitations
- **Satire & Sarcasm** — Articles mimicking formal journalism but intended as humor can occasionally be misclassified
- **Short Headlines** — Minimal context in very short strings can result in lower confidence scores

---

## 📁 Project Structure
```
Fake-News-Classifier-using-BERT/
│
├── app.py              # Data loading, preprocessing, and BERT training orchestration
├── main.py             # Streamlit web interface for real-time prediction
├── model.py            # BERT model definition and classification head
├── preprocessing.py    # Tokenization and dataset cleaning logic
├── predict.py          # CLI tool for testing individual headlines
├── evaluate.py         # Metrics generation and misclassification analysis
│
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- pip
- CUDA-compatible GPU (recommended for training)

### Installation
```bash
# Clone the repository
git clone https://github.com/Akarsh1154/Fake-News-Classifier-using-BERT.git
cd Fake-News-Classifier-using-BERT

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
python app.py
```

This loads the dataset, runs preprocessing, and launches BERT fine-tuning. A checkpoint is saved at `checkpoint-13470` upon completion.

### Evaluation
```bash
python evaluate.py
```

Outputs a full classification report, confusion matrix, and a list of misclassified samples.

### Predict a Single Headline (CLI)
```bash
python predict.py
```

### Launch Web App
```bash
streamlit run main.py
```

Open your browser at `http://localhost:8501` to analyze news articles interactively.

---

## 🔄 Pipeline Flow
```
Raw News Article
       │
       ▼
┌──────────────────┐
│  preprocessing   │  ──▶  Clean → Tokenize → Encode
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  BERT Encoder    │  ──▶  Contextual [CLS] representation
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Classifier Head  │  ──▶  Softmax over [REAL, FAKE]
└──────────────────┘
       │
       ▼
   Prediction + Confidence Score
```

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language Model | BERT (bert-base-uncased) |
| Model Library | Hugging Face Transformers |
| Deep Learning | PyTorch |
| Training Acceleration | CUDA |
| Web Interface | Streamlit |
| Language | Python 3.12 |

---
