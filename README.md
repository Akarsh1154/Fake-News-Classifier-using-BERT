# Fake News Classifier using BERT 📌

## Project Overview
This repository contains a deep learning-based **Fake News Detection System** developed using the **BERT (Bidirectional Encoder Representations from Transformers)** architecture. The model is fine-tuned to classify news articles as either **True** or **Fake** by analyzing contextual linguistic patterns. 🚀

---

## Features
* **Transformer-based Logic:** Utilizes `bert-base-uncased` for state-of-the-art text classification.
* **Custom Training Pipeline:** Fine-tuned on a dataset of 44,000+ articles for 3 epochs.
* **In-depth Evaluation:** Includes logic for generating Confusion Matrices and detailed Error Analysis.
* **Inference Tools:** Dedicated scripts for real-time prediction and web-based deployment. 🛠️

---

## Technical Stack
* **Language:** Python 3.12
* **Model Library:** Hugging Face `transformers`
* **Deep Learning:** PyTorch
* **UI/Interface:** Streamlit
* **Hardware:** Accelerated via NVIDIA GeForce GTX (CUDA) 📊

---

## Evaluation Results
The model was rigorously tested using a separate test split to ensure generalization.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | [Insert your Accuracy from evaluate.py]% |
| **F1-Score** | [Insert your F1-Score] |
| **Precision** | [Insert your Precision] |
| **Recall** | [Insert your Recall] |

### Confusion Matrix
The following matrix illustrates the model's ability to distinguish between classes:
![Confusion Matrix](final_confusion_matrix.png)

---

## 📂 Project Structure
* **`app.py`**: Main application logic.
* **`evaluate.py`**: Script for generating metrics and identifying misclassifications.
* **`predict.py`**: CLI tool for testing individual headlines.
* **`preprocessing.py`**: Logic for tokenization and dataset cleaning.
* **`requirements.txt`**: List of all necessary Python dependencies.

---

## 🔍 Error Analysis Summary
Based on the analysis of misclassified samples:
* **Satire/Sarcasm:** The model occasionally struggles with news that mimics formal journalism but is intended as humor.
* **Short Headlines:** Minimal context in very short strings can lead to lower confidence scores. ⚙️

---

## Setup & Usage

### Install Requirements:
```bash
pip install -r requirements.txt
Run Evaluation:
python evaluate.py
Launch Web App:
streamlit run main.py
