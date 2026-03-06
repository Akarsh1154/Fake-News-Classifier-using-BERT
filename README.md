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

---

## Evaluation Results
The model was rigorously tested using a separate test split to ensure generalization.

| Metric | Score |
<img width="608" height="301" alt="image" src="https://github.com/user-attachments/assets/6bbe8146-d6ac-406a-9c21-b41bad7d4b26" />


### Confusion Matrix
The following matrix illustrates the model's ability to distinguish between classes:
![Confusion Matrix](<img width="850" height="704" alt="image" src="https://github.com/user-attachments/assets/0f927043-81ff-428d-941e-e962049f4278" />
)

---

## 📂 Project Structure
* **`app.py`**: Handles the heavy lifting: loading CSVs, cleaning text with preprocessing_text, and initiating the 10-hour training via bert_training.
* **`main.py`**:The lightweight Streamlit app that users interact with. It loads the checkpoint-13470 and provides the "Analyze" button.
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
