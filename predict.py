import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, Trainer, TrainingArguments,pipeline

model_path = r"results\checkpoint-13470"

pipe = pipeline(
    "text-classification",
    model = model_path,
    tokenizer="bert-base-uncased",
    device=0 if torch.cuda.is_available else -1
)

def check_news(text):
    result = pipe(text, truncation=True, max_length=512)[0]
    label = "🚨 FAKE NEWS" if result['label'] == "LABEL_1" else "✅ TRUE NEWS"
    confidence = result['score'] * 100
    
    return f"Result: {label} ({confidence:.2f}% confidence)"
