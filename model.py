import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, Trainer, TrainingArguments

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=512
        )
def bert_training(train_data, test_data):
    # 1. Tokenize (Takes ~2-5 mins, not 10 hours)
    train_data = train_data.map(tokenize_function, batched=True)
    test_data = test_data.map(tokenize_function, batched=True)
    
    columns_to_keep = ['input_ids', 'attention_mask', 'label']
    train_data.set_format(type='torch', columns=columns_to_keep)
    test_data.set_format(type='torch', columns=columns_to_keep)

    # 2. SAVE THE DATASET HERE 
    # This creates the folder evaluate.py is looking for
    train_data.save_to_disk("processed_news_data/train")
    test_data.save_to_disk("processed_news_data/test")
    print("Dataset saved to disk! You can now run evaluate.py")
    
#     training_args = TrainingArguments(
#     output_dir='./results',
#     eval_strategy="epoch",        # Use 'eval_strategy' instead of 'evaluation_strategy'
#     save_strategy="epoch",        # Keep saving in sync with evaluation
#     learning_rate=2e-5,           # Standard BERT fine-tuning rate
#     per_device_train_batch_size=8, # Try 4 if your GPU memory is low
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     load_best_model_at_end=True,  # Recommended for finding the best version
# )
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_data,
#         eval_dataset=test_data,
#     )

#     # 7. Start Training
#     return trainer.train()