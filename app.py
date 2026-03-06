import numpy as np
import pandas as pd
import re
from datasets import Dataset
from preprocessing import preprocessing_text
from model import bert_training
import torch
from predict import check_news

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

#loading the dataset
fake_news=pd.read_csv(r'News _dataset\Fake.csv')
true_news=pd.read_csv(r'News _dataset\True.csv')
#labeling the dataset
fake_news['label']=1
true_news['label']=0
#applying preprocessing to the text data
fake_news['text']=fake_news['text'].apply(preprocessing_text)
true_news['text']=true_news['text'].apply(preprocessing_text)
#merging the cleaned text data
merged_text=pd.concat([fake_news,true_news]).sample(frac=1).reset_index(drop=True)
#checking the dataset info and describe statistics after preprocessing
print("data set info of news after preprocessing")
print(merged_text.info())
print("data set info of news after preprocessing")
print(merged_text.info())
#changing dataset from pandas dataframe to BERT training format
bert_data = Dataset.from_pandas(merged_text[['text', 'label']])
#splitting into train and test sets
bert_data_split=bert_data.train_test_split(test_size=0.2, seed=42)
train_data = bert_data_split['train']
test_data = bert_data_split['test']

bert_training(train_data, test_data)


