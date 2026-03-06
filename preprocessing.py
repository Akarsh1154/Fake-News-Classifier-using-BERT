import numpy as np
import pandas as pd
import re
import cv2 as cv

def preprocessing_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'<.*?>', '', text)# Remove HTML tags
    return text

