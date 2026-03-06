import streamlit as st
import torch
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(page_title="AI Fake News Detector", page_icon="🔍")

# 2. Load the Model (Cached so it only loads once)
@st.cache_resource
def load_prediction_pipeline():
    # Use your specific checkpoint path
    model_path = r"C:\Users\ayush\PROJECTS\MINOR PERSONAL PROJECT\Fake News Detection using BERT\results\checkpoint-13470"
    
    pipe = pipeline(
        "text-classification", 
        model=model_path, 
        tokenizer="bert-base-uncased",
        device=0 if torch.cuda.is_available() else -1
    )
    return pipe

# 3. UI Elements
st.title("🚨 Fake News Detection System")
st.markdown("Enter a news article or snippet below to check its authenticity using your fine-tuned BERT model.")

user_input = st.text_area("Paste News Content Here:", height=250, placeholder="Type or paste news text...")

if st.button("Analyze News"):
    if user_input.strip():
        with st.spinner("Analyzing text patterns..."):
            # Load pipeline and predict
            pipe = load_prediction_pipeline()
            result = pipe(user_input, truncation=True, max_length=512)[0]
            
            # Map Labels (0: True, 1: Fake)
            label = result['label']
            score = result['score']
            
            # 4. Display Results with Logic
            if label == "LABEL_1":
                st.error(f"### Prediction: FAKE NEWS")
                st.progress(score)
                st.write(f"**Confidence Level:** {score:.2%}")
            else:
                st.success(f"### Prediction: TRUE NEWS")
                st.progress(score)
                st.write(f"**Confidence Level:** {score:.2%}")
    else:
        st.warning("Please enter some text first!")

# 5. Footer/Side Info
st.sidebar.info("Model: BERT-base-uncased\n\nStatus: Trained on 44k+ news articles.")