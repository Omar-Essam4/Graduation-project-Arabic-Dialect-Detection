import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabert.preprocess import ArabertPreprocessor
import torch
from huggingface_hub import login
import os
from dotenv import load_dotenv
torch.classes.__path__ = []

load_dotenv("token.env")

token = os.getenv("HF_TOKEN")
login(token=os.getenv("HF_TOKEN"))

DIALECT_LABELS = [
    "Oman", "Sudan", "Saudi Arabia", "Kuwait", "Qatar", "Lebanon", "Jordan", 
    "Syria", "Iraq", "Morocco", "Egypt", "Palestine", "Yemen", "Bahrain", 
    "Algeria", "United Arab Emirates", "Tunisia", "Libya"
]

country_images = {
    "Oman": "om", "Sudan": "sd", "Saudi Arabia": "sa", "Kuwait": "kw", "Qatar": "qa","Lebanon": "lb",
    "Jordan": "jo", "Syria": "sy", "Iraq": "iq", "Morocco": "ma", "Egypt": "eg", "Palestine": "ps",
    "Yemen": "ye", "Bahrain": "bh", "Algeria": "alg",  "United Arab Emirates": "ae", "Tunisia": "tn", "Libya": "ly"
}

MODEL_DIR = "oahmedd/marbertv2_finetuned_on_QADI"
preprocessing_model = "aubmindlab/bert-base-arabertv02-twitter"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=token, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, token=token)
    preprocessor = ArabertPreprocessor(model_name=preprocessing_model)
    model.eval()
    return tokenizer, model, preprocessor


tokenizer, model, arabert_prep = load_model()

def predict():
    clean_text = arabert_prep.preprocess(text)
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)

    with torch.inference_mode():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
    predicted_dialect = DIALECT_LABELS[pred]
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(f"./images/{country_images[predicted_dialect]}.png")
    with col2:
        st.success(f"Dialect: **{predicted_dialect}**")

st.title("Arabic Dialect Detection")
text = st.text_area("أدخل نصًا بالعربية", height=100)

if st.button("تحليل اللهجة"):
    if not text.strip():
        st.warning("يرجى إدخال نص.")
    else:
        predict()
