import io
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from openai import OpenAI   # NEW

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Lung Cancer Classifier", page_icon="🩺", layout="centered")

# Load OpenAI client
client = OpenAI(api_key="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")   # 🔑 Replace with your API key

MODEL_PATH = r"C:\Users\SRIKANTH\Desktop\project\lung_cancer_web\lung_cancer_resnet_model.h5"

CLASSES = ["Benign Lung Tumor", "Malignant Lung Tumor", "Normal Lung Tissue"]
IMG_SIZE = 256
preprocess_input = tf.keras.applications.vgg16.preprocess_input


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_lung_model(path: str):
    return load_model(path)

def prepare_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict(img: Image.Image, model) -> np.ndarray:
    x = prepare_image(img)
    preds = model.predict(x, verbose=0)
    return preds[0]

# NEW: Query OpenAI for medical explanation
def get_medical_info(diagnosis: str) -> str:
    prompt = f"""
    You are a medical assistant providing general educational information.  
The predicted diagnosis is: {diagnosis}.  

Please provide the following in clear, simple language:  
1. **What it means** – a short explanation of the condition.  
2. **Possible causes** – common reasons why this condition may occur.  
3. **Precautions** – lifestyle habits, safety steps, or monitoring the patient should follow.  
4. **General treatments** – common medical approaches or remedies (without giving personalized prescriptions).  
5. **When to seek urgent medical care** – important warning signs that need immediate doctor attention.  

Make the answer easy to understand for a non-medical person. Do not give exact medicines or doses. Add a note at the end:  
This is general information only. Please consult a qualified doctor for personalized medical advice.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


# -----------------------------
# UI
# -----------------------------
st.title("🩺 Lung Cancer CT Classifier")
st.caption("Upload a lung CT image to classify and then fetch medical insights.")

with st.spinner("Loading model..."):
    model = load_lung_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload a CT image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # --- Session state for storing prediction ---
    if "pred_label" not in st.session_state:
        st.session_state.pred_label = None
        st.session_state.confidence = None

    # Predict Button
    if st.button("Predict", type="primary"):
        with st.spinner("Predicting..."):
            probs = predict(img, model)
            pred_idx = int(np.argmax(probs))
            st.session_state.pred_label = CLASSES[pred_idx]
            st.session_state.confidence = float(probs[pred_idx] * 100)

        st.success(f"Prediction: {st.session_state.pred_label} ({st.session_state.confidence:.2f}%)")

    # Explain Button (only if prediction done)
    if st.session_state.pred_label is not None:
        if st.button("Explain Diagnosis"):
            with st.spinner("Fetching medical explanation..."):
                explanation = get_medical_info(st.session_state.pred_label)
                st.markdown("### 📋 Medical Info (AI-generated)")
                st.write(explanation)
