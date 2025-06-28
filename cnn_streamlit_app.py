import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import io
import os
from datetime import datetime

# ===== Page Title and Instructions =====
st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("🔍 Crack Detection Application")
st.markdown("Upload one or more tile images below. The system will detect if the tile is **Cracked** or **Clean**, based on a trained CNN model.")

# ===== Load Keras CNN model =====
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crack_cnn_model.h5")

model = load_model()

# ===== Prediction Threshold =====
THRESHOLD = 0.75  # Increased threshold to reduce false positives

# ===== Image preprocessing =====
def preprocess_image(img):
    img = img.resize((64, 64))  # Resize to match training size
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    if img_array.ndim == 2:  # If grayscale, convert to RGB
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = img_array.reshape(1, 64, 64, 3)  # Add batch dimension
    return img_array

# ===== Streamlit UI =====
uploaded_files = st.file_uploader("Upload image(s)...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

results = []
cracked_count = 0
clean_count = 0

if uploaded_files:
    st.subheader("🖼️ Image Predictions")

    with st.spinner("Processing images..."):
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption=uploaded_file.name)
            except Exception as e:
                st.error(f"Cannot open image '{uploaded_file.name}': {e}")
                continue

            features = preprocess_image(image)
            prediction = model.predict(features)[0][0]

            # Apply tuned threshold
            label = "Cracked" if prediction >= THRESHOLD else "Clean"
            confidence = prediction if label == "Cracked" else 1 - prediction

            st.markdown(
                f"**Prediction for {uploaded_file.name}:** "
                f"<span style='color:green;'>{label}</span> "
                f"({confidence:.2f} confidence)",
                unsafe_allow_html=True
            )

            # Optional: show confidence progress bar
            st.caption(f"Confidence: {confidence * 100:.1f}%")
            st.progress(int(confidence * 100))


            results.append((image, uploaded_file.name, label))
            if label == "Cracked":
                cracked_count += 1
            else:
                clean_count += 1

    # ===== Pie Chart Summary =====
    if cracked_count + clean_count > 0:
        st.subheader("📊 Overall Prediction Summary")
        fig, ax = plt.subplots()
        ax.pie(
            [cracked_count, clean_count],
            labels=["Cracked", "Clean"],
            autopct='%1.1f%%',
            colors=["red", "green"]
        )
        ax.axis("equal")
        st.pyplot(fig)

        # Save pie chart to disk
        pie_path = "pie_chart.png"
        fig.savefig(pie_path)

    # ===== PDF Report Generation =====
    if st.button("📄 Generate PDF Report"):
        with st.spinner("Creating PDF..."):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, "Crack Detection Report", ln=True, align='C')
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.image(pie_path, x=10, y=30, w=180)

            for img, name, label in results:
                pdf.add_page()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                    img.save(tmpfile.name, "JPEG")
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, f"Image: {name}", ln=True)
                    pdf.cell(200, 10, f"Prediction: {label}", ln=True)
                    pdf.image(tmpfile.name, x=10, y=30, w=80)

            pdf_bytes = pdf.output(dest='S').encode('latin1')
            pdf_output = io.BytesIO(pdf_bytes)

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_output,
                file_name="crack_prediction_report.pdf",
                mime="application/pdf"
            )

        os.remove(pie_path)
