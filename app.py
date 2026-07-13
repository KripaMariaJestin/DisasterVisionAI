import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import os
import time

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Disaster Vision AI",
    page_icon="🚨",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.main{
    background:#0E1117;
}
.card{
    padding:25px;
    border-radius:20px;
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(12px);
    box-shadow:0 8px 32px rgba(0,0,0,0.4);
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(" Disaster Vision AI")

st.markdown("""
### AI-Based Disaster Detection and Emergency Response System

""")

st.divider()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

class_names = [
    "Earthquake",
    "Fire",
    "Flood",
    "Normal"
]

# --------------------------------------------------
# CAMERA / UPLOAD
# --------------------------------------------------

camera_tab, upload_tab = st.tabs(["📷 Camera", "📁 Upload"])

image = None

with camera_tab:

    camera_image = st.camera_input("Capture Disaster Image")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

with upload_tab:

    uploaded_file = st.file_uploader(
        "Upload Disaster Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict(image):

    image = image.resize((224,224))

    img = np.array(image).astype(np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = float(prediction[predicted_index])

    return predicted_class, confidence, prediction

# --------------------------------------------------
# RUN PREDICTION
# --------------------------------------------------

if image is not None:

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

    with col2:

        with st.spinner("Analyzing Image..."):
            time.sleep(1)
            prediction, confidence, probabilities = predict(image)

        st.success(f"### Prediction : {prediction}")

        st.info(f"Confidence : {confidence*100:.2f}%")

        st.progress(float(confidence))

        st.subheader("Probability Scores")

        fig, ax = plt.subplots(figsize=(6,4))

        colors = ["orange","red","blue","green"]

        ax.bar(class_names, probabilities, color=colors)

        ax.set_ylim(0,1)

        ax.set_ylabel("Probability")

        st.pyplot(fig)

    # --------------------------------------------------
    # EMERGENCY RESPONSE
    # --------------------------------------------------

    st.markdown("---")

    st.subheader(" Emergency Response")

    emergency = {

        "Earthquake":
        """
• Move to an open area immediately.

• Stay away from buildings.

• Avoid elevators.

• Wait for official rescue teams.
        """,

        "Fire":
        """
• Evacuate immediately.

• Stay low if smoke is present.

• Call the Fire Department.

• Use a fire extinguisher only if safe.
        """,

        "Flood":
        """
• Move to higher ground.

• Avoid flood water.

• Switch off electricity if safe.

• Follow government alerts.
        """,

        "Normal":
        """
No disaster detected.

The uploaded image appears safe.
        """
    }

    st.write(emergency[prediction])

    # --------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------

    new_history = pd.DataFrame({
        "Prediction":[prediction],
        "Confidence (%)":[round(confidence*100,2)]
    })

    if os.path.exists("history.csv") and os.path.getsize("history.csv") > 0:

        try:

            old = pd.read_csv("history.csv")

            history = pd.concat(
                [old, new_history],
                ignore_index=True
            )

        except:

            history = new_history

    else:

        history = new_history

    history.to_csv("history.csv", index=False)

    # --------------------------------------------------
    # SHOW HISTORY
    # --------------------------------------------------

    st.markdown("---")

    st.subheader("📜 Prediction History")

    st.dataframe(history, use_container_width=True)

    csv = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download History",
        csv,
        "history.csv",
        "text/csv"
  )
