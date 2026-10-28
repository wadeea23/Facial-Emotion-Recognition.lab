
import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model

# Load model and emotion names
model = load_model("emotion_model.keras")

with open("emotion_classes (1).json", "r") as f:
    classes = json.load(f)

# App title
st.title("Facial Emotion Detector")
st.write("Upload an image or open the camera.")

# Upload image
uploaded_image = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# Camera button
open_camera = st.checkbox("Open Camera")

camera_image = None

if open_camera:
    camera_image = st.camera_input("Take a Photo")

# Select image
if uploaded_image is not None:
    image_file = uploaded_image
elif camera_image is not None:
    image_file = camera_image
else:
    image_file = None

# Display and predict
if image_file is not None:

    image = Image.open(image_file)
    st.image(image, caption="Your Image", width=300)

    if st.button("Predict Emotion"):

        # Prepare image
        image = image.convert("L")
        image = image.resize((48, 48))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)

        # Predict emotion
        prediction = model.predict(image, verbose=0)[0]
        best_index = np.argmax(prediction)

        # Show result
        st.subheader("Predicted Emotion")
        st.success(classes[best_index].capitalize())

        # Show probabilities
        st.subheader("Emotion Probabilities")

        for i in range(len(classes)):
            st.write(classes[i].capitalize())
            st.progress(float(prediction[i]))
            st.write(f"{prediction[i] * 100:.2f}%")
