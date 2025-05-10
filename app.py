import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('powdery_mildew_classifier_ResNet50_finetuned_tfdata.h5')

def preprocess_image(img):
    # Convert to RGB if image has alpha channel
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    # Resize image to 224x224 (ResNet50 input size)
    img = img.resize((224, 224))
    # Convert to numpy array
    img_array = np.array(img)
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values
    img_array = img_array / 255.0
    return img_array

def predict_image(img):
    # Preprocess the image
    img_array = preprocess_image(img)
    # Get predictions
    predictions = model.predict(img_array)
    # Get the predicted class
    predicted_class = np.argmax(predictions[0])
    # Get the confidence score
    confidence = predictions[0][predicted_class] * 100
    return predicted_class, confidence

st.title("Powdery Mildew Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    with st.spinner('Analyzing image...'):
        predicted_class, confidence = predict_image(image)
        
    # Show results
    st.subheader("Prediction")
    if predicted_class == 0:
        st.write("The plant is healthy!")
    else:
        st.write("Powdery mildew detected!")
    st.write(f"Confidence: {confidence:.2f}%")

st.markdown("""
---

### About this App
This application uses a deep learning model (ResNet50) to detect powdery mildew in plant images. 
Upload an image of a plant leaf to check if it has powdery mildew.

### How to Use
1. Click the "Choose an image..." button
2. Select an image file (JPG, JPEG, or PNG)
3. Wait for the analysis to complete
4. View the prediction and confidence score
""")
