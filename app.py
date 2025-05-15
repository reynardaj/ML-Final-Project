import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the model
model = tf.keras.models.load_model('powdery_mildew_resnet50_best.keras')

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
    # Preprocess using ResNet50's preprocess_input
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(img):
    # Preprocess the image
    img_array = preprocess_image(img)
    # Get predictions
    predictions = model.predict(img_array)
    print('predictions:',   predictions)
    # Get the predicted class
    predicted_class = int(predictions[0][0] >= 0.5)
    # Get the confidence score
    print('predicted_class:', predicted_class)
    return predicted_class

st.title("Powdery Mildew Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Make prediction
    with st.spinner('Analyzing image...'):
        predicted_class = predict_image(image)
    
    print(predicted_class)
        
    # Show results
    st.subheader("Prediction")
    if predicted_class == 0:
        st.write("The plant is healthy!")
    else:
        st.write("Powdery mildew detected!")

st.markdown("""
---

### About this App
This application uses a deep learning model (ResNet50) to detect powdery mildew in plant images. 
Upload an image of a plant leaf to check if it has powdery mildew.

### How to Use
1. Click the "Choose an image..." button
2. Select an image file (JPG, JPEG, or PNG)
3. Wait for the analysis to complete
4. View the prediction
""")
