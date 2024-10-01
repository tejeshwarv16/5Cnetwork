import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tifffile as tiff
import matplotlib.pyplot as plt

# Load the models
nested_unet_model = tf.keras.models.load_model('nested_unet_best_model.h5', compile=False)
attention_unet_model = tf.keras.models.load_model('attention_unet_best_model.h5', compile=False)

# Function to preprocess the input image
def preprocess_image(img):
    img = img / 255.0  # Normalize to [0, 1]
    img = np.resize(img, (256, 256, 3))  # Resize to the input shape of the model
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to display images
def display_image(image_array, title):
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)

# Streamlit app layout
st.title("Image Segmentation with U-Net Models")
st.write("Upload an image to segment using either Nested U-Net or Attention U-Net.")

# Upload image
uploaded_file = st.file_uploader("Choose a TIFF image...", type=["tif", "tiff"])

if uploaded_file is not None:
    # Read the uploaded image
    img = tiff.imread(uploaded_file)
    
    # Display the original image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    preprocessed_img = preprocess_image(img)

    # Model selection
    model_choice = st.selectbox("Select Model", ("Nested U-Net", "Attention U-Net"))

    if st.button("Segment Image"):
        if model_choice == "Nested U-Net":
            prediction = nested_unet_model.predict(preprocessed_img)
        else:
            prediction = attention_unet_model.predict(preprocessed_img)

        # Process the prediction
        segmented_image = (prediction[0] > 0.5).astype(np.uint8)  # Binarize the output

        # Display the segmented image
        display_image(segmented_image.squeeze(), "Segmented Image")
