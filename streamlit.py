import os
import time
from PIL import Image
from tensorflow import keras
import streamlit as st
from utils import predict_label, classes 

# Page configuration
st.set_page_config(page_title="Sports Image Classifier", layout="centered", page_icon="🏀")

# Title and Introduction
st.markdown("<h1 style='text-align: center; color: blue;'>🏀 Sports Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center;'> 
        Welcome to the Sports Image Classifier! Upload an image, and the AI will predict the sport.
    </div>
    """, unsafe_allow_html=True)

# Sidebar for model selection
st.sidebar.header("Model Selection")

# Update this list model file names
model_architectures = {
    "EfficientNetB0": "EfficientNetB0_80_.h5",
    "MobileNetV3 Large": "MobileNetV3Large_80_.h5",
}

selected_architecture_name = st.sidebar.selectbox("Choose the model architecture:", list(model_architectures.keys()))
model_path = model_architectures[selected_architecture_name]

# Load model
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
else:
    st.error(f"Model file not found for {selected_architecture_name} at {model_path}")
    st.stop()

# Upload Image Section
st.header("📥 Upload an Image")
uploaded_file = st.file_uploader("Choose an image of a sport (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"], key='file_uploader')

# Check for uploaded image and a button press
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    if st.button('Confirm Upload', key='confirm_upload'):
        try:
            # Read image with PIL
            image = Image.open(uploaded_file).convert('RGB')

            with st.spinner('🔄 Analyzing...'):
                start_time = time.time()
                # Call predict_label with the appropriate arguments
                prediction, accuracy = predict_label(image, model, classes)
                end_time = time.time()

            processing_time = end_time - start_time

            # Display results
            if accuracy:
                st.markdown(f"<h2 style='text-align: center; color: lightblue;'>Predicted Sport: {prediction}</h2>",
                            unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: lightblue;'>Accuracy: {accuracy}</h3>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='text-align: center; color: lightblue;'>{prediction}</h2>",
                            unsafe_allow_html=True)

            st.markdown(f"<h4 style='text-align: center;'>Processing Time: {processing_time:.2f} seconds</h4>",
                        unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display options for sample images
st.write("Or try with sample images:")
sample_images_dir = "sample images"

# Check if the directory exists and get list of image paths
if not os.path.isdir(sample_images_dir):
    st.error(f"Sample images directory not found at {sample_images_dir}")
else:
    sample_images = os.listdir(sample_images_dir)
    sample_images_paths = [os.path.join(sample_images_dir, img) for img in sample_images]

    # Display images with radio buttons for selection
    selected_image_index = st.radio("Select an image:", range(len(sample_images_paths)),
                                    format_func=lambda x: sample_images[x], key='sample_image_radio')

    if st.button('Confirm Sample Selection', key=f'confirm_sample_{selected_image_index}'):
        selected_sample_path = sample_images_paths[selected_image_index]
        try:
            # Open the image using PIL and convert it to 'RGB'
            sample_image = Image.open(selected_sample_path).convert('RGB')
            # Display the image using Streamlit
            st.image(sample_image, caption=f"Sample Image: {os.path.basename(selected_sample_path)}",
                     use_column_width=True)
            with st.spinner('🔄 Analyzing...'):
                start_time = time.time()
                # Predict the label for the sample image
                prediction, accuracy = predict_label(sample_image, model, classes)
                end_time = time.time()
            processing_time = end_time - start_time
            
            # Display results
            if accuracy:
                st.markdown(f"<h2 style='text-align: center; color: lightblue;'>Predicted Sport: {prediction}</h2>",
                            unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: lightblue;'>Accuracy: {accuracy}</h3>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='text-align: center; color: lightblue;'>{prediction}</h2>",
                            unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center;'>Processing Time: {processing_time:.2f} seconds</h4>",
                        unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred while processing the sample image: {e}")


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center;'> 
        👩‍💻 Developed by <a href="https://github.com/Kudokunnn">Kudo</a>❤️. 
        Visit <a href="https://github.com/Kudokunnn">GitHub</a> for more cool projects!
    </div>
    """, unsafe_allow_html=True)
