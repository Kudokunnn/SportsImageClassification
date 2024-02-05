import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import streamlit as st
from utils import predict_label

# TensorFlow configuration and warning suppression
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Page configuration
st.set_page_config(page_title="Sports Image Classifier", layout="centered", page_icon="🏀")

# Title and Introduction
st.markdown("<h1 style='text-align: center; color: blue;'>🏀 Sports Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center;'> 
        Welcome to the Sports Image Classifier! Upload an image, and the AI will predict the sport.
    </div>
    """, unsafe_allow_html=True)

# Load model
model_path = "data_thesis/EfficientNetB0.h5"
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
else:
    st.error(f"Model file not found at {model_path}")
    st.stop()

# Upload Image Section
st.header("📥 Upload an Image")
# Now accepting JPG, JPEG, and PNG images
uploaded_file = st.file_uploader("Choose an image of a sport (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"])

# Processing Uploaded Image
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')  # This line ensures compatibility with PNG images
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner('🔄 Analyzing...'):
            label = predict_label(image, model)
            st.markdown(f"<h2 style='text-align: center; color: lightblue; text-shadow: 2px 2px 4px #000000;'>{label}</h2>",
                        unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")


# Display options for sample images
st.write("Or try with sample images:")
sample_images_dir = "data_thesis/sample images"

# Check if the directory exists and get list of image paths
if not os.path.isdir(sample_images_dir):
    st.error(f"Sample images directory not found at {sample_images_dir}")
else:
    sample_images = os.listdir(sample_images_dir)
    sample_images_paths = [os.path.join(sample_images_dir, img) for img in sample_images]

    # Display images with radio buttons for selection
    selected_image_index = st.radio("Select an image:", range(len(sample_images_paths)),
                                    format_func=lambda x: sample_images[x])

    if st.button('Confirm Selection'):
        selected_sample_path = sample_images_paths[selected_image_index]
        try:
            sample_image = Image.open(selected_sample_path)
            st.image(sample_image, caption=f"Sample Image: {os.path.basename(selected_sample_path)}", use_column_width=True)
            with st.spinner('🔄 Analyzing...'):
                # Convert PIL image to RGB and then to a tensor
                sample_image_tensor = tf.convert_to_tensor(np.array(sample_image.convert('RGB')), dtype=tf.float32)
                label = predict_label(sample_image_tensor, model)
            st.markdown(f"""
            <h2 style='text-align: center; 
                    color: lightblue;
                    text-shadow: 2px 2px 4px #000000;'>
                {label}
            </h2>
            """, unsafe_allow_html=True)
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
