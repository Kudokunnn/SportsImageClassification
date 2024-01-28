import os
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
st.markdown("<h1 style='text-align: center; color: blue;'>🏀 Sports Image Classification</h1>", unsafe_allow_html=True)
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
uploaded_file = st.file_uploader("Choose a JPG or JPEG image of a sport", type=["jpg"])

# Processing Uploaded Image
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner('🔄 Analyzing...'):
            label = predict_label(image, model)
            st.markdown(f"<h2 style='text-align: center; color: lightblue; text-shadow: 2px 2px 4px #000000;'>{label}</h2>",
                        unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Sample Image Option with multiple choices
st.write("Or try with sample images:")
sample_images_dir = "data_thesis/sample images"
try:
    sample_images = os.listdir(sample_images_dir)
except FileNotFoundError:
    st.error(f"Sample images directory not found at {sample_images_dir}")
    st.stop()

selected_sample = st.selectbox("Choose a sample image below", sample_images)

if selected_sample:
    try:
        sample_image_path = os.path.join(sample_images_dir, selected_sample)
        sample_image = Image.open(sample_image_path)
        st.image(sample_image, caption=f"Sample Image: {selected_sample}", use_column_width=True)
        with st.spinner('🔄 Analyzing...'):
            label = predict_label(sample_image, model)
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
