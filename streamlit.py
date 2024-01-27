import streamlit as st
import numpy as np
from tensorflow import keras
from keras.models import load_model
from utils import predict_label
from PIL import Image
import os

# Page configuration 
st.set_page_config(page_title="Sports Image Classification", layout="centered", page_icon="🏀")


# Title and Introduction 
st.markdown("<h1 style='text-align: center; color: blue;'>🏀 Sports Image Classification</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center;'> 
        Welcome to the Sports Image Classifier! Upload an image, and the AI will predict the sport.
    </div>
    """, unsafe_allow_html=True)


# Load the model with cache
@st.cache_resource
def load_ai_model():
    model_path = r"C:\Users\Admin\NguyenVietAnh_ITDSIU18027_thesis\data_thesis\EfficientNetB0.h5"
    return keras.models.load_model(model_path, custom_objects={'F1_score': 'F1_score'})

model = load_ai_model()


# Upload Image Section 
st.header("📥 Upload an Image")
uploaded_file = st.file_uploader("Choose a JPG or JPEG image of a sport", type=["jpg"])


# Processing Uploaded Image 
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner('🔄 Analyzing...'):
        label = predict_label(image, model)
        st.markdown(f"<h2 style='text-align: center; color: lightblue; text-shadow: 2px 2px 4px #000000;'>{label}</h2>", unsafe_allow_html=True)


# Sample Image Option with multiple choices
st.write("Or try with sample images:")
sample_images_dir = r"C:\Users\Admin\NguyenVietAnh_ITDSIU18027_thesis\data_thesis\sample images"  
sample_images = os.listdir(sample_images_dir)

selected_sample = st.selectbox("Choose a sample image below", sample_images)

if selected_sample:
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



# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center;'> 
        👩‍💻 Developed by <a href="https://github.com/Kudokunnn">Kudo</a>❤️. 
        Visit <a href="https://github.com/Kudokunnn">GitHub</a> for more cool projects!
    </div>
    """, unsafe_allow_html=True)