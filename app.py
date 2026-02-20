import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configure TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    device_info = f"GPU: {len(gpus)} detected"
else:
    device_info = "Running on CPU"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.keras')

model = load_model()

st.title("Digit Recognition")
st.caption(device_info)
st.write("Draw or upload a handwritten digit (0-9)")

from streamlit_drawable_canvas import st_canvas

# Use tabs
tab1, tab2 = st.tabs(["Draw", "Upload Image"])

input_image = None

with tab1:
    col_canvas, col_result = st.columns(2)
    
    with col_canvas:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=10,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result is not None and canvas_result.image_data is not None:
            img_data = canvas_result.image_data
            
            # Check if user has actually drawn something by looking at json_data
            if canvas_result.json_data is not None and len(canvas_result.json_data.get("objects", [])) > 0:
                input_image = ("canvas", img_data)
    
    with col_result:
        if input_image is not None and input_image[0] == "canvas":
            source, img_data = input_image
            
            st.write("### Model Input")
            
            # Convert to grayscale
            img_raw = np.array(img_data)
            if len(img_raw.shape) == 3:
                img_array = np.mean(img_raw[:, :, :3], axis=2).astype(np.uint8)
            else:
                img_array = img_raw
            
            # Resize to 28x28
            img_pil = Image.fromarray(img_array)
            img_28 = np.array(img_pil.resize((28, 28), Image.Resampling.LANCZOS))
            
            # Invert colors
            img_28 = 255 - img_28
            
            st.image(img_28, width=150, caption="28x28")
            
            # Predict
            img_input = img_28.reshape(1, 28, 28, 1).astype('float32') / 255.0
            prediction = model.predict(img_input, verbose=0)
            
            # Show predictions
            top3 = np.argsort(prediction[0])[-3:][::-1]
            
            for idx in top3:
                st.write(f"{idx}: {prediction[0][idx]*100:.1f}%")
            
            st.write(f"Predicted: {np.argmax(prediction[0])}")

with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        if img.mode != 'L':
            img = img.convert('L')
        input_image = ("upload", np.array(img))

# Process and show results for Upload tab
if input_image is not None and input_image[0] == "upload":
    source, img_data = input_image
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Input")
        img_display = img_data
        st.image(img_display, width=200, caption="Input")
    
    with col2:
        st.write("### Model Input")
        
        # Convert to grayscale
        img_array = img_data
        
        # Resize to 28x28
        img_pil = Image.fromarray(img_array)
        img_28 = np.array(img_pil.resize((28, 28), Image.Resampling.LANCZOS))
        
        # Invert colors
        img_28 = 255 - img_28
        
        st.image(img_28, width=150, caption="28x28")
        
        # Predict
        img_input = img_28.reshape(1, 28, 28, 1).astype('float32') / 255.0
        prediction = model.predict(img_input, verbose=0)
        
        # Show predictions
        top3 = np.argsort(prediction[0])[-3:][::-1]
        
        for idx in top3:
            st.write(f"{idx}: {prediction[0][idx]*100:.1f}%")
        
        st.write(f"Predicted: {np.argmax(prediction[0])}")
