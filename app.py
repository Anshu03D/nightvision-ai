import streamlit as st
import cv2
import numpy as np
from PIL import Image

# This version is ultra-light for faster cloud loading
def apply_dehaze(img):
    # Denoise to fix the "grain" from your screenshot
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Simple Dehazing Logic
    inverted = 255 - img
    # ... (rest of your math) ...
    return 255 - inverted 

st.title("🌌 NightVision AI")
uploaded = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded:
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Process
    result = apply_dehaze(img)
    
    # Display side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
    with col2:
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Dehazed")