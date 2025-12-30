import streamlit as st
import cv2
import numpy as np
from PIL import Image

def enhance_night_vision(img):
    # 1. DENOISE: Essential to stop the 'blue grain'
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # 2. INVERT: Treat night haze like day haze
    inv_img = 255 - img
    
    # 3. ESTIMATE TRANSMISSION (The "Dehaze" part)
    # A larger kernel (25x25) helps find thicker haze
    kernel = np.ones((25, 25), np.uint8)
    dark_channel = cv2.erode(np.min(inv_img, axis=2), kernel)
    
    # Atmospheric Light (A) estimation
    A = np.percentile(inv_img, 99) 
    
    # Transmission map
    t = 1 - 0.95 * (dark_channel / A)
    t = np.maximum(t, 0.1) # Prevent division by zero
    
    # 4. RECOVER SCENE
    res = np.zeros(img.shape, dtype=np.float32)
    for i in range(3):
        res[:, :, i] = (inv_img[:, :, i] - A) / t + A
    
    # Re-invert back to normal
    dehazed = 255 - np.clip(res, 0, 255).astype(np.uint8)
    
    # 5. FINAL ENHANCEMENT: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This is the "magic" that pulls details out of the dark
    lab = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    return enhanced

st.title("🌌 Professional NightVision Dehazer")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    with st.spinner('Applying Deep Dehaze...'):
        output = enhance_night_vision(img)
    
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Dehazed & Enhanced Result", use_column_width=True)