import streamlit as st
import cv2
import numpy as np
from PIL import Image

def apply_pro_dehaze(img):
    # 1. Denoise to clean up night grain
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # 2. Invert image (Night dehazing requires inversion)
    inv_img = 255 - img
    
    # 3. Dark Channel Prior
    kernel = np.ones((15, 15), np.uint8)
    dark = cv2.erode(np.min(inv_img, axis=2), kernel)
    
    # 4. Atmospheric Light & Transmission
    # Increased omega to 0.95 for more visible dehazing
    A = np.max(dark)
    t = 1 - 0.95 * (dark / float(A if A > 0 else 1))
    t = np.maximum(t, 0.1)
    
    # 5. Scene Recovery
    res = np.zeros(img.shape, dtype=np.float32)
    for i in range(3):
        res[:, :, i] = (inv_img[:, :, i] - A) / t + A
    
    dehazed = 255 - np.clip(res, 0, 255).astype(np.uint8)
    
    # 6. BRIGHTNESS BOOST (The "Magic" step)
    # Using CLAHE to pull detail out of the dark areas
    lab = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    final = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    
    return final

st.title("🌌 NightVision AI Pro")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Run the process
    output = apply_pro_dehaze(img)
    
    # Display Side-by-Side
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
    with col2:
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Dehazed & Enhanced")