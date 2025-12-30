import streamlit as st
import cv2
import numpy as np

def apply_pro_dehaze(img, brightness_boost):
    # 1. Strong Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # 2. Invert with a safety floor (prevents the black screen)
    inv_img = 255 - img
    
    # 3. Dark Channel Prior
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(np.min(inv_img, axis=2), kernel)
    
    # 4. Better Atmospheric Light estimation
    A = np.percentile(dark, 99)
    A = max(A, 1) # Safety check
    
    # 5. Transmission with safety floor
    t = 1 - 0.95 * (dark / float(A))
    t = np.maximum(t, 0.1)
    
    # 6. Scene Recovery
    res = np.zeros(img.shape, dtype=np.float32)
    for i in range(3):
        res[:, :, i] = (inv_img[:, :, i].astype(np.float32) - A) / t + A
    
    # Re-invert
    dehazed = 255 - np.clip(res, 0, 255).astype(np.uint8)
    
    # 7. PRO ENHANCEMENT (Fixes the darkness)
    # Convert to LAB to boost only the Lightness channel
    lab = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Use CLAHE + Simple Brightness Multiplier
    clahe = cv2.createCLAHE(clipLimit=brightness_boost, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced

st.set_page_config(page_title="NightVision AI Pro", layout="wide")
st.title("🌌 NightVision AI: Professional Dehazer")

# Sidebar controls
st.sidebar.header("Settings")
boost = st.sidebar.slider("Enhancement Strength", 1.0, 10.0, 4.0)

uploaded = st.file_uploader("Upload Night Photo", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    output = apply_pro_dehaze(img, boost)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Hazy Night")
    with col2:
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Dehazed & Enhanced")