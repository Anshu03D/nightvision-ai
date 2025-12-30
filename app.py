import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_comparison import image_comparison
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NightVision AI Pro",
    page_icon="🌌",
    layout="wide"
)

# --- 2. LOGIC (Safe Version) ---
def process_night_vision(img, d_strength, b_boost, c_limit):
    # Denoising
    img_clean = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Dehazing (Inversion method)
    inv_img = 255 - img_clean
    kernel = np.ones((15, 15), np.uint8)
    dark_channel = cv2.erode(np.min(inv_img, axis=2), kernel)
    A = np.percentile(dark_channel, 99)
    
    # Transmission safety
    t = 1 - d_strength * (dark_channel / max(A, 1))
    t = np.maximum(t, 0.1)
    
    res = np.zeros(img.shape, dtype=np.float32)
    for i in range(3):
        res[:, :, i] = (inv_img[:, :, i].astype(np.float32) - A) / t + A
    
    dehazed = 255 - np.clip(res, 0, 255).astype(np.uint8)
    
    # Final Enhancement (CLAHE)
    lab = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=c_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    l = cv2.convertScaleAbs(l, alpha=b_boost, beta=10)
    
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# --- 3. UI ---
st.title("🌌 NightVision AI Pro")

# Sidebar Controls
with st.sidebar:
    st.header("Engine Settings")
    d_strength = st.slider("Dehaze Intensity", 0.5, 1.0, 0.9)
    b_boost = st.slider("Brightness Boost", 1.0, 3.0, 1.3)
    c_limit = st.slider("Contrast Limit", 1.0, 5.0, 2.5)

# File Uploader
uploaded = st.file_uploader("Upload Night Photo", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    # Read Image safely
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is not None:
        # Run Algorithm
        with st.spinner("Processing..."):
            output = process_night_vision(img, d_strength, b_boost, c_limit)
            
            # Convert for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
        # Display Metrics
        m1, m2 = st.columns(2)
        m1.metric("Original Brightness", f"{np.mean(img):.1f}")
        m2.metric("New Brightness", f"{np.mean(output):.1f}")

        # Comparison Slider
        image_comparison(
            img1=img_rgb,
            img2=output_rgb,
            label1="Original",
            label2="Enhanced"
        )
        
        # Download Buffer
        result_pil = Image.fromarray(output_rgb)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.sidebar.download_button(
            label="📥 Download Result",
            data=byte_im,
            file_name="nightvision_pro.png",
            mime="image/png"
        )
else:
    st.info("Please upload an image to begin.")