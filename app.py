import streamlit as st
import cv2
import numpy as np
from streamlit_image_comparison import image_comparison
from PIL import Image
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- Advanced Processing Logic ---
def apply_dehaze(img, omega, clip_limit):
    start_time = time.time()
    
    # 1. PRE-PROCESSING: Denoising (Fixes the 'grain' in your screenshot)
    # We use a light denoise to keep details but remove sensor noise
    img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # 2. NIGHT INVERSION
    inverted_img = 255 - img_denoised
    
    # 3. TRANSMISSION ESTIMATION
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(np.min(inverted_img, axis=2), kernel)
    
    # Estimate Atmospheric Light (A)
    flat_dark = dark.flatten()
    indices = np.argpartition(flat_dark, -100)[-100:]
    A = np.max(inverted_img.reshape(-1, 3)[indices], axis=0)
    
    # Raw Transmission
    norm_img = inverted_img.astype(np.float64) / A.astype(np.float64)
    transmission = 1 - omega * cv2.erode(np.min(norm_img, axis=2), kernel)
    
    # INNOVATION: Guided Filter (Smoothes the 'blocky' artifacts)
    gray_guide = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    transmission = cv2.ximgproc.guidedFilter(guide=gray_guide, 
                                            src=transmission.astype(np.float32), 
                                            radius=40, eps=0.001)
    transmission = np.maximum(transmission, 0.1)
    
    # 4. SCENE RECOVERY
    result = np.zeros(img.shape, dtype=np.float64)
    for i in range(3):
        result[:, :, i] = (inverted_img[:, :, i] - A[i]) / transmission + A[i]
    
    dehazed = 255 - np.clip(result, 0, 255).astype(np.uint8)
    
    # 5. POST-PROCESSING: Color & Contrast Fix
    # Fix the blue tint with Simple White Balance
    result_norm = dehazed.astype(np.float32)
    for i in range(3):
        avg = np.mean(result_norm[:,:,i])
        result_norm[:,:,i] *= (128.0 / avg) # Force channels toward neutral gray
    dehazed = np.clip(result_norm, 0, 255).astype(np.uint8)

    # Professional Enhancement (CLAHE)
    lab = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    final = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    
    proc_time = time.time() - start_time
    
    # METRICS
    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    psnr_v = psnr(img, final)
    ssim_v = ssim(gray_orig, gray_final)
    
    return final, proc_time, psnr_v, ssim_v

# --- UI Setup ---
st.set_page_config(page_title="NightVision AI Pro", layout="wide")
st.title("🌌 NightVision AI: Professional Edition")

with st.sidebar:
    st.header("🎛️ Control Center")
    uploaded_file = st.file_uploader("Upload Hazy Night Image", type=["jpg", "png", "jpeg"])
    
    st.subheader("Settings")
    omega = st.slider("Haze Removal Strength", 0.70, 0.98, 0.85)
    clip = st.slider("Contrast Boost", 1.0, 4.0, 1.5)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    processed_bgr, p_time, psnr_v, ssim_v = apply_dehaze(img_bgr, omega, clip)
    
    # Display Result
    image_comparison(
        img1=Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
        img2=Image.fromarray(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)),
        label1="Original Haze",
        label2="Professional Dehazed"
    )

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("PSNR (Quality)", f"{psnr_v:.2f} dB")
    col2.metric("SSIM (Structure)", f"{ssim_v:.4f}")
    col3.metric("Speed", f"{p_time:.3f}s")
else:
    st.info("Please upload an image to start.")