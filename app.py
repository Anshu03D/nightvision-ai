import streamlit as st
import cv2
import numpy as np

def apply_dehaze(img, omega=0.95, window_size=15):
    # --- STEP 1: PRE-PROCESSING (Crucial for Night) ---
    # We use a negative correction because night haze behaves like "inverted" day haze
    temp_img = img.astype(np.float32)
    inv_img = 255.0 - temp_img

    # --- STEP 2: DARK CHANNEL PRIOR (From piyas31 repo) ---
    # Finding the darkest pixels in the inverted image (which are the hazy areas)
    dark_channel = np.min(inv_img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(dark_channel, kernel)

    # --- STEP 3: ATMOSPHERIC LIGHT (A) ---
    # We take the top 0.1% brightest pixels in the dark channel
    num_pixels = dark_channel.size
    num_search = int(max(num_pixels * 0.001, 1))
    dark_vec = dark_channel.flatten()
    indices = np.argsort(dark_vec)[-num_search:]
    
    # Calculate A from the original inverted image using these indices
    A = np.mean(inv_img.reshape(-1, 3)[indices], axis=0)

    # --- STEP 4: TRANSMISSION MAP (t) ---
    # This is the 'depth' of the haze. We add a safety '0.1' to prevent black screens.
    t = 1.0 - omega * (dark_channel / np.max(A))
    t = np.maximum(t, 0.1) 

    # --- STEP 5: RADIANCE RECOVERY ---
    res = np.zeros(inv_img.shape, dtype=np.float32)
    for i in range(3):
        res[:, :, i] = (inv_img[:, :, i] - A[i]) / t + A[i]
    
    # Invert back to get the final night vision result
    result = 255.0 - res
    result = np.clip(result, 0, 255).astype(np.uint8)

    # --- STEP 6: POST-PROCESSING (Brightness & Contrast) ---
    # Night images are still dark after dehazing; we use CLAHE to fix this.
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    return result

# --- STREAMLIT UI ---
st.set_page_config(page_title="NightVision AI Pro", layout="wide")
st.title("🌌 NightVision AI (Based on Dark Channel Prior)")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Process
    with st.spinner("Analyzing atmospheric light..."):
        output = apply_dehaze(img)
    
    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
    with col2:
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="NightVision AI Result")