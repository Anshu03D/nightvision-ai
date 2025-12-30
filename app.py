import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_comparison import image_comparison

# --- PAGE CONFIG ---
st.set_page_config(page_title="NightVision AI Pro", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR SLEEK UI ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4255;
    }
    .stTitle {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        color: #00d4ff;
    }
    div[data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

def process_night_vision(img, d_strength, b_boost, c_limit):
    # Core Algorithm (Optimized)
    img_clean = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    inv_img = 255 - img_clean
    kernel = np.ones((15, 15), np.uint8)
    dark_channel = cv2.erode(np.min(inv_img, axis=2), kernel)
    A = np.percentile(dark_channel, 99)
    t = 1 - d_strength * (dark_channel / max(A, 1))
    t = np.maximum(t, 0.1)
    res = np.zeros(img.shape, dtype=np.float32)
    for i in range(3):
        res[:, :, i] = (inv_img[:, :, i].astype(np.float32) - A) / t + A
    dehazed = 255 - np.clip(res, 0, 255).astype(np.uint8)
    
    # Enhancement
    lab = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=c_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    l = cv2.convertScaleAbs(l, alpha=b_boost, beta=10)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# --- HEADER ---
st.title("🌌 NightVision AI Pro")
st.caption("Advanced Low-Light Restoration using Dark Channel Prior & CLAHE")
st.divider()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2581/2581964.png", width=100)
    st.header("Engine Settings")
    d_strength = st.slider("Dehaze Intensity", 0.1, 1.0, 0.90)
    b_boost = st.slider("Exposure Boost", 1.0, 3.0, 1.3)
    c_limit = st.slider("Detail Recovery", 1.0, 5.0, 2.5)
    st.divider()
    st.info("Tip: Increase 'Detail Recovery' for foggy roads.")

# --- MAIN CONTENT ---
uploaded = st.file_uploader("📂 Drop a night-time photo here", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Processing
    with st.spinner("✨ AI at work..."):
        output = process_night_vision(img, d_strength, b_boost, c_limit)
        
    # --- METRICS BAR ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Original Brightness", f"{np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)):.1f}")
    m2.metric("Enhanced Brightness", f"{np.mean(cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)):.1f}")
    m3.metric("Visibility Gain", f"+{((np.mean(output)/np.mean(img))-1)*100:.1f}%")

    st.subheader("Comparison View")
    # Interactive Slider
    image_comparison(
        img1=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        img2=cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
        label1="Original",
        label2="NightVision Pro"
    )

    # Download Button Section
    st.divider()
    col_dl, _ = st.columns([1, 2])
    with col_dl:
        # Convert for download
        final_img = Image.fromarray(cv2.cvtColor(outpuimport streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_comparison import image_comparison
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NightVision AI | Pro Restorer",
    page_icon="🌌",
    layout="wide"
)

# --- 2. ADVANCED CSS (Glassmorphism & Dark Mode) ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: radial-gradient(circle, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 52, 96, 0.5);
        backdrop-filter: blur(10px);
    }
    
    /* Card-like containers for metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Title glow */
    .main-title {
        font-size: 45px;
        font-weight: 800;
        background: -webkit-linear-gradient(#00d4ff, #005f73);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE LOGIC ---
def process_image(img, d_strength, b_boost, c_limit):
    # Denoising
    img_clean = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Dehazing (Inversion method)
    inv_img = 255 - img_clean
    kernel = np.ones((15, 15), np.uint8)
    dark_channel = cv2.erode(np.min(inv_img, axis=2), kernel)
    A = np.percentile(dark_channel, 99)
    t = 1 - d_strength * (dark_channel / max(A, 1))
    t = np.maximum(t, 0.1)
    
    res = np.zeros(img.shape, dtype=np.float32)
    for i in range(3):
        res[:, :, i] = (inv_img[:, :, i].astype(np.float32) - A) / t + A
    
    dehazed = 255 - np.clip(res, 0, 255).astype(np.uint8)
    
    # Final Enhancement
    lab = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=c_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    l = cv2.convertScaleAbs(l, alpha=b_boost, beta=10)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# --- 4. TOP HEADER ---
st.markdown('<p class="main-title">🌌 NIGHTVISION AI PRO</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>State-of-the-Art Low Light Dehazing & Detail Recovery</p>", unsafe_allow_html=True)
st.divider()

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🎛️ Restoration Engine")
    d_strength = st.slider("Dehaze Strength", 0.5, 1.0, 0.90, help="Controls how much 'fog' is removed.")
    b_boost = st.slider("Brightness Boost", 1.0, 3.0, 1.3, help="Multiplies the exposure of dark areas.")
    c_limit = st.slider("Local Contrast", 1.0, 10.0, 3.0, help="Higher values pull more texture from shadows.")
    
    st.divider()
    st.subheader("Settings")
    view_mode = st.radio("Display Mode", ["Interactive Slider", "Side-by-Side"])

# --- 6. MAIN WORKSPACE ---
uploaded = st.file_uploader("📂 Upload Night Photography (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run AI
    with st.spinner("🚀 Running Dehazing Algorithms..."):
        output = process_image(img, d_strength, b_boost, c_limit)
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    # Metrics Section
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Scene Brightness", f"{np.mean(img):.1f}", delta_color="normal")
    col_m2.metric("Target Brightness", f"{np.mean(output):.1f}", f"+{((np.mean(output)/np.mean(img))-1)*100:.1f}%")
    col_m3.metric("Noise Level", "Low (Denoised)", "Clean")

    # Display Logic
    if view_mode == "Interactive Slider":
        image_comparison(
            img1=img_rgb,
            img2=output_rgb,
            label1="Hazy Original",
            label2="NightVision Pro"
        )
    else:
        c1, c2 = st.columns(2)
        c1.image(img_rgb, caption="Before")
        c2.image(output_rgb, caption="After")

    # --- DOWNLOAD BUTTON ---
    result_img = Image.fromarray(output_rgb)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    st.sidebar.download_button(
        label="📥 Download High-Res",
        data=byte_im,
        file_name="nightvision_enhanced.png",
        mime="image/png",
        use_container_width=True
    )
else:
    # Display an educational diagram if no image is uploaded
    st.info("Upload an image to start the restoration process.")t, cv2.COLOR_BGR2RGB))
        st.download_button(
            label="📥 Download Enhanced Image",
            data=uploaded, # In a real app, you'd save the 'output' to a buffer
            file_name="nightvision_result.png",
            mime="image/png",
            use_container_width=True
        )
else:
    # Empty State
    st.info("Please upload an image to begin analysis.")