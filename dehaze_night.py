import cv2
import numpy as np

def guided_filter(I, p, r, eps):
    """Refines the transmission map to respect image edges."""
    return cv2.ximgproc.createGuidedFilter(guide=I, radius=r, eps=eps).filter(p)

def get_dark_channel(img, size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_channel, kernel)

def estimate_atmospheric_light(img, dark_channel):
    h, w = dark_channel.shape
    num_pixels = h * w
    num_brightest = max(1, num_pixels // 1000)
    flat_dark = dark_channel.flatten()
    indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
    flat_img = img.reshape(-1, 3)
    return np.max(flat_img[indices], axis=0)

def enhance_night_image(img):
    # Convert to LAB color space to enhance brightness without ruining colors
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel (Lightness)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def dehaze_professional(image_path, output_path='comparison_result.jpg'):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # --- 1. PRE-PROCESS & INVERT ---
    # We work on a day-like version of the night image
    inverted_img = 255 - img
    
    # --- 2. CORE DEHAZING ---
    dark = get_dark_channel(inverted_img)
    A = estimate_atmospheric_light(inverted_img, dark)
    
    # Raw Transmission Map
    norm_img = inverted_img.astype(np.float64) / A.astype(np.float64)
    transmission = 1 - 0.95 * get_dark_channel(norm_img)
    
    # Refine Transmission with Guided Filter (Requires gray version of guide)
    gray_guide = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    transmission = guided_filter(gray_guide, transmission.astype(np.float32), 40, 0.001)
    transmission = np.maximum(transmission, 0.1)

    # Recover Scene
    result = np.zeros(img.shape, dtype=np.float64)
    for i in range(3):
        result[:, :, i] = (inverted_img[:, :, i].astype(np.float64) - A[i]) / transmission + A[i]
    
    dehazed = 255 - np.clip(result, 0, 255).astype(np.uint8)

    # --- 3. POST-PROCESS ENHANCEMENT ---
    final_output = enhance_night_image(dehazed)

    # --- 4. CREATE COMPARISON ---
    # Stack original and result side-by-side
    comparison = np.hstack((img, final_output))
    cv2.imwrite(output_path, comparison)
    
    print(f"Success! Comparison saved as {output_path}")
    cv2.imshow('Left: Original | Right: Professional Dehazed', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the process
dehaze_professional('night.jpg')