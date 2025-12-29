def apply_dehaze(img, omega, clip_limit):
    start_time = time.time()
    
    # 1. Inversion (The Night-to-Day trick)
    inverted_img = 255 - img
    
    # 2. Refined Dark Channel
    # We use a smaller window (7x7) to detect haze more precisely
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dark = cv2.erode(np.min(inverted_img, axis=2), kernel)
    
    # 3. Robust Atmospheric Light (A)
    # We take the brightest 0.1% of pixels to find the "haze color"
    num_pixels = dark.size
    num_brightest = int(num_pixels * 0.001)
    flat_img = inverted_img.reshape(-1, 3)
    flat_dark = dark.flatten()
    search_indices = np.argsort(flat_dark)[-num_brightest:]
    A = np.mean(flat_img[search_indices], axis=0)
    
    # 4. Transmission Map + Guided Filter
    # This is the "Innovation": It smooths the haze removal so it's not blocky
    norm_img = inverted_img.astype(np.float64) / A
    transmission = 1 - omega * cv2.erode(np.min(norm_img, axis=2), kernel)
    
    # Guide the filter using the grayscale version of the image
    gray_guide = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    transmission = cv2.ximgproc.guidedFilter(guide=gray_guide, 
                                            src=transmission.astype(np.float32), 
                                            radius=40, eps=0.001)
    transmission = np.maximum(transmission, 0.1)
    
    # 5. Scene Recovery
    result = np.zeros(img.shape, dtype=np.float64)
    for i in range(3):
        result[:, :, i] = (inverted_img[:, :, i] - A[i]) / transmission + A[i]
    
    dehazed = 255 - np.clip(result, 0, 255).astype(np.uint8)
    
    # 6. Final Enhancement (CLAHE)
    lab = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    final = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    
    proc_time = time.time() - start_time
    
    # Calculate Metrics for the UI
    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    psnr_v = psnr(img, final)
    ssim_v = ssim(gray_orig, gray_final)
    
    return final, transmission, proc_time, psnr_v, ssim_v