import cv2
import numpy as np
import os
from tqdm import tqdm
import tifffile as tiff

# ==========================================
# 0. CONFIGURATION
# ==========================================
BG_FOLDER = r"D:\Image Segmentation Task\DATA_SEG\8kV_ROI2_FV_75kfps_0p5slpm_R1_bg_C001H001S0001"
INPUT_FOLDER = r"D:\Image Segmentation Task\DATA_SEG\8kV_ROI2_FV_75kfps_0p5slpm_R1_C001H001S0001"
OUTPUT_FOLDER = r"D:\Image Segmentation Task\Test3\8kV_Preprocessed_Results\PP4_60_(CLOSE_KERNEL_SIZE_11)_(CLOSE_ITERATIONS_1)"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

AVG_COUNT = 100 

# --- PIPELINE CONFIGURATION ---
MIN_AREA_THRESHOLD = 60 # Step 1: Erase anything smaller than 60 pixels
CLOSE_KERNEL_SIZE = 11   # Step 2: Fill holes inside the surviving objects (Must be ODD)
CLOSE_ITERATIONS = 1

# ==========================================
# 1. UTILITY FUNCTIONS
# ==========================================
def load_16bit_raw(path):
    try:
        img = tiff.imread(path)
        return img.astype(np.uint16)
    except Exception as e:
        print(f"\n[IMAGE LOAD ERROR] Could not read file: {path}")
        return None

def apply_pro_panel_for_video(img_16bit, label):
    """Creates a clean, professional panel with small text and sharp borders."""
    img_float = img_16bit.astype(np.float32)
    i_min, i_max = img_float.min(), img_float.max()
    
    # Normalize to 8-bit for video display
    if i_max > i_min:
        img_8bit = (((img_float - i_min) / (i_max - i_min)) * 255.0).astype(np.uint8)
    else:
        img_8bit = np.zeros_like(img_16bit, dtype=np.uint8)
        
    img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    
    # Add a crisp 1-pixel white border around the frame
    img_bgr = cv2.copyMakeBorder(img_bgr, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    # Add a sleek, dark header bar at the top (only 25 pixels tall)
    cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 25), (30, 30, 30), -1)
    
    # Add smaller, cleaner text
    cv2.putText(img_bgr, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    
    return img_bgr

# ==========================================
# 2. THE CUSTOM PIPELINE ENGINE
# ==========================================
def isolate_with_custom_pipeline(img_16bit, min_area, close_k, close_iter):
    img_float = img_16bit.astype(np.float32)
    i_min, i_max = img_float.min(), img_float.max()
    
    if i_max == i_min:
        return np.zeros_like(img_16bit)
        
    scout_8bit = (((img_float - i_min) / (i_max - i_min)) * 255.0).astype(np.uint8)
    
    # Thresholding
    _, binary_all = cv2.threshold(scout_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Area Filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_all, connectivity=8)
    filtered_binary = np.zeros_like(binary_all)
    for i in range(1, num_labels): 
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_binary[labels == i] = 255
            
    # Morph Close
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    closed_binary = cv2.morphologyEx(filtered_binary, cv2.MORPH_CLOSE, close_kernel, iterations=close_iter)
            
    clean_16bit = np.where(closed_binary > 0, img_16bit, 0).astype(np.uint16)
    return clean_16bit

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def run_pipeline():
    if not os.path.exists(BG_FOLDER) or not os.path.exists(INPUT_FOLDER):
        print(f"\n[FATAL ERROR] Check your folder paths.")
        return

    bg_files = sorted([f for f in os.listdir(BG_FOLDER) if f.lower().endswith('.tif')])
    bg_limit = min(AVG_COUNT, len(bg_files))
    
    sample_bg = load_16bit_raw(os.path.join(BG_FOLDER, bg_files[0]))
    h, w = sample_bg.shape
    acc = np.zeros((h, w), np.float64)
    
    for i in tqdm(range(bg_limit), desc="Averaging BG"):
        frame = load_16bit_raw(os.path.join(BG_FOLDER, bg_files[i]))
        if frame is not None: acc += frame
    avg_bg = (acc / bg_limit).astype(np.uint16)

    target_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.tif')])[:4000]

    # Set up the 3-panel video output with the requested simple name
    video_w, video_h = (w + 2) * 3, (h + 2) # Adjusted for the 1-pixel borders
    video_path = os.path.join(OUTPUT_FOLDER, "8kv_Pre_processed.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_w, video_h))

    for f in tqdm(target_files, desc="Processing"):
        img = load_16bit_raw(os.path.join(INPUT_FOLDER, f))
        if img is None: continue

        # --- Subtraction ---
        diff = cv2.absdiff(img, avg_bg).astype(np.float32)
        d_min, d_max = diff.min(), diff.max()
        normalized = ((diff - d_min) / (d_max - d_min)) * 65535.0 if d_max > d_min else diff
        diff_16bit = normalized.astype(np.uint16)
        
        # --- Processing ---
        clean_16bit = isolate_with_custom_pipeline(diff_16bit, MIN_AREA_THRESHOLD, CLOSE_KERNEL_SIZE, CLOSE_ITERATIONS)
        
        # Save the actual cleaned frames directly into the simplified folder
        tiff.imwrite(os.path.join(OUTPUT_FOLDER, f), clean_16bit)

        # --- Panel 3 Logic: Pure Visual Delta ---
        delta_16bit = cv2.absdiff(diff_16bit, clean_16bit)

        # --- VIDEO PREPARATION ---
        p1 = apply_pro_panel_for_video(diff_16bit, "1. BG SUBTRACTED")
        p2 = apply_pro_panel_for_video(clean_16bit, "2. CLEANED")
        p3 = apply_pro_panel_for_video(delta_16bit, "3. DELTA (NOISE CUT OFF)")

        combined_frame = np.hstack((p1, p2, p3))
        writer.write(combined_frame)

    writer.release()
    print(f"\n[SUCCESS] Pipeline completed. Files and Video saved to: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    run_pipeline()
