import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import traceback

# -----------------------------------------------------------------
#
# TASK 1: DEFINE THE CNN MODEL (Phase 2: "Identifier")
#
# -----------------------------------------------------------------

GROWTH_RATE = 32

def dense_layer(input_tensor):
    """Creates one "H_l" layer from Figure 12a."""
    x = layers.BatchNormalization()(input_tensor)
    x = layers.ReLU()(x)
    x = layers.Conv2D(GROWTH_RATE * 4, (1, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(GROWTH_RATE, (3, 3), strides=(1, 1), padding='same')(x)
    return x

def dense_block(input_tensor, num_layers=6):
    """Creates one "Block" from Figure 12b."""
    feature_list = [input_tensor]
    for i in range(num_layers):
        x = layers.Concatenate(axis=-1)(feature_list)
        output = dense_layer(x)
        feature_list.append(output)
    final_output = layers.Concatenate(axis=-1)(feature_list)
    return final_output

def transition_layer(input_tensor):
    """Creates the "Transition Layer" from Table III."""
    num_channels = input_tensor.shape[-1]
    x = layers.BatchNormalization()(input_tensor)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_channels // 2, (1, 1), strides=(1, 1), padding='same')(x)
    x = layers.AveragePooling2D((2, 2), strides=2)(x)
    return x

def create_model(input_shape=(64, 64, 3), num_classes=6):
    """Assembles the full network from Figure 12."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = layers.MaxPool2D((3, 3), strides=2, padding='same')(x)
    x = dense_block(x, num_layers=6)
    x = transition_layer(x)
    x = dense_block(x, num_layers=6)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# -----------------------------------------------------------------
#
# TASK 2: DEFINE THE INSPECTION PIPELINE (Phase 1: "Locator")
#
# -----------------------------------------------------------------

def run_inspection(template_path, test_image_path, model, class_names):
    """
    Runs the full Phase 1 and Phase 2 pipeline on a single
    template and test image.
    """
    
    print(f"--- Starting Inspection ---")
    print(f"Template: {template_path}")
    print(f"Test Image: {test_image_path}")

    # --- 1. Load Images ---
    template_color = cv2.imread(template_path)
    test_color = cv2.imread(test_image_path)
    
    if template_color is None:
        print(f"Error: Could not load template image from {template_path}")
        return
    if test_color is None:
        print(f"Error: Could not load test image from {test_image_path}")
        return

    template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_color, cv2.COLOR_BGR2GRAY)
    
    # --- 2. Registration (ORB) ---
    print("\nStep 1: Aligning images (Registration)...")
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) <= 10:
        print("Error: Not enough matches found for alignment.")
        return

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    h, w = template_gray.shape
    aligned_color = cv2.warpPerspective(test_color, M, (w, h))
    aligned_gray = cv2.cvtColor(aligned_color, cv2.COLOR_BGR2GRAY)
    
    # --- 3. Binarization & XOR ---
    print("\nStep 2: Finding differences (Binarization & XOR)...")
    thresh_template = cv2.adaptiveThreshold(template_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
    thresh_aligned = cv2.adaptiveThreshold(aligned_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    xor_result = cv2.bitwise_xor(thresh_template, thresh_aligned)
    
    # --- 4. Filtering & Morphology (Table VI) ---
    print("\nStep 3: Cleaning up differences (Filtering & Morphology)...")
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    step1 = cv2.medianBlur(xor_result, 5)
    kernel_15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    step2 = cv2.morphologyEx(step1, cv2.MORPH_CLOSE, kernel_15)
    step3 = cv2.morphologyEx(step2, cv2.MORPH_OPEN, kernel_5)
    step4 = cv2.medianBlur(step3, 5)
    kernel_29_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
    step5 = cv2.morphologyEx(step4, cv2.MORPH_CLOSE, kernel_29_ellipse)
    final_mask = cv2.morphologyEx(step5, cv2.MORPH_OPEN, kernel_5)
    
    # --- 5. Cropping Defects ---
    print("\nStep 4: Locating and cropping defects...")
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    defect_crops = []
    defect_bboxes = []
    padding = 15
    img_h, img_w = aligned_color.shape[:2]
    MIN_DEFECT_AREA = 75
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_DEFECT_AREA:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        x_end = min(img_w, x + w + padding)
        y_end = min(img_h, y + h + padding)
        
        defect_crop = aligned_color[y_pad:y_end, x_pad:x_end]
        
        defect_crops.append(defect_crop)
        defect_bboxes.append((x, y, w, h))
        
    print(f"Found {len(defect_crops)} potential defect(s) after filtering.")
    
    # --- 6. Classification (Phase 2) ---
    output_image = aligned_color.copy() 
    cropped_defects_display = [] 
    predictions_for_crops = []   

    if not defect_crops:
        print("\nNo defects found. The PCB is clean!")
    else:
        print("\nStep 5: Classifying defects with trained model...")
        for i, crop in enumerate(defect_crops):
            resized_crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
            normalized_crop = resized_crop.astype('float32') / 255.0
            input_tensor = np.expand_dims(normalized_crop, axis=0)
            
            predictions = model.predict(input_tensor)
            pred_index = np.argmax(predictions[0])
            pred_confidence = np.max(predictions[0])
            pred_class_name = class_names[pred_index]
            
            cropped_defects_display.append(resized_crop)
            predictions_for_crops.append({'name': pred_class_name, 'conf': pred_confidence})
            
            x, y, w, h = defect_bboxes[i]
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output_image, f"{pred_class_name} ({pred_confidence*100:.1f}%)", (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- 7. FINAL DISPLAY: Show all steps in one window ---
    print("\n--- Inspection Complete ---")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    axes[0, 0].imshow(cv2.cvtColor(aligned_color, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Step 1: Aligned Test Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(thresh_template, cmap='gray')
    axes[0, 1].set_title("Step 2a: Binary Template")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(thresh_aligned, cmap='gray')
    axes[0, 2].set_title("Step 2b: Binary Aligned Image")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(xor_result, cmap='gray')
    axes[1, 0].set_title("Step 2c: XOR Result (Noisy)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(final_mask, cmap='gray')
    axes[1, 1].set_title("Step 3: Final Cleaned Mask")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Step 5: Final Result with BBoxes")
    axes[1, 2].axis('off')

    plt.suptitle("PCB Inspection Pipeline Summary", fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    if cropped_defects_display:
        num_defects = len(cropped_defects_display)
        if num_defects == 1:
            plt.figure(figsize=(5, 6))
        else:
            plt.figure(figsize=(num_defects * 5, 6))

        for i, crop in enumerate(cropped_defects_display):
            if num_defects == 1:
                plt.subplot(1, 1, 1)
            else:
                plt.subplot(1, num_defects, i + 1)
            
            plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            pred_info = predictions_for_crops[i]
            title = f"Prediction: '{pred_info['name']}'\nConf: {pred_info['conf']*100:.2f}%"
            plt.title(title)
            
            plt.axis('off')
        
        plt.suptitle("Individual Defect Classifications", fontsize=18)
        plt.show()

# -----------------------------------------------------------------
#
# TASK 3: PUT IT ALL TOGETHER
#
# -----------------------------------------------------------------

if __name__ == "__main__":

    TEMPLATE_PATH = 'template.JPG' # <<< PASTE TEMPLATE IMAGE PATH HERE (e.g., 'template.JPG')
    TEST_PATH = 'test_normal.jpg'     # <<< PASTE TEST IMAGE PATH HERE (e.g., 'test_normal.jpg')
    
    MODEL_PATH = 'best_model.h5'
    
    CLASS_NAMES = [
        'missing_hole', 'mouse_bite', 'open_circuit', 
        'short', 'spur', 'spurious_copper'
    ]

    # --- Validate Paths ---
    if TEMPLATE_PATH is None or TEST_PATH is None:
        print("Error: Please set the TEMPLATE_PATH and TEST_PATH variables at the bottom of the script.")
    elif not os.path.exists(TEMPLATE_PATH):
        print(f"Error: Template file not found at {TEMPLATE_PATH}")
    elif not os.path.exists(TEST_PATH):
        print(f"Error: Test file not found at {TEST_PATH}")
    elif not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please place 'best_model.h5' in the same folder as this script.")
    else:
        # Suppress TensorFlow GPU warnings for a cleaner output
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        
        try:
            print("Loading trained model...")
            model = create_model()
            model.load_weights(MODEL_PATH)
            print("Model loaded successfully.")
            
            run_inspection(TEMPLATE_PATH, TEST_PATH, model, CLASS_NAMES)
            
        except Exception as e:
            print(f"An error occurred during inspection: {e}")
            traceback.print_exc()