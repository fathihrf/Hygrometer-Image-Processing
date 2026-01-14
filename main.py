import cv2
import json
import os
import glob
import numpy as np
import easyocr
import re
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Constants
ANNOTATIONS_FILE = "annotations/annotations.json"
TRAIN_DIR = "images/train"
TEST_DIR = "images/test"
TEMPLATE_DIR = "templates"

def load_annotations():
    with open(ANNOTATIONS_FILE, 'r') as f:
        return json.load(f)

def iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def extract_templates(annotations):
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    
    # Find a training image in annotations
    # We assume the user has annotated at least one image in images/train
    # In our logic, we look for an image that exists in images/train
    
    train_images = glob.glob(os.path.join(TRAIN_DIR, "*"))
    ref_image_path = None
    
    for path in train_images:
        filename = os.path.basename(path)
        if filename in annotations:
            ref_image_path = path
            break
            
    if not ref_image_path:
        print("No annotated reference image found in images/train/")
        return False

    print(f"Using Master Image: {ref_image_path}")
    img = cv2.imread(ref_image_path)
    filename = os.path.basename(ref_image_path)
    annots = annotations[filename]
    
    templates = {}
    
    for label, rect in annots.items():
        if rect is None: continue
        x, y, w, h = rect
        template = img[y:y+h, x:x+w]
        template_path = os.path.join(TEMPLATE_DIR, f"{label}_template.jpg")
        cv2.imwrite(template_path, template)
        templates[label] = template
        print(f"Saved template for {label} to {template_path}")
        
    return templates

def preprocess_for_ocr(img_roi):
    # 1. Upscale
    scale = 3
    h, w = img_roi.shape[:2]
    upscaled = cv2.resize(img_roi, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    
    # 3. Normalize
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # 4. Otsu Thresholding
    # This automatically finds the best threshold
    _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Ensure Black Text on White Background
    # If empty, try INVERTED image (for Light text on Dark background)
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
        
    # Connect segments!
    # The digits are composed of separated segments (black on white).
    # We need to expand the black regions to touch each other.
    # Since background is white (255) and text is black (0),
    # EROSION of the image will shrink white regions -> expand black regions.
    kernel = np.ones((5,5), np.uint8)
    # Erode to connect components
    processed = cv2.erode(thresh, kernel, iterations=2)
    
    return processed

def process_test_images(templates, annotations, reader):
    test_images = glob.glob(os.path.join(TEST_DIR, "*"))
    
    results = []
    
    print("\n--- Testing ---")
    for img_path in test_images:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        # Ground Truth
        gt = annotations.get(filename)
        
        print(f"\nImage: {filename}")
        
        for label, template in templates.items():
            # Template Match
            # We use TM_CCOEFF_NORMED
            
            # Note: Ideally convert to gray, but for colored objects, color matching helps too.
            # Let's try simple BGR matching first (works fine with matchTemplate)
            
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # Predict Box
            h, w = template.shape[:2]
            top_left = max_loc
            pred_box = [top_left[0], top_left[1], w, h]
            
            score = max_val
            print(f"  [{label}] Match Score: {score:.4f}")
            
            # Draw Prediction (Red)
            cv2.rectangle(img, (pred_box[0], pred_box[1]), (pred_box[0]+w, pred_box[1]+h), (0, 0, 255), 2)
            
            # --- OCR Section ---
            # Crop the detected region
            roi_x, roi_y, roi_w, roi_h = pred_box
            # Ensure boundaries
            roi_y = max(0, roi_y)
            roi_x = max(0, roi_x)
            roi_img = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            if roi_img.size > 0:
                # Preprocess
                processed_roi = preprocess_for_ocr(roi_img)
                
                # Debug: save processed roi
                # cv2.imwrite(f"debug_{label}_{filename}", processed_roi)
                
                # Read text
                # Try without allowlist first to see what it detects
                raw_result = reader.readtext(processed_roi)
                # print(f"  [{label}] Raw OCR (Normal): {raw_result}")
                
                detected_text = ""
                
                # Helper to parse result
                def parse_ocr_result(res):
                     text_out = ""
                     for (_, text, conf) in res:
                         # Filter using regex to only keep numbers and dots
                         cleaned = re.sub(r'[^0-9.]', '', text)
                         text_out += cleaned
                     return text_out

                detected_text = parse_ocr_result(raw_result)

                # If empty, try INVERTED image (for Light text on Dark background)
                if not detected_text:
                    inverted_roi = cv2.bitwise_not(processed_roi)
                    raw_result_inv = reader.readtext(inverted_roi)
                    # print(f"  [{label}] Raw OCR (Inverted): {raw_result_inv}")
                    detected_text = parse_ocr_result(raw_result_inv)
                
                print(f"  [{label}] Detected Value: {detected_text}")
            # -------------------
            
            # Evaluate if GT exists
            if gt and label in gt:
                gt_box = gt[label]
                iou_val = iou(pred_box, gt_box)
                print(f"  [{label}] IoU: {iou_val:.4f}")
                
                 # Draw GT (Green)
                cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]), (0, 255, 0), 2)
            else:
                print(f"  [{label}] No GT found.")

        # Save result image
        output_path = f"output_{filename}"
        cv2.imwrite(output_path, img) 
        print(f"  Saved result to {output_path}")

def main():
    if not os.path.exists(ANNOTATIONS_FILE):
        print("No annotations found. Please run annotate.py first.")
        return

    annotations = load_annotations()
    
    # Phase A: Train
    templates = extract_templates(annotations)
    if not templates:
        return
        
    # Init OCR Reader once
    print("Initializing OCR Engine...")
    reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

    # Phase B: Test
    process_test_images(templates, annotations, reader)

if __name__ == "__main__":
    main()
