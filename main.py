import cv2
import json
import os
import glob
import numpy as np
from paddleocr import PaddleOCR
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
    # PaddleOCR works best on natural images, but for 7-segment, we often need to "connect" the dots.
    # 1. Upscale
    candidate_list = []

    # 1. Natural @ 2x Scale (Reduced from 4x for speed)
    h, w = img_roi.shape[:2]  # Fix: Define h, w before usage
    
    # Safety Check: Clamp max dimension to avoid OOM
    MAX_DIM = 2000
    scale_factor = 2.0
    if h * scale_factor > MAX_DIM or w * scale_factor > MAX_DIM:
        scale_factor = min(MAX_DIM / w, MAX_DIM / h)
    
    # Base Upscale to work with
    img_upscaled = cv2.resize(img_roi, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    
    # ---------------------------
    # Candidate 1: Natural (Upscaled + Border)
    # ---------------------------
    cand1 = cv2.copyMakeBorder(img_upscaled, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[127, 127, 127])
    candidate_list.append(cand1)

    # ---------------------------
    # Candidate 2: Adaptive Threshold (Best for Glare/Shadows)
    # ---------------------------
    gray = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2GRAY) if len(img_upscaled.shape) == 3 else img_upscaled
    
    # Adaptive Gaussian
    thresh_adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    
    # Check if we need to invert (ensure text is black on white)
    # Heuristic: Check corners. If corners are white, background is white.
    h_t, w_t = thresh_adap.shape
    corners = [thresh_adap[0,0], thresh_adap[0,w_t-1], thresh_adap[h_t-1,0], thresh_adap[h_t-1,w_t-1]]
    if np.mean(corners) < 127: # Background is black
         thresh_adap = cv2.bitwise_not(thresh_adap)
         
    cand2 = cv2.cvtColor(thresh_adap, cv2.COLOR_GRAY2BGR)
    cand2 = cv2.copyMakeBorder(cand2, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    candidate_list.append(cand2)
    
    # ---------------------------
    # Candidate 3: Thickened (Morphological Closing to connect segments)
    # ---------------------------
    kernel = np.ones((3,3), np.uint8)
    # Erode black text (which thickens it)
    thickened = cv2.erode(thresh_adap, kernel, iterations=1)
    
    cand3 = cv2.cvtColor(thickened, cv2.COLOR_GRAY2BGR)
    cand3 = cv2.copyMakeBorder(cand3, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    candidate_list.append(cand3)
    
    # ---------------------------
    # Candidate 4: Inverted Natural (For white-on-black text)
    # ---------------------------
    inverted_natural = cv2.bitwise_not(cand1)
    candidate_list.append(inverted_natural)

    return candidate_list

def analyze_image(img, templates, reader, ground_truth=None):
    """
    Analyzes a single image using the provided templates and OCR reader.
    Returns:
        processed_img: The image with annotations drawn.
        results: A dictionary of detected values {label: {'value': str, 'score': float, 'box': list}}.
    """
    results = {}
    
    for label, template in templates.items():
        # Template Match
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Predict Box
        h, w = template.shape[:2]
        top_left = max_loc
        pred_box = [top_left[0], top_left[1], w, h]
        
        score = max_val
        
        # Draw Prediction (Red)
        cv2.rectangle(img, (pred_box[0], pred_box[1]), (pred_box[0]+w, pred_box[1]+h), (0, 0, 255), 2)
        
        # --- OCR Section ---
        x1, y1, w1, h1 = pred_box
        x1 = max(0, x1)
        y1 = max(0, y1)
        roi_img_orig = img[y1:y1+h1, x1:x1+w1]
        
        # Expanded coords
        expand_w = int(w1 * 0.2)
        expand_h = int(h1 * 0.1)
        x2 = max(0, x1 - expand_w)
        y2 = max(0, y1 - expand_h)
        w2 = min(img.shape[1] - x2, w1 + 2 * expand_w)
        h2 = min(img.shape[0] - y2, h1 + 2 * expand_h)
        roi_img_exp = img[y2:y2+h2, x2:x2+w2]
        
        candidates = []
        if roi_img_orig.size > 0:
             candidates.extend(preprocess_for_ocr(roi_img_orig))
        if roi_img_exp.size > 0:
             candidates.extend(preprocess_for_ocr(roi_img_exp))
        
        final_text = ""
        if candidates:
            def parse_ocr_result(res):
                text_out = ""
                if not res: return text_out
                 
                # Case 0: Rec-only mode (det=False) -> returns [(text, score), ...]
                flat_res = []
                if isinstance(res, list):
                    for item in res:
                        if isinstance(item, tuple) and len(item) == 2:
                            flat_res.append(item)
                        elif isinstance(item, list):
                            for subitem in item:
                                if isinstance(subitem, tuple) and len(subitem) == 2:
                                    flat_res.append(subitem)
                 
                if flat_res:
                    for (text, score) in flat_res:
                        cleaned = re.sub(r'[^0-9.]', '', text)
                        text_out += cleaned
                    return text_out

                # Case 1: Dict format
                if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                    for item in res:
                        texts = item.get('rec_texts', [])
                        for text in texts:
                             cleaned = re.sub(r'[^0-9.]', '', text)
                             text_out += cleaned
                    return text_out
                 
                # Case 2: Standard Det+Rec format
                for line in res:
                    if isinstance(line, list):
                        if len(line) > 0 and isinstance(line[0], list) and len(line[0]) == 4 and isinstance(line[0][0], list):
                            text = line[1][0]
                            cleaned = re.sub(r'[^0-9.]', '', text)
                            text_out += cleaned
                        else:
                            for subline in line:
                                if isinstance(subline, list) and len(subline) >= 2:
                                    text = subline[1][0]
                                    cleaned = re.sub(r'[^0-9.]', '', text)
                                    text_out += cleaned
                return text_out

            best_text = ""
            best_text = ""
            best_score = 0
            
            for i, img_cand in enumerate(candidates):
                # OPTIMIZATION: Disable detection (det=False) and cls (cls=False)
                try:
                    result = reader.ocr(img_cand, det=False, cls=False)
                except Exception:
                    result = reader.ocr(img_cand)
                    
                detected_text = parse_ocr_result(result)
                
                # Heuristic: Prefer longer strings (e.g. "24.3" > "2")
                # Also check digit only
                if len(detected_text) > len(best_text):
                    best_text = detected_text
                elif len(detected_text) == len(best_text):
                    # Tie breaker? Usually first candidate (Natural) is safest if same length
                    pass
            
            final_text = best_text
            
        results[label] = {
            'value': final_text,
            'score': score,
            'box': pred_box
        }

        # Evaluate if GT exists
        if ground_truth and label in ground_truth:
            gt_box = ground_truth[label]
            iou_val = iou(pred_box, gt_box)
            # Draw GT (Green)
            cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]), (0, 255, 0), 2)
            results[label]['iou'] = iou_val

    return img, results

def process_test_images(templates, annotations, reader):
    test_images = glob.glob(os.path.join(TEST_DIR, "*"))
    
    print("\n--- Testing ---")
    for img_path in test_images:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        # Ground Truth
        gt = annotations.get(filename)
        
        print(f"\nImage: {filename}")
        
        processed_img, results = analyze_image(img, templates, reader, ground_truth=gt)
        
        for label, data in results.items():
            print(f"  [{label}] Match Score: {data['score']:.4f}")
            print(f"  [{label}] Detected Value: {data['value']}")
            if 'iou' in data:
                print(f"  [{label}] IoU: {data['iou']:.4f}")
            else:
                print(f"  [{label}] No GT found.")

        # Save result image
        output_path = f"output_{filename}"
        cv2.imwrite(output_path, processed_img) 
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
    print("Initializing OCR Engine (Mobile Mode v4)...")
    # Force PP-OCRv4 Mobile
    reader = PaddleOCR(lang='en', use_textline_orientation=False, ocr_version='PP-OCRv4')

    # Phase B: Test
    process_test_images(templates, annotations, reader)

if __name__ == "__main__":
    main()
