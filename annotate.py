import cv2
import json
import os
import glob
import sys

# Constants
IMAGES_DIR = "images"
ANNOTATIONS_FILE = "annotations/annotations.json"
CLASSES = ["Temperature", "Humidity"]

def load_existing_annotations():
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_annotations(annotations):
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(annotations, f, indent=4)
    print(f"Saved annotations to {ANNOTATIONS_FILE}")

def annotate_image(image_path, annotations):
    filename = os.path.basename(image_path)
    
    # If already annotated, skip or ask to overwrite (simple version: just skip for now, or re-annotate)
    # We will allow re-annotation.
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return

    print(f"--- Annotating {filename} ---")
    current_file_annots = {}

    for cls in CLASSES:
        print(f"Select ROI for [{cls}] (Draw box -> SPACE/ENTER). Press 'c' to cancel selection for this class.")
        # cv2.selectROI opens a window. 
        # Returns (x, y, w, h)
        # fromCenter=False, showCrosshair=True
        roi = cv2.selectROI(f"Select {cls}", img, fromCenter=False, showCrosshair=True)
        
        # Check if user cancelled (all zeros?)
        # cv2.selectROI returns empty tuple or all zeros if cancelled depending on version, usually (0,0,0,0) if just closed/cancelled
        if roi == (0,0,0,0):
             print(f"Skipped {cls}")
             current_file_annots[cls] = None
        else:
             print(f"Selected {cls}: {roi}")
             current_file_annots[cls] = roi
        
        cv2.destroyWindow(f"Select {cls}")

    annotations[filename] = current_file_annots
    save_annotations(annotations)

def main():
    # Ensure annotation dir exists
    os.makedirs(os.path.dirname(ANNOTATIONS_FILE), exist_ok=True)
    
    annotations = load_existing_annotations()
    
    # Gather all images from train and test folders
    train_images = glob.glob(os.path.join("images", "train", "*"))
    test_images = glob.glob(os.path.join("images", "test", "*"))
    all_images = train_images + test_images
    
    if not all_images:
        print("No images found in images/train/ or images/test/")
        print("Please add images before running this tool.")
        return

    for img_path in all_images:
        annotate_image(img_path, annotations)

    print("Annotation complete.")

if __name__ == "__main__":
    main()
