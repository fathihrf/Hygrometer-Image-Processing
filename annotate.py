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
    
    # Handle high-res images by downsizing for display
    height, width = img.shape[:2]
    max_disp_width = 1280
    scale = 1.0
    img_disp = img
    
    if width > max_disp_width:
        scale = max_disp_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_disp = cv2.resize(img, (new_width, new_height))
        print(f"Resized for display (Scale: {scale:.2f})")

    for cls in CLASSES:
        print(f"Select ROI for [{cls}] (Draw box -> SPACE/ENTER). Press 'c' to cancel/skip.")
        
        # Add instruction on image
        display_copy = img_disp.copy()
        cv2.putText(display_copy, f"Select {cls} (SPACE to confirm, c to cancel)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        roi = cv2.selectROI(f"Annotating {filename}", display_copy, fromCenter=False, showCrosshair=True)
        
        # Handle cancel/skip
        if roi == (0,0,0,0) or roi[2] == 0 or roi[3] == 0:
             print(f"Skipped {cls}")
             current_file_annots[cls] = None
        else:
             # Scale coordinates back to original size
             x, y, w, h = roi
             final_rect = [
                 int(x / scale),
                 int(y / scale),
                 int(w / scale),
                 int(h / scale)
             ]
             print(f"Selected {cls}: {final_rect}")
             current_file_annots[cls] = final_rect
        
        # MacOS Fix: WaitKey loop to ensure window events process
        cv2.waitKey(1)
        
    cv2.destroyAllWindows()
    cv2.waitKey(1) # Extra wait for macOS to close window

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

    # Filter out already annotated images
    new_images = []
    for img_path in all_images:
        filename = os.path.basename(img_path)
        if filename in annotations:
            print(f"Skipping {filename} (already annotated)")
        else:
            new_images.append(img_path)
    
    if not new_images:
        print("No new images to annotate. All images have been annotated already.")
        print("To re-annotate, delete entries from annotations.json or use --force flag.")
        return
    
    print(f"\nFound {len(new_images)} new image(s) to annotate.\n")
    
    for img_path in new_images:
        annotate_image(img_path, annotations)

    print("Annotation complete.")

if __name__ == "__main__":
    main()
