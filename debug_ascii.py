import cv2
import numpy as np
import sys

def print_ascii(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: {image_path} not found")
        return

    # Resize to small width (e.g. 80 chars)
    h, w = img.shape
    aspect = h/w
    new_w = 80
    new_h = int(new_w * aspect * 0.5) # 0.5 because char height > width
    small = cv2.resize(img, (new_w, new_h))
    
    chars = " .:-=+*#%@"
    for r in range(new_h):
        line = ""
        for c in range(new_w):
            val = small[r, c]
            idx = int(val / 255 * (len(chars)-1))
            line += chars[idx]
        print(line)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_ascii(sys.argv[1])
