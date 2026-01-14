# ROI Tracker Guide

## Step 1: Setup Environment
Ensure you are in the project directory and the virtual environment is active.
```bash
source venv/bin/activate
```
*(If you haven't installed dependencies yet)*
```bash
pip install -r requirements.txt
```

## Step 2: Prepare Images
Place your images in the following folders. **Ensure all filenames are unique** across both folders.
- **Reference/Training Images**: Place clearly visible images of your target objects in `images/train/`.
- **Testing Images**: Place the images you want to track/detect in `images/test/`.

## Step 3: Annotate Images
Run the annotation tool to define what you want to track.
```bash
python annotate.py
```
**Controls:**
- The script will open each image one by one.
- **Draw Box**: Click and drag to select the region for the requested class (e.g., "Temperature", "Humidity").
- **Confirm**: Press `SPACE` or `ENTER` to confirm the selection.
- **Cancel/Skip**: Press `c` to cancel the selection for a specific class if the object is not visible.

## Step 4: Run Tracking
Run the main script to perform template matching.
```bash
python main.py
```
**What happens:**
1. It creates templates from the annotated images in `images/train/`.
2. It searches for those templates in `images/test/`.
3. It prints the Match Score and IoU (if ground truth exists).
4. It saves the results with bounding boxes as `output_<filename>.jpg`.

## Step 5: Check Results
View the generated `output_*.jpg` files in the project root to visualize the detection accuracy.
