# HygroScan: Seven-Segment ROI Tracker & OCR

A robust computer vision tool designed to track and read **Temperature** and **Humidity** values from seven-segment displays (e.g., Hygrometers) using Template Matching and advanced PaddleOCR techniques.

## Features
- **ROI Tracking**: Uses Template Matching to locate Temperature and Humidity zones on the display.
- **Robust OCR**: Implements a Multi-Strategy Ensemble using **PaddleOCR**:
    - **Multi-ROI**: Checks both "Tight" and "Expanded" crops to capture truncated digits.
    - **Multi-Process**: Applies "Natural" (grayscale) and "Binary+Dilated" (morphological) preprocessing to handle both solid and segmented digits reliably.
    - **Accuracy**: Capable of reading difficult values like "24.3" even when partially cut off, and correctly interpreting seven-segment gaps.
- **Graphical Dashboard**: Built with **PyQt5** for easy image upload, preview, and verification.
- **Data Logging**: Automatically logs detected results (Timestamp, Filename, Values) to `results.csv`.

## Installation

### Prerequisites
- Python 3.10+
- Virtual Environment (recommended)

### Setup
1. **Clone/Download** the repository.
2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This installs `opencv-python`, `paddlepaddle`, `paddleocr`, and `pyqt5`.*

## Project Structure
- `app.py`: **Main Application**. Run this to start the GUI.
- `main.py`: **Backend Logic**. Contains the core OCR and image processing algorithms.
- `annotate.py`: Tool to create new templates/ground truth (if running from scratch).
- `ui/mainwindow.ui`: Qt Designer file for the interface.
- `images/`: Directory for Training (Templates) and Testing images.
- `results.csv`: Log file for exported data.

## Usage

### 1. Start the App
```bash
python app.py
```

### 2. Dashboard Workflow
- **Upload Image**: Click "Select Photo" to choose a test image.
- **View Results**: The app will automatically process the image:
    - Draw bounding boxes (Red = Prediction, Green = Ground Truth).
    - Display detected **Temperature** and **Humidity**.
- **Edit/Verify**: You can manually correct the values in the text boxes if needed.
- **Save**: Click **Confirm**. This saves the data to `results.csv`.

### 3. Annotating (Optional)
If you want to train on new device types:
1. Place reference images in `images/train/`.
2. Run `python annotate.py`.
3. Select the regions for "Temperature" and "Humidity".

## Troubleshooting
- **PaddleOCR Warning**: First run might take a moment to download/cache models.
- **"2" vs "24.3"**: If digits are cut off, the system now automatically expands the search area. If issues persist, check your template quality.
