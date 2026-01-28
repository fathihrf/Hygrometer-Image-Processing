# HygroScan: Seven-Segment ROI Tracker & OCR

A robust computer vision tool designed to track and read **Temperature** and **Humidity** values from seven-segment displays (e.g., Hygrometers) using Template Matching and advanced PaddleOCR techniques.

## Features
- **Smart OCR**: Uses lightweight **mobile models (PP-OCRv4)** for fast processing.
    - **Adaptive Thresholding**: Handles glare and shadow automatically.
    - **Multi-Candidate Processing**: Simultaneously analyzes "Normal", "Thickened", and "Inverted" image variations to correctly read broken segments (e.g., "24.3").
- **Automatic Cloud Upload**: Seamlessly pushes results to the **"Hygrometer Scan"** Google Sheet.
- **Workflow Integration**:
    - **QR Code Scanner**: Identify device location via webcam.
    - **RFID Placeholder**: Ready for RFID tag integration.
- **Graphical Dashboard**: 
    - Real-time preview and manual correction.
    - Branded UI with custom logos.
- **Single Entry Point**: Run the entire app with one command.

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
4. **Google Sheets Setup**:
   - Place your `credentials.json` (Service Account Key) in the root folder.
   - Create a Google Sheet named **"Hygrometer Scan"** and share it with your Service Account email.

## Usage

### 1. Start the App
```bash
python main.py
```

### 2. Workflow
1. **Identify Location**:
   - Use **"Scan QR Code"** to use your webcam.
   - Or use **"Scan RFID Tag"** (Simulation) to enter a tag ID.
2. **Upload & Analyze**:
   - After location selection, you are taken to the Upload page.
   - **Select** or **Drag & Drop** an image file.
   - The system automatically detects Temperature and Humidity.
3. **Review & Save**:
   - Verify the values (edit if necessary).
   - Click **Confirm & Save**.
   - Data is saved locally to `results.csv` AND uploaded to your Google Sheet.

## Project Structure
- `main.py`: **The App**. Contains both the GUI (PyQt5) and the Backend (OCR) logic.
- `annotate.py`: Tool to create new templates/ground truth.
- `ui/mainwindow.ui`: The graphical interface layout.
- `resource/`: Logos and assets.
- `images/`: Directory for Training (Templates) and Testing images.
- `results.csv`: Local log file.

## Troubleshooting
- **"Credentials not found"**: Ensure `credentials.json` is in the same folder as `main.py`.
- **"Spreadsheet not found"**: Make sure your Google Sheet is named exactly **"Hygrometer Scan"**.
