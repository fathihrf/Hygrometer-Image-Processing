import cv2
import json
import os
import glob
import numpy as np
from paddleocr import PaddleOCR
import re
import ssl
import sys
import csv
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QGroupBox, QFormLayout, QScrollArea, QMessageBox, QInputDialog, QDialog, QStackedWidget)
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
# Google Sheets Imports
import gspread
from google.oauth2.service_account import Credentials

ssl._create_default_https_context = ssl._create_unverified_context

# --- CONSTANTS & BACKEND LOGIC ---
ANNOTATIONS_FILE = "annotations/annotations.json"
TRAIN_DIR = "images/train"
TEST_DIR = "images/test"
TEMPLATE_DIR = "templates"

def load_annotations():
    with open(ANNOTATIONS_FILE, 'r') as f:
        return json.load(f)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def extract_templates(annotations):
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
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
    if img is None:
        print(f"Error reading image: {ref_image_path}")
        return False
        
    annots = annotations[os.path.basename(ref_image_path)]
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
    # PaddleOCR works best on natural images, but for 7-segment, we might need adjustments
    # 1. Upscale
    candidate_list = []

    # 1. Natural @ 2x Scale (Reduced from 4x for speed/memory)
    h, w = img_roi.shape[:2]
    MAX_DIM = 2000
    scale_factor = 2.0
    if h * scale_factor > MAX_DIM or w * scale_factor > MAX_DIM:
        scale_factor = min(MAX_DIM / w, MAX_DIM / h)
    
    img_upscaled = cv2.resize(img_roi, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    
    # Candidate 1: Natural (Upscaled + Border)
    cand1 = cv2.copyMakeBorder(img_upscaled, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[127, 127, 127])
    candidate_list.append(cand1)

    # Candidate 2: Adaptive Threshold (Best for Glare/Shadows)
    gray = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2GRAY) if len(img_upscaled.shape) == 3 else img_upscaled
    thresh_adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    
    # Check if we need to invert (ensure text is black on white)
    h_t, w_t = thresh_adap.shape
    corners = [thresh_adap[0,0], thresh_adap[0,w_t-1], thresh_adap[h_t-1,0], thresh_adap[h_t-1,w_t-1]]
    if np.mean(corners) < 127: # Background is black
         thresh_adap = cv2.bitwise_not(thresh_adap)
         
    cand2 = cv2.cvtColor(thresh_adap, cv2.COLOR_GRAY2BGR)
    cand2 = cv2.copyMakeBorder(cand2, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    candidate_list.append(cand2)
    
    # Candidate 3: Thickened (Morphological Closing to connect segments)
    kernel = np.ones((3,3), np.uint8)
    thickened = cv2.erode(thresh_adap, kernel, iterations=1)
    cand3 = cv2.cvtColor(thickened, cv2.COLOR_GRAY2BGR)
    cand3 = cv2.copyMakeBorder(cand3, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    candidate_list.append(cand3)
    
    # Candidate 4: Inverted Natural (For white-on-black text)
    inverted_natural = cv2.bitwise_not(cand1)
    candidate_list.append(inverted_natural)

    return candidate_list

def analyze_image(img, templates, reader, ground_truth=None):
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
        x1 = max(0, x1); y1 = max(0, y1)
        roi_img_orig = img[y1:y1+h1, x1:x1+w1]
        
        # Expanded coords
        expand_w = int(w1 * 0.2); expand_h = int(h1 * 0.1)
        x2 = max(0, x1 - expand_w); y2 = max(0, y1 - expand_h)
        w2 = min(img.shape[1] - x2, w1 + 2 * expand_w)
        h2 = min(img.shape[0] - y2, h1 + 2 * expand_h)
        roi_img_exp = img[y2:y2+h2, x2:x2+w2]
        
        candidates = []
        if roi_img_orig.size > 0: candidates.extend(preprocess_for_ocr(roi_img_orig))
        if roi_img_exp.size > 0: candidates.extend(preprocess_for_ocr(roi_img_exp))
        
        final_text = ""
        if candidates:
            def parse_ocr_result(res):
                text_out = ""
                if not res: return text_out
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

                # Standard format
                if isinstance(res, list) and len(res) > 0:
                     # Attempt to parse standard format just in case
                     pass 
                return text_out # Fallback handled by det=False usually returning simpler format

            best_text = ""
            for img_cand in candidates:
                try:
                    result = reader.ocr(img_cand, det=False, cls=False)
                except Exception:
                    try:
                        result = reader.ocr(img_cand)
                    except:
                        continue
                
                detected_text = parse_ocr_result(result)
                if len(detected_text) > len(best_text):
                    best_text = detected_text
            
            final_text = best_text
            
        results[label] = {
            'value': final_text,
            'score': score,
            'box': pred_box
        }

        if ground_truth and label in ground_truth:
            gt_box = ground_truth[label]
            iou_val = iou(pred_box, gt_box)
            cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]), (0, 255, 0), 2)
            results[label]['iou'] = iou_val

    return img, results


# --- GUI CLASSES ---

class QRScannerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan QR Code")
        self.resize(800, 600)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #333; background-color: black;")
        self.video_label.setFixedSize(640, 480)
        self.layout.addWidget(self.video_label)
        
        self.status_label = QLabel("Scanning...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)
        
        self.detected_data = None
        self.cap = cv2.VideoCapture(0)
        self.detector = cv2.QRCodeDetector()
        
        self.timer = None
        self.start_camera()

    def start_camera(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        data, bbox, _ = self.detector.detectAndDecode(frame)
        if bbox is not None and len(bbox) > 0:
            points = bbox[0] if len(bbox.shape) == 3 else bbox
            points = points.astype(int)
            for i in range(len(points)):
                cv2.line(frame, tuple(points[i]), tuple(points[(i+1) % len(points)]), (0, 255, 0), 3)

            if data:
                self.detected_data = data
                self.accept()
                return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        if self.timer: self.timer.stop()
        if self.cap.isOpened(): self.cap.release()
        event.accept()

class HygroScanApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load UI
        ui_file_path = os.path.join(os.path.dirname(__file__), 'ui', 'mainwindow.ui')
        if not os.path.exists(ui_file_path):
             QMessageBox.critical(self, "Error", f"UI file not found: {ui_file_path}")
             sys.exit(1)
        uic.loadUi(ui_file_path, self)
        
        # Navigation Indices
        self.idx_dashboard = 0
        self.idx_location = 1
        self.idx_upload = 2
        
        self.current_location_id = None
        self.templates = {}
        self.ocr_reader = None
        self.current_image_path = None
        self.current_cv_image = None
        
        # Connections
        self.btn_scan_qr.clicked.connect(self.scan_qr)
        self.btn_scan_rfid.clicked.connect(self.scan_rfid)
        
        self.init_ui_connections()
        self.init_backend()
        
    def init_ui_connections(self):
        self.btn_select_photo = self.findChild(QPushButton, "btn_select_photo")
        self.btn_confirm = self.findChild(QPushButton, "btn_confirm")
        self.frame_dropzone = self.findChild(QWidget, "frame_dropzone")
        
        if self.btn_select_photo: self.btn_select_photo.clicked.connect(self.upload_image)
        if self.btn_confirm: self.btn_confirm.clicked.connect(self.save_result)
        if self.frame_dropzone: self.setAcceptDrops(True)
        
        self.btn_nav_dashboard = self.findChild(QPushButton, "btn_nav_dashboard")
        self.btn_nav_upload = self.findChild(QPushButton, "btn_nav_upload")
        
        if self.btn_nav_dashboard:
             self.btn_nav_dashboard.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(self.idx_dashboard))
        if self.btn_nav_upload:
             self.btn_nav_upload.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(self.idx_location))

        # Logos
        self.label_brand = self.findChild(QLabel, "label_brand")
        if self.label_brand:
            logo_path = "resource/images-8.png"
            if os.path.exists(logo_path):
                pixmap = QPixmap(logo_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaledToWidth(180, Qt.SmoothTransformation)
                    self.label_brand.setPixmap(scaled_pixmap)
                    self.label_brand.setText("")
                    self.label_brand.setAlignment(Qt.AlignCenter)
                    self.label_brand.setStyleSheet("padding: 10px; background-color: transparent;")

        self.label_brand_2 = self.findChild(QLabel, "label_brand_2")
        self.label_brand_3 = self.findChild(QLabel, "label_brand_3")
        
        logo2_path = "resource/enhR8cXOPjAZb2lPuJGHsvASVuq_eBFPvQPUifdVV8I.jpg.avif"
        if self.label_brand_2 and os.path.exists(logo2_path):
            pixmap = QPixmap(logo2_path)
            if pixmap.isNull():
                 img = cv2.imread(logo2_path)
                 if img is not None:
                      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                      h, w, ch = img.shape
                      bytes_per_line = ch * w
                      qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                      pixmap = QPixmap.fromImage(qimg)
            if not pixmap.isNull():
                scaled = pixmap.scaledToWidth(180, Qt.SmoothTransformation)
                self.label_brand_2.setPixmap(scaled)
                self.label_brand_2.setStyleSheet("padding: 5px; background-color: transparent;")

        logo3_path = "resource/parb.jpg"
        if self.label_brand_3 and os.path.exists(logo3_path):
             pixmap = QPixmap(logo3_path)
             if not pixmap.isNull():
                  scaled = pixmap.scaledToWidth(180, Qt.SmoothTransformation)
                  self.label_brand_3.setPixmap(scaled)
                  self.label_brand_3.setStyleSheet("padding: 5px; background-color: transparent;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    if self.stackedWidget.currentIndex() == self.idx_upload:
                        self.current_image_path = f
                        self.load_image_preview(f)
                        self.process_image(f)
                    else:
                        QMessageBox.warning(self, "Navigation", "Please select a location first before uploading.")
                    break

    def init_backend(self):
        try:
            if not os.path.exists(ANNOTATIONS_FILE):
                self.lbl_status.setText("Error: Annotations file missing!")
                return

            self.lbl_status.setText("Loading annotations...")
            QApplication.processEvents()
            annotations = load_annotations()
            
            self.lbl_status.setText("Extracting templates...")
            QApplication.processEvents()
            self.templates = extract_templates(annotations)
            
            if not self.templates:
                self.lbl_status.setText("Error: Could not extract templates.")
                return
                
            self.lbl_status.setText("Initializing OCR Engine (Mobile Mode v4)...")
            QApplication.processEvents()
            # Force PP-OCRv4 Mobile
            self.ocr_reader = PaddleOCR(lang='en', use_textline_orientation=False, ocr_version='PP-OCRv4')
            
            self.lbl_status.setText("Ready.")
        except Exception as e:
            self.lbl_status.setText(f"Initialization Error: {str(e)}")
            print(e)
            
    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            self.current_image_path = file_name
            self.load_image_preview(file_name)
            self.process_image(file_name)
            
    def load_image_preview(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            if pixmap.width() > 800: pixmap = pixmap.scaledToWidth(800, Qt.SmoothTransformation)
            self.lbl_image_preview.setPixmap(pixmap)
            
    def process_image(self, path):
        if not self.ocr_reader or not self.templates:
            QMessageBox.warning(self, "Error", "Backend not ready.")
            return
        self.lbl_status.setText("Processing...")
        QApplication.processEvents()
        try:
            img = cv2.imread(path)
            if img is None:
                self.lbl_status.setText("Error reading image.")
                return
            # Call backend function directly
            processed_img, results = analyze_image(img, self.templates, self.ocr_reader)
            self.current_cv_image = processed_img
            
            temp_val = results.get("Temperature", {}).get("value", "")
            humid_val = results.get("Humidity", {}).get("value", "")
            
            self.input_temp.setText(temp_val)
            self.input_humidity.setText(humid_val)
            
            rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            if pixmap.width() > 800: pixmap = pixmap.scaledToWidth(800, Qt.SmoothTransformation)
            self.lbl_image_preview.setPixmap(pixmap)
            self.lbl_status.setText("Analysis Complete.")
        except Exception as e:
            self.lbl_status.setText(f"Processing Error: {str(e)}")
            print(e)

    def scan_qr(self):
        dialog = QRScannerDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            location_id = dialog.detected_data
            QMessageBox.information(self, "Location Identified", f"QR Location ID: {location_id}")
            self.go_to_upload_page(location_id)

    def scan_rfid(self):
        text, ok = QInputDialog.getText(self, "RFID Scan", "Waiting for RFID Tag (Simulated):\nEnter Tag ID:")
        if ok and text:
            self.go_to_upload_page(text)
            
    def go_to_upload_page(self, location_id):
        self.current_location_id = location_id
        self.stackedWidget.setCurrentIndex(self.idx_upload)

    def save_result(self):
        if not self.current_image_path: return
        temp = self.input_temp.text()
        hum = self.input_humidity.text()
        csv_file = "results.csv"
        file_exists = os.path.isfile(csv_file)
        try:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists: writer.writerow(["Timestamp", "Image Name", "Temperature", "Humidity", "Location ID"])
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                image_name = os.path.basename(self.current_image_path)
                loc_id = self.current_location_id if self.current_location_id else "Unknown"
                writer.writerow([timestamp, image_name, temp, hum, loc_id])
            QMessageBox.information(self, "Saved", f"Data saved to {csv_file}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save to CSV: {str(e)}")
            
        # Google Sheets
        if os.path.exists('credentials.json'):
             sheet_name = "Hygrometer Scan"
             try:
                 self.lbl_status.setText("Uploading to Drive...")
                 QApplication.processEvents()
                 scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                 creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
                 client = gspread.authorize(creds)
                 try:
                     sheet = client.open(sheet_name).sheet1
                     loc_id = self.current_location_id if self.current_location_id else "Unknown"
                     sheet.append_row([timestamp, image_name, temp, hum, loc_id])
                     QMessageBox.information(self, "Success", "Data uploaded to Google Sheet!")
                 except gspread.exceptions.SpreadsheetNotFound:
                      QMessageBox.warning(self, "Error", f"Spreadsheet '{sheet_name}' not found.\nCheck share.")
             except Exception as e:
                 QMessageBox.warning(self, "Google Sheets Error", str(e))
             finally:
                 self.lbl_status.setText("Ready.")

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HygroScanApp()
    window.show()
    sys.exit(app.exec_())
