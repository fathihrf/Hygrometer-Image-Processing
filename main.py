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
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
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
    
    # Structure: templates[label] = [ {image: numpy_array, ...}, ... ]
    # Actually simpler: templates[label] = [numpy_array1, numpy_array2, ...]
    templates = {}
    
    count = 0
    for filename, annots in annotations.items():
        if filename.startswith("test_"): continue # Skip test data
        
        path = os.path.join(TRAIN_DIR, filename)
        if not os.path.exists(path):
            continue
            
        img = cv2.imread(path)
        if img is None: continue
        
        for label, rect in annots.items():
            if rect is None: continue
            x, y, w, h = rect
            
            # Basic sanity check
            if w <= 0 or h <= 0 or x < 0 or y < 0: continue
            
            template = img[y:y+h, x:x+w]
            if template.size == 0: continue
            
            if label not in templates: templates[label] = []
            templates[label].append(template)
        
        count += 1
        
    print(f"Extracted templates from {count} training images.")
    # Debug: Save first template of each label
    for label, t_list in templates.items():
        print(f"  - {label}: {len(t_list)} variations")
        if t_list:
            cv2.imwrite(os.path.join(TEMPLATE_DIR, f"{label}_master_debug.jpg"), t_list[0])
            
    return templates

def preprocess_for_ocr(img_roi):
    """Optimized preprocessing for LCD/7-segment displays"""
    candidate_list = []
    h, w = img_roi.shape[:2]
    
    # Scale up for better OCR (3x works well for small LCD displays)
    scale_factor = 3.0
    img_upscaled = cv2.resize(img_roi, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2GRAY) if len(img_upscaled.shape) == 3 else img_upscaled
    
    # Candidate 1: High contrast adaptive threshold (best for LCD with glare)
    # Denoise first
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    
    # Ensure black text on white background
    if np.mean(thresh[:10, :10]) < 127:
        thresh = cv2.bitwise_not(thresh)
    
    # Clean noise with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    cand1 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cand1 = cv2.copyMakeBorder(cand1, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    candidate_list.append(cand1)
    
    # Candidate 2: Simple Otsu threshold with padding
    denoised2 = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(denoised2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(otsu[:10, :10]) < 127:
        otsu = cv2.bitwise_not(otsu)
    
    cand2 = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    cand2 = cv2.copyMakeBorder(cand2, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    candidate_list.append(cand2)
    
    # Candidate 3: Contrast enhancement + threshold
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, enhanced_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(enhanced_thresh[:10, :10]) < 127:
        enhanced_thresh = cv2.bitwise_not(enhanced_thresh)
    
    cand3 = cv2.cvtColor(enhanced_thresh, cv2.COLOR_GRAY2BGR)
    cand3 = cv2.copyMakeBorder(cand3, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    candidate_list.append(cand3)

    return candidate_list

def analyze_image(img, templates, reader, ground_truth=None):
    results = {}
    
    # 0. Resize Input to match Training Domain (720p: 1280x720) for better ROI recognition
    # This helps template matching scale and matches training data resolution.
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720
    h_orig, w_orig = img.shape[:2]
    scale_factor = 1.0
    
    img_processing = img.copy()
    
    # Resize if image is larger than 720p
    if w_orig > TARGET_WIDTH or h_orig > TARGET_HEIGHT:
        # Calculate scale to fit within 720p while maintaining aspect ratio
        scale_w = TARGET_WIDTH / w_orig
        scale_h = TARGET_HEIGHT / h_orig
        scale_factor = min(scale_w, scale_h)
        
        new_w = int(w_orig * scale_factor)
        new_h = int(h_orig * scale_factor)
        img_processing = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 1. Multi-Template Matching
    for label, template_list in templates.items():
        best_score = -1
        best_box = None # [x, y, w, h] on img_processing
        
        for template in template_list:
            # Need to ensure template is smaller than image
            th, tw = template.shape[:2]
            ih, iw = img_processing.shape[:2]
            if th >= ih or tw >= iw: continue 
            
            res = cv2.matchTemplate(img_processing, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_score:
                best_score = max_val
                best_box = [max_loc[0], max_loc[1], tw, th]
        
        if best_box is None:
            print(f"Warning: No match found for {label}")
            continue

        # Scale predictions back to original image
        pred_box_orig = [
            int(best_box[0] / scale_factor),
            int(best_box[1] / scale_factor),
            int(best_box[2] / scale_factor),
            int(best_box[3] / scale_factor)
        ]

        # Draw Prediction (Red) on Original Image
        x, y, w, h = pred_box_orig
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        
        # --- OCR Section ---
        # Crop from original resolution for max detail
        x1 = max(0, x); y1 = max(0, y)
        roi_img_orig = img[y1:y1+h, x1:x1+w]
        
        # Expanded coords
        expand_w = int(w * 0.2); expand_h = int(h * 0.1)
        x2 = max(0, x1 - expand_w); y2 = max(0, y1 - expand_h)
        w2 = min(img.shape[1] - x2, w + 2 * expand_w)
        h2 = min(img.shape[0] - y2, h + 2 * expand_h)
        roi_img_exp = img[y2:y2+h2, x2:x2+w2]
        
        candidates = []
        if roi_img_orig.size > 0: candidates.extend(preprocess_for_ocr(roi_img_orig))
        if roi_img_exp.size > 0: candidates.extend(preprocess_for_ocr(roi_img_exp))
        
        final_text = ""
        if candidates:
            def clean_lcd_text(text):
                """Clean and normalize LCD digit text"""
                if not text:
                    return ""
                
                # Only keep digits and decimal points
                cleaned = re.sub(r'[^0-9.]', '', text)
                
                # Remove multiple consecutive dots
                while '..' in cleaned:
                    cleaned = cleaned.replace('..', '.')
                
                return cleaned
            
            def parse_ocr_result(res):
                """Parse PaddleOCR result and extract text with confidence"""
                if not res or res[0] is None:
                    return "", 0.0
                
                try:
                    # PaddleOCR with det=True returns: [[[box], (text, confidence)], ...]
                    if isinstance(res, list) and len(res) > 0:
                        all_texts = []
                        for item in res:
                            if isinstance(item, list) and len(item) >= 2:
                                # Format: [[points], (text, score)]
                                text_info = item[1] if len(item) > 1 else item[0]
                                if isinstance(text_info, tuple) and len(text_info) == 2:
                                    text, conf = text_info
                                    cleaned = clean_lcd_text(str(text))
                                    if cleaned:
                                        all_texts.append((cleaned, float(conf)))
                        
                        if all_texts:
                            # Return highest confidence result
                            best = max(all_texts, key=lambda x: x[1])
                            return best[0], best[1]
                except Exception as e:
                    print(f"    [Parse Error] {e}")
                
                return "", 0.0

            os.makedirs("debug_output", exist_ok=True)
            import time
            ts = int(time.time() * 1000)

            best_text = ""
            best_confidence = 0.0
            
            for idx, img_cand in enumerate(candidates):
                # Save debug image
                cv2.imwrite(f"debug_output/debug_ocr_candidate_{ts}_{idx}.png", img_cand)

                try:
                    # Use detection mode for better accuracy on LCD displays
                    result = reader.ocr(img_cand, det=True, cls=False)
                    detected_text, confidence = parse_ocr_result(result)
                    
                    if detected_text:
                        print(f"  > [{label}] Candidate {idx} Text: '{detected_text}' (conf: {confidence:.3f})")
                        
                        # Select based on confidence, not length
                        if confidence > best_confidence:
                            best_text = detected_text
                            best_confidence = confidence
                        elif confidence == best_confidence and len(detected_text) > len(best_text):
                            # Tie-breaker: longer text if same confidence
                            best_text = detected_text
                            
                except Exception as e:
                    print(f"  > [{label}] Candidate {idx} Error: {e}")
                    continue
            
            final_text = best_text
            
        results[label] = {
            'value': final_text,
            'score': best_score,
            'box': pred_box_orig
        }

        if ground_truth and label in ground_truth:
             gt_box = ground_truth[label] # These are for scaled images usually in test? 
             # Assuming GT is for the image provided.
             iou_val = iou(pred_box_orig, gt_box)
             cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]), (0, 255, 0), 2)
             results[label]['iou'] = iou_val

    return img, results



class OCRWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, img_path, templates, reader):
        super().__init__()
        self.img_path = img_path
        self.templates = templates
        self.reader = reader

    def run(self):
        try:
            img = cv2.imread(self.img_path)
            if img is None:
                self.error.emit(f"Error reading image: {self.img_path}")
                return
            
            # Run analysis
            # We copy Analyze Image logic here or call the global function.
            # Calling global function is cleaner.
            processed_img, results = analyze_image(img, self.templates, self.reader)
            self.finished.emit((processed_img, results))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

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
        self.worker = None
        
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

        self.lbl_status.setText("Processing... Please wait.")
        self.set_ui_busy(True)
        
        # Stop existing worker if running
        if self.worker and self.worker.isRunning():
            self.worker.wait()

        self.worker = OCRWorker(path, self.templates, self.ocr_reader)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_processing_finished(self, data):
        self.set_ui_busy(False)
        processed_img, results = data
        self.current_cv_image = processed_img
        
        try:
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
            self.lbl_status.setText(f"Display Error: {str(e)}")
            print(e)
            
    def on_processing_error(self, err_msg):
        self.set_ui_busy(False)
        self.lbl_status.setText(f"Error: {err_msg}")
        QMessageBox.warning(self, "Processing Error", err_msg)

    def set_ui_busy(self, busy):
        self.btn_select_photo.setEnabled(not busy)
        self.btn_confirm.setEnabled(not busy)
        if busy:
             self.setCursor(Qt.WaitCursor)
        else:
             self.setCursor(Qt.ArrowCursor)

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
