import sys
import os
import cv2
import csv
from datetime import datetime
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QGroupBox, QFormLayout, QScrollArea, QMessageBox, QInputDialog)
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import main as backend
import gspread
from google.oauth2.service_account import Credentials

from PyQt5.QtWidgets import QDialog, QStackedWidget
# Import the QR logic (we'll implement the dialog inside app.py or as a separate class)
# process_events is needed for UI updates during loops

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
        from PyQt5.QtCore import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
            
        data, bbox, _ = self.detector.detectAndDecode(frame)
        
        if bbox is not None and len(bbox) > 0:
            points = bbox[0] if len(bbox.shape) == 3 else bbox
            points = points.astype(int)
            n = len(points)
            for i in range(n):
                cv2.line(frame, tuple(points[i]), tuple(points[(i+1) % n]), (0, 255, 0), 3)

            if data:
                self.detected_data = data
                self.accept() # Close dialog with success
                return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        if self.timer:
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()

# Removed separate LocationSelectionPage class since we merged the UI


# Start defining the main class based on the UI file
class HygroScanApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load the UI file
        ui_file_path = os.path.join(os.path.dirname(__file__), 'ui', 'mainwindow.ui')
        if not os.path.exists(ui_file_path):
             QMessageBox.critical(self, "Error", f"UI file not found: {ui_file_path}")
             sys.exit(1)
             
        uic.loadUi(ui_file_path, self)
        
        # Initialize Location Selection Page
        # Since we merged the UI, the widgets page_location, btn_scan_qr, btn_scan_rfid 
        # are now directly available after uic.loadUi(ui_file_path, self)
        
        # Add to stackedWidget logic
        # Current Stacked: Index 0=Dashboard, Index 1=Location, Index 2=Upload
        # We need to ensure we know the indices.
        
        # Let's inspect what we have. page_dashboard, page_location, page_upload
        # The stackedWidget in mainwindow.ui has children in order of insertion in XML.
        # Order in XML was: page_dashboard, page_location, page_upload
        
        self.idx_dashboard = 0
        self.idx_location = 1
        self.idx_upload = 2
        
        # Track current location ID
        self.current_location_id = None
        
        # Init new UI location connections
        self.btn_scan_qr.clicked.connect(self.scan_qr)
        self.btn_scan_rfid.clicked.connect(self.scan_rfid)
        
        # Track current location ID
        self.current_location_id = None
        
        self.templates = {}
        self.ocr_reader = None
        self.current_image_path = None
        self.current_cv_image = None
        
        self.init_ui_connections()
        self.init_backend()
        
    def init_ui_connections(self):
        # Explicitly find widgets to ensure they exist
        self.btn_select_photo = self.findChild(QPushButton, "btn_select_photo")
        self.btn_confirm = self.findChild(QPushButton, "btn_confirm")
        self.frame_dropzone = self.findChild(QWidget, "frame_dropzone")
        
        if not self.btn_select_photo:
            print("CRITICAL ERROR: btn_select_photo not found!")
        else:
            print("DEBUG: btn_select_photo found, connecting...")
            self.btn_select_photo.clicked.connect(self.upload_image)
            
        if self.btn_confirm:
            self.btn_confirm.clicked.connect(self.save_result)
            
        # Enable Drag and Drop
        if self.frame_dropzone:
            self.frame_dropzone.setAcceptDrops(True)
            # We need to install event filter or subclass/patch the methods
            # Since uic loads it, we can patch the methods on the instance or the specific widget
            # Easier to set the MainWindow to accept drops and filter for the frame, 
            # OR just make the whole window accept drops for simplicity
            self.setAcceptDrops(True)
        
        # Navigation buttons
        self.btn_nav_dashboard = self.findChild(QPushButton, "btn_nav_dashboard")
        self.btn_nav_upload = self.findChild(QPushButton, "btn_nav_upload")
        
        if self.btn_nav_dashboard:
             self.btn_nav_dashboard.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(self.idx_dashboard))
        if self.btn_nav_upload:
             self.btn_nav_upload.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(self.idx_location))

        # Setup Brand Logo
        self.label_brand = self.findChild(QLabel, "label_brand")
        if self.label_brand:
            logo_path = "resource/images-8.png"
            if os.path.exists(logo_path):
                pixmap = QPixmap(logo_path)
                if not pixmap.isNull():
                    # Scale nicely to fit sidebar width (approx 200px wide)
                    scaled_pixmap = pixmap.scaledToWidth(180, Qt.SmoothTransformation)
                    self.label_brand.setPixmap(scaled_pixmap)
                    self.label_brand.setText("")
                    self.label_brand.setAlignment(Qt.AlignCenter)
                    # Override style to remove large padding if needed, keep it clean
                    self.label_brand.setStyleSheet("padding: 10px; background-color: transparent;")
            else:
                print(f"Warning: Logo file not found: {logo_path}")

        # Setup Secondary Logos
        self.label_brand_2 = self.findChild(QLabel, "label_brand_2")
        self.label_brand_3 = self.findChild(QLabel, "label_brand_3")
        
        # Logo 2: AVIF file (Try QPixmap, Fallback to CV2)
        logo2_path = "resource/enhR8cXOPjAZb2lPuJGHsvASVuq_eBFPvQPUifdVV8I.jpg.avif"
        if self.label_brand_2 and os.path.exists(logo2_path):
            pixmap = QPixmap(logo2_path)
            if pixmap.isNull():
                 # Try CV2 load
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

        # Logo 3: JPG File
        logo3_path = "resource/parb.jpg"
        if self.label_brand_3 and os.path.exists(logo3_path):
             pixmap = QPixmap(logo3_path)
             if not pixmap.isNull():
                  scaled = pixmap.scaledToWidth(180, Qt.SmoothTransformation)
                  self.label_brand_3.setPixmap(scaled)
                  self.label_brand_3.setStyleSheet("padding: 5px; background-color: transparent;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            # Just take the first valid image
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    print(f"DEBUG: Dropped file {f}")
                    # Switch to upload page if not already there? 
                    # Actually, if they drop, we should assume they want to process it.
                    # But we usually enforce Location Selection first.
                    # For now, let's just load it if we are on the upload page.
                    if self.stackedWidget.currentIndex() == self.idx_upload:
                        self.current_image_path = f
                        self.load_image_preview(f)
                        self.process_image(f)
                    else:
                        QMessageBox.warning(self, "Navigation", "Please select a location first before uploading.")
                    break

    def init_backend(self):
        # This might be slow, so we could thread it, but for now keep it simple
        try:
            if not os.path.exists(backend.ANNOTATIONS_FILE):
                self.lbl_status.setText("Error: Annotations file missing!")
                self.btn_select_photo.setEnabled(False)
                return

            self.lbl_status.setText("Loading annotations...")
            QApplication.processEvents()
            annotations = backend.load_annotations()
            
            self.lbl_status.setText("Extracting templates...")
            QApplication.processEvents()
            self.templates = backend.extract_templates(annotations)
            
            if not self.templates:
                self.lbl_status.setText("Error: Could not extract templates from training images.")
                self.btn_select_photo.setEnabled(False)
                return
                
            self.lbl_status.setText("Initializing OCR Engine (Mobile Mode v4)...")
            QApplication.processEvents()
            from paddleocr import PaddleOCR
            # Use Mobile models (default) and disable structure analysis which loads heavy models
            self.ocr_reader = PaddleOCR(lang='en', use_textline_orientation=False, ocr_version='PP-OCRv4')
            
            self.lbl_status.setText("Ready.")
            
        except Exception as e:
            self.lbl_status.setText(f"Initialization Error: {str(e)}")
            print(e)

    def upload_image(self):
        print("DEBUG: upload_image called")
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        
        if file_name:
            self.current_image_path = file_name
            self.load_image_preview(file_name)
            self.process_image(file_name)
            
    def load_image_preview(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            # Scale if too large
            if pixmap.width() > 800:
                pixmap = pixmap.scaledToWidth(800, Qt.SmoothTransformation)
            self.lbl_image_preview.setPixmap(pixmap)
            
    def process_image(self, path):
        if not self.ocr_reader or not self.templates:
            QMessageBox.warning(self, "Error", "Backend not ready.")
            return
            
        self.lbl_status.setText("Processing...")
        QApplication.processEvents()
        
        try:
            # Read image with CV2
            img = cv2.imread(path)
            if img is None:
                self.lbl_status.setText("Error reading image.")
                return
            
            # Analyze
            # We don't pass ground_truth here as we interpret new images
            processed_img, results = backend.analyze_image(img, self.templates, self.ocr_reader)
            self.current_cv_image = processed_img
            
            # Update Results
            temp_val = results.get("Temperature", {}).get("value", "")
            humid_val = results.get("Humidity", {}).get("value", "")
            
            self.input_temp.setText(temp_val)
            self.input_humidity.setText(humid_val)
            
            # Show processed image (convert BGR to RGB for Qt)
            rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            if pixmap.width() > 800:
                pixmap = pixmap.scaledToWidth(800, Qt.SmoothTransformation)
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
        if not self.current_image_path:
             return
             
        temp = self.input_temp.text()
        hum = self.input_humidity.text()
        
        # Save to CSV
        csv_file = "results.csv"
        file_exists = os.path.isfile(csv_file)
        
        try:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header if new file
                if not file_exists:
                    writer.writerow(["Timestamp", "Image Name", "Temperature", "Humidity", "Location ID"])
                    
                # Write data
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                image_name = os.path.basename(self.current_image_path)
                loc_id = self.current_location_id if self.current_location_id else "Unknown"
                # Updated CSV structure to include Location ID
                writer.writerow([timestamp, image_name, temp, hum, loc_id])
                
            QMessageBox.information(self, "Saved", f"Data saved to {csv_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save to CSV: {str(e)}")
            


        # Google Sheets Upload
        if os.path.exists('credentials.json'):
             # Automatic upload to "Hygrometer Scan"
             sheet_name = "Hygrometer Scan"
             try:
                 self.lbl_status.setText("Uploading to Drive...")
                 QApplication.processEvents()
                 
                 scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                 creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
                 client = gspread.authorize(creds)
                 
                 # Open Sheet
                 try:
                     sheet = client.open(sheet_name).sheet1
                     loc_id = self.current_location_id if self.current_location_id else "Unknown"
                     sheet.append_row([timestamp, image_name, temp, hum, loc_id])
                     QMessageBox.information(self, "Success", "Data uploaded to Google Sheet!")
                 except gspread.exceptions.SpreadsheetNotFound:
                      QMessageBox.warning(self, "Error", f"Spreadsheet '{sheet_name}' not found.\nPlease ensure you create a Google Sheet named '{sheet_name}' and share it with:\n{creds.service_account_email}")
                      
             except Exception as e:
                 QMessageBox.warning(self, "Google Sheets Error", str(e))
             finally:
                 self.lbl_status.setText("Ready.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HygroScanApp()
    window.show()
    sys.exit(app.exec_())
