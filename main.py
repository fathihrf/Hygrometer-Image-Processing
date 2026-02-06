import sys
import cv2
import json
import os

venv_scripts_path = os.path.dirname(sys.executable)
if os.name == 'nt':
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(venv_scripts_path)
            print(f"Added DLL Path: {venv_scripts_path}")
        except: pass
    os.environ['PATH'] = venv_scripts_path + os.pathsep + os.environ['PATH']

import glob
import numpy as np
import threading
import pickle
import re
import time
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QMessageBox, QInputDialog, QDialog)
from PyQt5 import uic
from paddleocr import PaddleOCR 
import gspread
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_SECRET_FILE = os.path.join(BASE_DIR, 'client_secret.json')
SPREADSHEET_ID = '1F1TRrJ3jsph7N1MzMjUgjaHeasH-SyUSoJUJcxCRI4M' 
DRIVE_FOLDER_ID = '11OT0wm7lfveDRLn1O2uCKUTyzsUZOKwj'
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "annotations", "annotations.json")
TRAIN_DIR = os.path.join(BASE_DIR, "images", "train")

# import UI
try:
    from capture import Ui_Form as Ui_Capture
    from confirm import Ui_Dialog as Ui_Confirm
except ImportError:
    try:
        from ui.capture import Ui_Form as Ui_Capture
        from ui.confirm import Ui_Dialog as Ui_Confirm
    except:
        sys.exit()


def optimized_template_match(img, template, scale=4):
    h, w = img.shape[:2]
    th, tw = template.shape[:2]

    if th < (scale * 8) or tw < (scale * 8): 
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc
        
    new_w, new_h = w // scale, h // scale
    new_tw, new_th = tw // scale, th // scale
    
    if new_w < new_tw or new_h < new_th or new_tw < 1 or new_th < 1:
         res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
         _, max_val, _, max_loc = cv2.minMaxLoc(res)
         return max_val, max_loc

    # downscale
    small_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_template = cv2.resize(template, (new_tw, new_th), interpolation=cv2.INTER_AREA)
    
    # coarse Match
    res = cv2.matchTemplate(small_img, small_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    # Refine Search Region
    top_left_x = max_loc[0] * scale
    top_left_y = max_loc[1] * scale
    padding = 30 
    start_x = max(0, top_left_x - padding)
    start_y = max(0, top_left_y - padding)
    end_x = min(w, top_left_x + tw + padding)
    end_y = min(h, top_left_y + th + padding)
    
    if (end_x - start_x) < tw or (end_y - start_y) < th:
         res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
         _, max_val, _, max_loc = cv2.minMaxLoc(res)
         return max_val, max_loc

    roi = img[start_y:end_y, start_x:end_x]
    
    # Fine Match
    res_fine = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val_fine, _, max_loc_fine = cv2.minMaxLoc(res_fine)
    
    final_x = start_x + max_loc_fine[0]
    final_y = start_y + max_loc_fine[1]
    
    return max_val_fine, (final_x, final_y)

class TemplateDetector:
    def __init__(self):
        self.templates = {}
        self.is_ready = False
        self.extract_templates()

    def extract_templates(self):
        print("Loading annotations and extracting templates")
        
        if not os.path.exists(ANNOTATIONS_FILE):
            print(f"[Detector] ERROR: File {ANNOTATIONS_FILE} tidak ditemukan!")
            return

        try:
            with open(ANNOTATIONS_FILE, 'r') as f:
                annotations = json.load(f)
            
            count = 0
            for filename, annots in annotations.items():
                if filename.startswith("test_"): continue 
                path = os.path.join(TRAIN_DIR, filename)
                
                if not os.path.exists(path):
                    path = os.path.join(BASE_DIR, "images", "train", os.path.basename(filename))

                if not os.path.exists(path):
                    continue
                    
                img = cv2.imread(path)
                if img is None: continue
                
                for label, rect in annots.items():
                    if rect is None: continue
                    x, y, w, h = rect
                    
                    if w <= 0 or h <= 0: continue
                    
                    # crop template
                    template = img[y:y+h, x:x+w]
                    if template.size == 0: continue
                    
                    if label not in self.templates: self.templates[label] = []
                    self.templates[label].append(template)
                count += 1
            
            print(f"Extracted templates from {count} images.")
            if self.templates:
                self.is_ready = True
                print("System Ready.")
                
        except Exception as e:
            print(f"Error Init: {e}")

    def detect_roi(self, img):
        if not self.is_ready or img is None: return {}, img
        TARGET_WIDTH = 1280
        TARGET_HEIGHT = 720
        h_orig, w_orig = img.shape[:2]
        scale_factor = 1.0
        img_processing = img.copy()
        
        # resize logic
        if w_orig > TARGET_WIDTH or h_orig > TARGET_HEIGHT:
            scale_w = TARGET_WIDTH / w_orig
            scale_h = TARGET_HEIGHT / h_orig
            scale_factor = min(scale_w, scale_h)
            new_w = int(w_orig * scale_factor)
            new_h = int(h_orig * scale_factor)
            img_processing = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        results = {}
        
        # find template
        for label, template_list in self.templates.items():
            best_score = -1
            best_box = None
            
            for template in template_list:
                th, tw = template.shape[:2]
                ih, iw = img_processing.shape[:2]
                if th >= ih or tw >= iw: continue 
                max_val, max_loc = optimized_template_match(img_processing, template)
                if max_val > best_score:
                    best_score = max_val
                    best_box = [max_loc[0], max_loc[1], tw, th]
            
            if best_box:
                # Scale back
                x = int(best_box[0] / scale_factor)
                y = int(best_box[1] / scale_factor)
                w = int(best_box[2] / scale_factor)
                h = int(best_box[3] / scale_factor)
                # Crop ROI 
                crop = img[y:y+h, x:x+w]
                results[label] = crop
                
                # Draw rectangle
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return results, img


# OCR Engine Initialization
def init_ocr_engine():
    custom_model_path = os.path.join(BASE_DIR, "inference_model_mobile")
    dict_path = os.path.join(BASE_DIR, "ppocr_keys_v1.txt")
    
    print(f"Loading Custom OCR Model, {custom_model_path}")
    
    if not os.path.exists(custom_model_path):
        print(f"Folder {custom_model_path} not found, using default model.")
        return PaddleOCR(use_angle_cls=False, lang='ch', show_log=False)

    return PaddleOCR(
        rec_model_dir=custom_model_path,
        rec_char_dict_path=dict_path,
        det=False,                       
        cls=False,                       
        use_angle_cls=False,                   
        show_log=False
    )

def read_text_from_crop(ocr, image):
    if image is None or image.size == 0: return "-"
    try:
        result = ocr.ocr(image, det=False, cls=False, rec=True)
        if result and result[0]:
            text, conf = result[0][0]
            clean_txt = re.sub(r'[^0-9.]', '', text).strip('.')
            if clean_txt and conf > 0.5:
                return clean_txt
    except: pass
    return "-"

#  Cloud Manager + UI
class GoogleManager:
    def __init__(self):
        self.scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']
        self.creds = None
        self.drive_service = None
        self.client = None
    
    def connect(self):
        try:
            token_path = os.path.join(BASE_DIR, 'token.pickle')
            if os.path.exists(token_path):
                with open(token_path, 'rb') as token: self.creds = pickle.load(token)
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                else:
                    if not os.path.exists(CLIENT_SECRET_FILE): return False
                    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, self.scope)
                    self.creds = flow.run_local_server(port=0)
                with open(token_path, 'wb') as token: pickle.dump(self.creds, token)
            self.drive_service = build('drive', 'v3', credentials=self.creds)
            self.client = gspread.authorize(self.creds)
            print("Google Cloud Connected.")
            return True
        except Exception as e:
            print(f"Gagal koneksi Google: {e}")
            return False

    def upload_image(self, file_path):
        try:
            meta = {'name': os.path.basename(file_path), 'parents': [DRIVE_FOLDER_ID]}
            media = MediaFileUpload(file_path, mimetype='image/jpeg')
            file = self.drive_service.files().create(body=meta, media_body=media, fields='id, webViewLink').execute()
            self.drive_service.permissions().create(fileId=file.get('id'), body={'type': 'anyone', 'role': 'reader'}).execute()
            return file.get('webViewLink')
        except: return None

    def append_data(self, *args):
        try:
            self.client.open_by_key(SPREADSHEET_ID).sheet1.append_row(list(args))
            return True
        except: return False

class ConfirmDialog(QtWidgets.QDialog, Ui_Confirm):
    def __init__(self, parent=None, image_path="", lokasi="", datetime_str="", temp_val="", hum_val=""):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.label_4.setText(lokasi); self.label_5.setText(datetime_str)
        self.temp.setText(temp_val); self.hum.setText(hum_val)
        
        self.img_label = QtWidgets.QLabel(self.frame_3)
        self.img_label.setGeometry(0, 0, self.frame_3.width(), self.frame_3.height())
        self.img_label.setScaledContents(False)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background-color: black;")
        
        if os.path.exists(image_path):
            pixmap = QtGui.QPixmap(image_path)
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.pushButton.clicked.connect(self.accept)
        self.pushButton_2.clicked.connect(self.reject)

class UserMainWindow(QtWidgets.QWidget, Ui_Capture):
    finished_signal = pyqtSignal()

    def __init__(self, received_location_id="Unknown"):
        super().__init__()
        self.setupUi(self)
        self.received_location = received_location_id
        
        # camera
        self.timer_camera = QTimer()
        self.timer_camera.timeout.connect(self.update_frame)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280); self.cap.set(4, 720)
        self.video_label = QtWidgets.QLabel(self.frame_3)
        self.video_label.setGeometry(0, 0, self.frame_3.width(), self.frame_3.height())
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.timer_camera.start(30)
        
        # clock
        self.timer_clock = QTimer()
        self.timer_clock.timeout.connect(self.update_clock)
        self.timer_clock.start(1000)
        self.update_clock()
        self.pushButton.clicked.connect(self.capture_image)
        
        self.g_manager = GoogleManager()
        threading.Thread(target=self.g_manager.connect).start()

        self.detector = TemplateDetector() 
        self.ocr_engine = None
        threading.Thread(target=self.load_ai_bg).start()

    def load_ai_bg(self):
        try:
            self.ocr_engine = init_ocr_engine()
            print("[System] OCR Engine Ready!")
        except Exception as e:
            print(f"[System] Gagal Init OCR: {e}")

    def update_clock(self):
        now = datetime.now()
        self.label.setText(now.strftime("%d/%m/%Y"))
        self.label_4.setText(now.strftime("%H:%M"))

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                qt_img = QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
                self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_img).scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            current_dt = datetime.now().strftime("%d/%m/%Y - %H:%M")
            val_temp, val_hum = "-", "-"
            
            # ROI detection & OCR
            if self.detector.is_ready:
                print("Searching Template ROI")
                try:
                    crops, vis_frame = self.detector.detect_roi(frame.copy())
                    
                    filename = "temp_capture.jpg"
                    cv2.imwrite(filename, vis_frame)
                    
                    if 'Temperature' in crops:
                        val_temp = read_text_from_crop(self.ocr_engine, crops['Temperature']) + " °C"
                    elif 'temp' in crops: # jaga2 beda label
                        val_temp = read_text_from_crop(self.ocr_engine, crops['temp']) + " °C"
                        
                    if 'Humidity' in crops:
                        val_hum = read_text_from_crop(self.ocr_engine, crops['Humidity']) + " %"
                    elif 'humidity' in crops:
                        val_hum = read_text_from_crop(self.ocr_engine, crops['humidity']) + " %"
                    elif 'hum' in crops:
                        val_hum = read_text_from_crop(self.ocr_engine, crops['hum']) + " %"
                    print(f"Hasil: {val_temp} | {val_hum}")
                    
                except Exception as e:
                    print(f"Error Detection: {e}")
                    filename = "temp_capture.jpg"
                    cv2.imwrite(filename, frame)
            else:
                print("Detector not ready.")
                filename = "temp_capture.jpg"
                cv2.imwrite(filename, frame)

            dialog = ConfirmDialog(self, filename, self.received_location, current_dt, val_temp, val_hum)
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                threading.Thread(target=self.process_upload, args=(filename, self.received_location, current_dt, val_temp, val_hum)).start()
                self.close_and_return()
            else:
                if os.path.exists(filename): os.remove(filename)

    def process_upload(self, filename, lokasi, datetime_str, suhu, kelembaban):
        link = self.g_manager.upload_image(filename)
        if link:
            parts = datetime_str.split(' - ')
            self.g_manager.append_data(parts[0], parts[1], lokasi, suhu, kelembaban, link)
        if os.path.exists(filename): os.remove(filename)

    def close_and_return(self):
        self.timer_camera.stop()
        if self.cap: self.cap.release()
        self.close(); self.finished_signal.emit() 
    def closeEvent(self, event):
        if self.cap: self.cap.release()
        event.accept()

# Launcher
class QRScannerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan QR Code")
        self.resize(640, 480)
        self.layout = QVBoxLayout(); self.setLayout(self.layout)
        self.video_label = QLabel("Initializing...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.layout.addWidget(self.video_label)
        self.detected_data = None
        self.cap = cv2.VideoCapture(0)
        self.detector = cv2.QRCodeDetector()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame); self.timer.start(30)
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        data, bbox, _ = self.detector.detectAndDecode(frame)
        if bbox is not None:
             if data: self.detected_data = data; self.accept(); return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(self.video_label.size(), Qt.KeepAspectRatio))
    def closeEvent(self, event): self.timer.stop(); self.cap.release(); event.accept()

class HygroScanApp(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join("ui", "mainwindow.ui")
        if not os.path.exists(ui_path): ui_path = "mainwindow.ui"
        if os.path.exists(ui_path): uic.loadUi(ui_path, self)
        else: sys.exit()
        self.btn_scan_qr = self.findChild(QtWidgets.QPushButton, "btn_scan_qr")
        self.btn_scan_rfid = self.findChild(QtWidgets.QPushButton, "btn_scan_rfid")
        if self.btn_scan_qr: self.btn_scan_qr.clicked.connect(self.run_qr_scan)
        if self.btn_scan_rfid: self.btn_scan_rfid.clicked.connect(self.run_rfid_scan)
    def run_qr_scan(self):
        d = QRScannerDialog(self)
        if d.exec_() == QDialog.Accepted: self.switch_to_user_app(d.detected_data)
    def run_rfid_scan(self):
        t, o = QInputDialog.getText(self, "Scan RFID", "Tap RFID Tag:")
        if o and t: self.switch_to_user_app(t)
    def switch_to_user_app(self, loc):
        self.hide(); self.w = UserMainWindow(loc)
        self.w.finished_signal.connect(self.on_finished); self.w.show()
    def on_finished(self): self.show(); QMessageBox.information(self, "Info", "The data has been saved successfully.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HygroScanApp()
    window.show()
    sys.exit(app.exec_())