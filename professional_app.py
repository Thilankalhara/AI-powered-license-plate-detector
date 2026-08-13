"""
License Plate Detector - Professional GUI
Beautiful dark-themed interface with modern design and live video preview
"""

import sys
import os

# Fix PyTorch DLL loading issue on Windows
if hasattr(os, 'add_dll_directory'):
    try:
        import torch
        torch_dir = os.path.dirname(torch.__file__)
        lib_dir = os.path.join(torch_dir, 'lib')
        if os.path.exists(lib_dir):
            os.add_dll_directory(lib_dir)
    except Exception:
        pass

import csv
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QTableWidget, QTableWidgetItem, QFileDialog, 
                             QProgressBar, QListWidget, QListWidgetItem, QMessageBox, 
                             QTabWidget, QSplitter, QTextEdit, QFrame)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QColor, QPixmap, QImage
from PyQt5.QtWidgets import QStyleFactory
from difflib import SequenceMatcher

# ==================== CONFIGURATION ====================
PLATE_CSV = "blacklist.csv"
DETECTED_LOG = "detected_vehicles.csv"
MIN_MATCH_RATIO = 0.75

def sanitize_plate_text(text):
    """Sanitize plate text to prevent injection attacks"""
    if not isinstance(text, str):
        return ""
    # Remove special characters, keep only alphanumeric and common plate chars
    sanitized = "".join(c for c in text.upper() if c.isalnum() or c in "- ")
    return sanitized.strip()

def safe_file_path(base_dir, filename):
    """Create safe file path preventing directory traversal"""
    base = Path(base_dir).resolve()
    # Remove path separators and special chars from filename
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._-")
    target = (base / safe_name).resolve()
    # Ensure target is within base directory
    if not str(target).startswith(str(base)):
        raise ValueError("Invalid file path")
    return target

# ==================== DETECTION WORKER ====================
class PlateDetectionWorker(QThread):
    """Worker thread for video processing without freezing GUI"""
    progress = pyqtSignal(int)
    frame_ready = pyqtSignal(np.ndarray)
    plate_detected = pyqtSignal(str, float, np.ndarray, bool)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, video_path, model, ocr_reader, blacklist):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.ocr_reader = ocr_reader
        self.blacklist = blacklist
        self.is_running = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit("Cannot open video file")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            processed_plates = set()
            
            # Create output directory for matched vehicles
            output_dir = Path("detected_plates").resolve()
            output_dir.mkdir(exist_ok=True)

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))
                display_frame, detected_plates = self.detect_and_draw(frame)

                for plate_text, confidence, plate_region, is_blacklisted in detected_plates:
                    normalized_plate = self.normalize_plate(plate_text)
                    
                    if normalized_plate not in processed_plates and is_blacklisted:
                        # Save full vehicle image with bounding box
                        self.save_vehicle_image(display_frame, plate_text, confidence, output_dir)
                        self.plate_detected.emit(plate_text, confidence, plate_region, is_blacklisted)
                        processed_plates.add(normalized_plate)

                self.frame_ready.emit(display_frame)
                frame_count += 1
                progress = min(99, int((frame_count / total_frames) * 100)) if total_frames > 0 else 0
                self.progress.emit(progress)

            cap.release()
            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
    
    def save_vehicle_image(self, frame, plate_text, confidence, output_dir):
        """Save full vehicle image with bounding box and plate info"""
        try:
            # Create a copy of the frame with detection info
            save_frame = frame.copy()
            
            # Add timestamp and plate info on the image
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(save_frame, f"Plate: {plate_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(save_frame, f"Confidence: {confidence*100:.1f}%", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(save_frame, f"Time: {timestamp}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(save_frame, "BLACKLISTED MATCH", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save image
            safe_plate = "".join(c if c.isalnum() else "_" for c in plate_text)
            filename = f"vehicle_{safe_plate}_{datetime.now().strftime('%H%M%S%f')}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), save_frame)
            print(f"[VEHICLE SAVED] {filepath}")
            
        except Exception as e:
            print(f"[SAVE ERROR] {e}")

    def detect_and_draw(self, frame):
        """Detect plates and draw bounding boxes with labels"""
        try:
            results = self.model(frame, conf=0.5)
            detected_plates = []
            display_frame = frame.copy()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    plate_region = frame[y1:y2, x1:x2]
                    
                    try:
                        ocr_result = self.ocr_reader.readtext(plate_region, detail=0)
                        plate_text = "".join(ocr_result).strip()
                        
                        if plate_text:
                            confidence = float(box.conf[0])
                            normalized_plate = self.normalize_plate(plate_text)
                            is_blacklisted = any(
                                self.is_similar(normalized_plate, bl, MIN_MATCH_RATIO)
                                for bl in self.blacklist
                            )
                            
                            if is_blacklisted:
                                color = (0, 0, 255)
                                label = f"{plate_text} [{confidence*100:.1f}%] MATCH"
                            else:
                                color = (0, 255, 0)
                                label = f"{plate_text} [{confidence*100:.1f}%]"
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            detected_plates.append((plate_text, confidence, plate_region, is_blacklisted))
                    except Exception as e:
                        print(f"[OCR ERROR] {e}")
                        pass

            return display_frame, detected_plates
        except Exception as e:
            print(f"Detection error: {e}")
            return frame, []

    @staticmethod
    def normalize_plate(plate_text):
        """Normalize plate for comparison"""
        return plate_text.upper().replace(" ", "").replace("-", "")
    
    @staticmethod
    def enhance_plate_image(plate_region):
        """Enhance plate image for better visibility"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            # Apply bilateral filter to reduce noise while keeping edges
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            # Apply adaptive thresholding
            enhanced = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            # Convert back to BGR for saving
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            return enhanced_bgr
        except Exception as e:
            print(f"[ENHANCE ERROR] {e}")
            return plate_region

    @staticmethod
    def is_similar(plate1, plate2, threshold):
        """Check if two plates are similar"""
        ratio = SequenceMatcher(None, plate1, plate2).ratio()
        return ratio >= threshold

# ==================== DARK THEME STYLESHEET ====================
DARK_STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
    color: #ffffff;
}

QWidget {
    background-color: #1e1e1e;
    color: #ffffff;
}

QLabel {
    color: #ffffff;
}

QLineEdit {
    background-color: #2d2d2d;
    color: #ffffff;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 5px;
    font-size: 12px;
}

QLineEdit:focus {
    border: 2px solid #0d47a1;
    background-color: #2d2d2d;
}

QPushButton {
    background-color: #0d47a1;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #1565c0;
}

QPushButton:pressed {
    background-color: #0a3a7f;
}

QPushButton:disabled {
    background-color: #404040;
    color: #808080;
}

QPushButton#startBtn {
    background-color: #00a86b;
}

QPushButton#startBtn:hover {
    background-color: #00c77d;
}

QPushButton#stopBtn {
    background-color: #d32f2f;
}

QPushButton#stopBtn:hover {
    background-color: #f44336;
}

QPushButton#clearBtn {
    background-color: #757575;
}

QPushButton#clearBtn:hover {
    background-color: #9e9e9e;
}

QTableWidget {
    background-color: #2d2d2d;
    color: #ffffff;
    border: 1px solid #404040;
    gridline-color: #404040;
    border-radius: 4px;
}

QTableWidget::item {
    padding: 5px;
    border: none;
}

QTableWidget::item:selected {
    background-color: #0d47a1;
}

QHeaderView::section {
    background-color: #404040;
    color: #ffffff;
    padding: 5px;
    border: none;
    font-weight: bold;
}

QListWidget {
    background-color: #2d2d2d;
    color: #ffffff;
    border: 1px solid #404040;
    border-radius: 4px;
}

QListWidget::item {
    padding: 8px;
    border: none;
}

QListWidget::item:selected {
    background-color: #0d47a1;
    border-radius: 2px;
}

QListWidget::item:hover {
    background-color: #353535;
}

QProgressBar {
    background-color: #404040;
    border: 1px solid #555555;
    border-radius: 4px;
    color: #ffffff;
    text-align: center;
    min-height: 20px;
}

QProgressBar::chunk {
    background-color: #00a86b;
    border-radius: 2px;
}

QTabWidget::pane {
    border: 1px solid #404040;
}

QTabBar::tab {
    background-color: #2d2d2d;
    color: #ffffff;
    padding: 8px 20px;
    border: 1px solid #404040;
}

QTabBar::tab:selected {
    background-color: #0d47a1;
    border-bottom: 2px solid #00a86b;
}

QTabBar::tab:hover {
    background-color: #353535;
}

QSplitter::handle {
    background-color: #404040;
}

QFrame {
    background-color: #1e1e1e;
    border: none;
}

QTextEdit {
    background-color: #2d2d2d;
    color: #b0b0b0;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 8px;
    font-family: Courier New;
    font-size: 11px;
}

QScrollBar:vertical {
    background-color: #2d2d2d;
    width: 12px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #555555;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #757575;
}
"""

# ==================== GLOBAL EXCEPTION HANDLER ====================
def exception_hook(exc_type, exc_value, exc_traceback):
    """Catch unhandled exceptions and show error message"""
    import traceback
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"Unhandled exception:\n{error_msg}")
    
    # Show error in GUI if app exists
    app = QApplication.instance()
    if app:
        QMessageBox.critical(None, "Unexpected Error", f"An unexpected error occurred:\n{str(exc_value)}\n\nPlease restart the application.")
    
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_hook

# ==================== MAIN APPLICATION ====================
class LicensePlateDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("License Plate Detector - AI Detection System")
        self.setGeometry(50, 50, 1400, 800)
        self.setMinimumSize(1200, 700)
        
        self.blacklist = self.load_blacklist()
        self.detected_vehicles = []
        self.video_path = None
        self.is_scanning = False
        self.model = None
        self.ocr_reader = None
        self.worker = None
        
        # Fix DLL loading issues on Windows
        import os
        if hasattr(os, 'add_dll_directory'):
            torch_dir = os.path.dirname(__import__('torch').__file__)
            lib_dir = os.path.join(torch_dir, 'lib')
            if os.path.exists(lib_dir):
                os.add_dll_directory(lib_dir)
        
        # Apply dark theme
        self.setStyleSheet(DARK_STYLESHEET)
        
        # Create UI
        self.init_ui()
        
        # Set window icon (optional)
        self.show_status("System Ready", "#00a86b")
        
        # Raise window to front
        self.raise_()
        self.activateWindow()
        
    def init_ui(self):
        """Initialize the main UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # ==================== LEFT PANEL: VIDEO DETECTION ====================
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)
        
        # Title
        title_label = QLabel("Live Video Detection Stream")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        left_layout.addWidget(title_label)
        
        # Video file selector button
        select_video_btn = QPushButton("SELECT VIDEO FILE")
        select_video_btn.setMinimumHeight(35)
        select_video_btn.setFont(QFont("Arial", 11, QFont.Bold))
        select_video_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565c0;
                color: #ffffff;
                border: 2px solid #0d47a1;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1976d2;
                border: 2px solid #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """)
        select_video_btn.clicked.connect(self.select_video_file)
        left_layout.addWidget(select_video_btn)
        
        # Video source indicator
        source_layout = QHBoxLayout()
        source_label = QLabel("Source:")
        source_label.setStyleSheet("font-weight: bold; color: #ffffff;")
        self.source_value = QLabel("No video loaded")
        self.source_value.setStyleSheet("color: #ff9800; font-weight: bold;")
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_value, 1)
        source_layout.addStretch()
        left_layout.addLayout(source_layout)
        
        # Video display area - LIVE PREVIEW
        self.video_display = QLabel()
        self.video_display.setMinimumHeight(400)
        self.video_display.setStyleSheet("border: 2px solid #404040; background-color: #000; border-radius: 4px;")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setText("Live stream feed will appear here")
        left_layout.addWidget(self.video_display)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("START DETECTION SCAN")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.start_btn.clicked.connect(self.start_detection)
        
        self.stop_btn = QPushButton("STOP SCAN")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_detection)
        
        button_layout.addWidget(self.start_btn, 2)
        button_layout.addWidget(self.stop_btn, 1)
        left_layout.addLayout(button_layout)
        
        # Export button
        export_btn = QPushButton("Export Audit Report (CSV/JSON)")
        export_btn.setMinimumHeight(35)
        export_btn.setFont(QFont("Arial", 10, QFont.Bold))
        export_btn.clicked.connect(self.export_results)
        left_layout.addWidget(export_btn)
        
        # Progress bar - VISIBLE
        progress_label = QLabel("Progress:")
        left_layout.addWidget(progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        left_layout.addWidget(self.progress_bar)
        
        # ==================== RIGHT PANEL: BLACKLIST MANAGEMENT ====================
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Detection Results
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        results_title = QLabel("Matched Blacklist Plates")
        results_title.setFont(QFont("Arial", 12, QFont.Bold))
        results_layout.addWidget(results_title)
        
        results_btn_layout = QHBoxLayout()
        export_images_btn = QPushButton("Export Matched Images")
        export_images_btn.setMinimumHeight(30)
        export_images_btn.clicked.connect(self.export_matched_images)
        
        clear_results_btn = QPushButton("Clear Results")
        clear_results_btn.setObjectName("clearBtn")
        clear_results_btn.setMinimumHeight(30)
        clear_results_btn.clicked.connect(self.clear_results)
        
        results_btn_layout.addWidget(export_images_btn)
        results_btn_layout.addWidget(clear_results_btn)
        results_layout.addLayout(results_btn_layout)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Image", "Plate ID", "Confidence", "Timestamp", "Status"]
        )
        self.results_table.horizontalHeader().setStretchLastSection(False)
        self.results_table.setColumnWidth(0, 100)
        self.results_table.setColumnWidth(1, 100)
        self.results_table.setColumnWidth(2, 80)
        self.results_table.setColumnWidth(3, 120)
        self.results_table.setColumnWidth(4, 100)
        self.results_table.verticalHeader().setDefaultSectionSize(70)
        results_layout.addWidget(self.results_table)
        
        self.tabs.addTab(results_tab, "Matched Blacklist Plates")
        
        # Tab 2: Blacklist Manager
        blacklist_tab = QWidget()
        blacklist_layout = QVBoxLayout(blacklist_tab)
        
        # Quick Add Plate Section (PROMINENT)
        quick_add_section = QFrame()
        quick_add_section.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 2px solid #0d47a1;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        quick_add_layout = QVBoxLayout(quick_add_section)
        quick_add_layout.setContentsMargins(10, 10, 10, 10)
        
        quick_add_title = QLabel("QUICK ADD PLATE")
        quick_add_title_font = QFont()
        quick_add_title_font.setPointSize(11)
        quick_add_title_font.setBold(True)
        quick_add_title.setFont(quick_add_title_font)
        quick_add_title.setStyleSheet("color: #00a86b;")
        quick_add_layout.addWidget(quick_add_title)
        
        quick_add_input_layout = QHBoxLayout()
        self.plate_input = QLineEdit()
        self.plate_input.setPlaceholderText("Enter plate ID (e.g., A1OO, ABC123, AB12CD, etc.)")
        self.plate_input.setMinimumHeight(40)
        self.plate_input.setFont(QFont("Arial", 11))
        self.plate_input.returnPressed.connect(self.add_to_blacklist)
        
        add_btn = QPushButton("ADD PLATE")
        add_btn.setMinimumHeight(40)
        add_btn.setMinimumWidth(120)
        add_btn.setFont(QFont("Arial", 11, QFont.Bold))
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #00a86b;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #00c77d;
            }
            QPushButton:pressed {
                background-color: #007d52;
            }
        """)
        add_btn.clicked.connect(self.add_to_blacklist)
        
        quick_add_input_layout.addWidget(self.plate_input, 1)
        quick_add_input_layout.addWidget(add_btn, 0)
        quick_add_layout.addLayout(quick_add_input_layout)
        
        blacklist_layout.addWidget(quick_add_section)
        
        # Search bar
        search_layout = QHBoxLayout()
        search_icon = QLabel("Search Plates:")
        search_label = QLabel("Search Plates:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search plates (e.g., A1OO)...")
        self.search_input.textChanged.connect(self.filter_blacklist)
        
        clear_search_btn = QPushButton("Clear")
        clear_search_btn.setObjectName("clearBtn")
        clear_search_btn.setMaximumWidth(70)
        clear_search_btn.clicked.connect(self.clear_search)
        
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input, 1)
        search_layout.addWidget(clear_search_btn)
        blacklist_layout.addLayout(search_layout)
        
        # Blacklist display
        self.blacklist_list = QListWidget()
        self.blacklist_list.itemDoubleClicked.connect(self.edit_plate)
        blacklist_layout.addWidget(self.blacklist_list)
        
        # Management buttons
        mgmt_layout = QHBoxLayout()
        
        import_btn = QPushButton("Import")
        import_btn.clicked.connect(self.import_blacklist)
        
        export_bl_btn = QPushButton("Export")
        export_bl_btn.clicked.connect(self.export_blacklist)
        
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_selected)
        
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setObjectName("stopBtn")
        clear_all_btn.clicked.connect(self.clear_all_blacklist)
        
        mgmt_layout.addWidget(import_btn)
        mgmt_layout.addWidget(export_bl_btn)
        mgmt_layout.addWidget(delete_btn)
        mgmt_layout.addWidget(clear_all_btn)
        blacklist_layout.addLayout(mgmt_layout)
        
        self.tabs.addTab(blacklist_tab, "Blacklist Manager")
        
        right_layout.addWidget(self.tabs)
        
        # ==================== COMBINE PANELS ====================
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 60)
        splitter.setStretchFactor(1, 40)
        main_layout.addWidget(splitter)
        
        central_widget.setLayout(main_layout)
        
        # ==================== STATUS BAR ====================
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #2d2d2d;
                color: #ffffff;
                border-top: 1px solid #404040;
            }
        """)
        self.show_status("System Ready", "#00a86b")
        
        # Load blacklist on startup
        self.refresh_blacklist_display()
        
    def load_blacklist(self):
        """Load blacklist from CSV"""
        blacklist = {}  # Now dict: {plate: added_date}
        try:
            csv_path = safe_file_path(".", PLATE_CSV)
            if csv_path.exists():
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if row and len(row) >= 1:
                            plate = sanitize_plate_text(row[0])
                            date_added = row[1] if len(row) > 1 else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            blacklist[plate] = date_added
        except Exception as e:
            self.show_status(f"Failed to load blacklist: {str(e)}", "#ff5722")
        return blacklist
    
    def select_video_file(self):
        """Open file dialog to select video file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Video File",
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm);;All Files (*.*)"
            )
            
            if file_path:
                self.video_path = file_path
                file_name = Path(file_path).name
                self.source_value.setText(f"{file_name}")
                self.source_value.setStyleSheet("color: #00a86b; font-weight: bold;")
                self.show_status(f"Video loaded: {file_name}", "#00a86b")
                
                # Enable detection buttons
                self.start_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to select video:\n{str(e)}")
            self.show_status(f"Error selecting video: {str(e)}", "#ff5722")
    
    def add_to_blacklist(self):
        """Add plate to blacklist"""
        plate = sanitize_plate_text(self.plate_input.text())
        if not plate:
            self.show_status("Please enter a valid plate ID", "#ff5722")
            QMessageBox.warning(self, "Input Required", "Please enter a valid plate ID")
            return
        
        # Check if already exists
        if plate in self.blacklist:
            self.show_status(f"Plate {plate} already in blacklist", "#ff9800")
            QMessageBox.information(self, "Duplicate", f"Plate {plate} is already in the blacklist")
            return
        
        # Add to blacklist
        self.blacklist[plate] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_blacklist()
        
        self.plate_input.clear()
        self.refresh_blacklist_display()
        self.show_status(f"Added {plate} to blacklist ({len(self.blacklist)} total)", "#00a86b")
    
    def save_blacklist(self):
        """Save blacklist to CSV"""
        try:
            csv_path = safe_file_path(".", PLATE_CSV)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Plate ID", "Added Date"])
                for plate, date in sorted(self.blacklist.items()):
                    writer.writerow([plate, date])
        except Exception as e:
            self.show_status(f"Failed to save blacklist: {str(e)}", "#ff5722")
            QMessageBox.critical(self, "Save Error", f"Failed to save blacklist: {str(e)}")
    
    def refresh_blacklist_display(self):
        """Update blacklist display"""
        self.blacklist_list.clear()
        for i, (plate, date) in enumerate(sorted(self.blacklist.items()), 1):
            item = QListWidgetItem(f"{i:2d}. {plate:12s} ({date[:10]})")
            item.setFlags(item.flags() | Qt.ItemIsSelectable)
            self.blacklist_list.addItem(item)
        
        self.show_status(f"Blacklist: {len(self.blacklist)} plates", "#2196f3")
    
    def filter_blacklist(self):
        """Filter blacklist by search"""
        query = self.search_input.text().strip().upper()
        
        for i in range(self.blacklist_list.count()):
            item = self.blacklist_list.item(i)
            if query:
                item.setHidden(query not in item.text().upper())
            else:
                item.setHidden(False)
    
    def clear_search(self):
        """Clear search field"""
        self.search_input.clear()
    
    def delete_selected(self):
        """Delete selected plate from blacklist"""
        current_item = self.blacklist_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a plate to delete")
            return
        
        text = current_item.text()
        plate = text.split(". ")[1].split("(")[0].strip()
        
        reply = QMessageBox.question(self, "Confirm Delete", 
                                    f"Delete '{plate}' from blacklist?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.blacklist[plate]
            self.save_blacklist()
            self.refresh_blacklist_display()
            self.show_status(f"Deleted {plate}", "#ff5722")
    
    def clear_all_blacklist(self):
        """Clear entire blacklist"""
        if not self.blacklist:
            QMessageBox.warning(self, "Empty", "Blacklist is already empty")
            return
        
        reply = QMessageBox.question(self, "Confirm Clear All", 
                                    f"Delete all {len(self.blacklist)} plates?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.blacklist.clear()
            Path(PLATE_CSV).unlink(missing_ok=True)
            self.refresh_blacklist_display()
            self.show_status("Blacklist cleared", "#ff5722")
    
    def export_blacklist(self):
        """Export blacklist to CSV"""
        if not self.blacklist:
            QMessageBox.warning(self, "Empty", "Blacklist is empty")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Blacklist", PLATE_CSV, "CSV Files (*.csv);;JSON Files (*.json)"
        )
        if file_path:
            if file_path.endswith('.json'):
                import json
                with open(file_path, 'w') as f:
                    json.dump(self.blacklist, f, indent=2)
            else:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Plate ID", "Added Date"])
                    for plate, date in sorted(self.blacklist.items()):
                        writer.writerow([plate, date])
            
            self.show_status(f"Exported to {Path(file_path).name}", "#00a86b")
            QMessageBox.information(self, "Success", f"Blacklist exported to {Path(file_path).name}")
    
    def import_blacklist(self):
        """Import blacklist from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Blacklist", "", "CSV Files (*.csv);;JSON Files (*.json)"
        )
        if file_path:
            try:
                if file_path.endswith('.json'):
                    import json
                    with open(file_path, 'r') as f:
                        imported = json.load(f)
                        self.blacklist.update(imported)
                else:
                    with open(file_path, 'r') as f:
                        reader = csv.reader(f)
                        next(reader, None)  # Skip header
                        for row in reader:
                            if row and len(row) >= 1:
                                plate = row[0].upper()
                                date = row[1] if len(row) > 1 else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                self.blacklist[plate] = date
                
                self.save_blacklist()
                self.refresh_blacklist_display()
                self.show_status(f"Imported {len(self.blacklist)} plates", "#00a86b")
                QMessageBox.information(self, "Success", f"Imported {len(self.blacklist)} plates")
            except Exception as e:
                self.show_status(f"Import failed: {str(e)}", "#ff5722")
                QMessageBox.critical(self, "Import Error", str(e))
    
    def edit_plate(self, item):
        """Edit selected plate (double-click)"""
        text = item.text()
        plate = text.split(". ")[1].split("(")[0].strip()
        new_plate, ok = QMessageBox.getText(self, "Edit Plate", "New plate ID:", text=plate)
        
        if ok and new_plate:
            new_plate = new_plate.strip().upper()
            if new_plate != plate:
                date = self.blacklist[plate]
                del self.blacklist[plate]
                self.blacklist[new_plate] = date
                self.save_blacklist()
                self.refresh_blacklist_display()
                self.show_status(f"Updated {plate} -> {new_plate}", "#00a86b")
    
    def load_detection_engine(self):
        """Load YOLO and EasyOCR models"""
        try:
            import warnings
            warnings.filterwarnings("ignore")
            
            from ultralytics import YOLO
            import easyocr
            
            self.model = YOLO("yolov8n.pt")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            self.show_status("Detection engine loaded", "#00a86b")
            return True
        except ImportError as e:
            self.show_status(f"Engine load failed: {str(e)}", "#ff5722")
            QMessageBox.critical(self, "Error", 
                               f"Failed to load detection engine:\n{str(e)}\n\n"
                               f"Please ensure all dependencies are installed:\n"
                               f"pip install -r requirements.txt")
            return False
        except Exception as e:
            self.show_status(f"Engine error: {str(e)}", "#ff5722")
            QMessageBox.critical(self, "Error", 
                               f"Error loading detection engine:\n{str(e)}\n\n"
                               f"This may be a compatibility issue. Try reinstalling PyTorch:\n"
                               f"pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            return False
    
    def start_detection(self):
        """Start detection scan"""
        if not self.video_path:
            self.show_status("Please select a video file first!", "#ff5722")
            QMessageBox.warning(self, "No Video Selected", "Please select a video file first.")
            return
        
        # Load engine if not loaded
        if self.model is None or self.ocr_reader is None:
            if not self.load_detection_engine():
                return
        
        self.is_scanning = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.source_value.setText("SCANNING...")
        self.source_value.setStyleSheet("color: #ff5722; font-weight: bold;")
        self.video_display.setText("SCANNING FOR BLACKLISTED PLATES...")
        self.show_status("Scanning for blacklisted plates...", "#00a86b")
        
        # Clear previous results
        self.results_table.setRowCount(0)
        self.detected_vehicles = []
        self.progress_bar.setValue(0)
        
        # Start worker thread
        self.worker = PlateDetectionWorker(
            self.video_path, self.model, self.ocr_reader, list(self.blacklist.keys())
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.plate_detected.connect(self.on_plate_detected)
        self.worker.finished.connect(self.detection_finished)
        self.worker.error.connect(self.on_detection_error)
        self.worker.start()
    
    def stop_detection(self):
        """Stop detection scan and save matched plates/results"""
        self.is_scanning = False
        if self.worker and self.worker.isRunning():
            self.worker.is_running = False
            self.worker.wait()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.video_display.setText("Live stream feed will appear here")
        
        if self.video_path:
            file_name = Path(self.video_path).name
            self.source_value.setText(file_name)
            self.source_value.setStyleSheet("color: #00a86b; font-weight: bold;")
        else:
            self.source_value.setText("No video loaded")
            self.source_value.setStyleSheet("color: #ff9800; font-weight: bold;")
        
        # Auto-save matched plates if any were found
        if self.detected_vehicles:
            try:
                plate_dir = Path("detected_plates").resolve()
                plate_dir.mkdir(exist_ok=True)
                
                # Save results CSV
                import csv
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = plate_dir / f"matched_plates_{timestamp}.csv"
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Plate", "Confidence", "Timestamp", "Status"])
                    for vehicle in self.detected_vehicles:
                        writer.writerow([
                            vehicle["plate"],
                            f"{vehicle['confidence']*100:.2f}%",
                            vehicle["timestamp"],
                            vehicle["status"]
                        ])
                
                self.show_status(f"Detection stopped. Saved {len(self.detected_vehicles)} matched plates", "#00a86b")
                QMessageBox.information(self, "Detection Stopped", 
                                      f"Found {len(self.detected_vehicles)} blacklisted plate matches.\n\n"
                                      f"Results saved to:\n{csv_path}\n\n"
                                      f"Plate images saved to:\n{plate_dir}")
            except Exception as e:
                self.show_status(f"Save failed: {str(e)}", "#ff5722")
                QMessageBox.critical(self, "Save Error", f"Failed to save results:\n{str(e)}")
        else:
            self.show_status("Detection stopped. No blacklisted plates matched.", "#ff9800")
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def update_frame(self, frame):
        """Update video preview with current frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_display.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def on_plate_detected(self, plate_text, confidence, plate_region, is_blacklisted):
        """Handle detected plate - add to results with thumbnail"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"[PLATE DETECTED] Text: '{plate_text}', Confidence: {confidence*100:.1f}%, Matched: {is_blacklisted}")
        
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Save matched plate images to disk
        plate_dir = Path("detected_plates").resolve()
        plate_dir.mkdir(exist_ok=True)
        safe_plate = "".join(c if c.isalnum() else "_" for c in plate_text)
        base_name = f"{safe_plate}_{datetime.now().strftime('%H%M%S%f')}"
        
        # Save enhanced plate crop
        plate_crop_path = plate_dir / f"{base_name}_crop.jpg"
        vehicle_context_path = plate_dir / f"{base_name}_vehicle.jpg"
        
        try:
            # Enhance and save plate crop
            enhanced_plate = self.enhance_plate_image(plate_region)
            cv2.imwrite(str(plate_crop_path), enhanced_plate)
            print(f"[PLATE CROP SAVED] {plate_crop_path}")
            
            # Save full vehicle context image (plate_region is the car crop)
            cv2.imwrite(str(vehicle_context_path), plate_region)
            print(f"[VEHICLE IMAGE SAVED] {vehicle_context_path}")
            
        except Exception as e:
            print(f"[IMAGE ERROR] Failed to save: {e}")
        
        # Create thumbnail from plate crop
        thumb_label = QLabel("No Image")
        thumb_label.setAlignment(Qt.AlignCenter)
        thumb_label.setStyleSheet("color: #808080; font-size: 10px;")
        
        if plate_crop_path.exists() and plate_crop_path.stat().st_size > 0:
            try:
                pixmap = QPixmap(str(plate_crop_path))
                if not pixmap.isNull():
                    thumb = pixmap.scaled(80, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    thumb_label.setPixmap(thumb)
                    thumb_label.setStyleSheet("")
                else:
                    thumb_label.setText("No Image")
            except Exception as e:
                thumb_label.setText("Error")
        
        self.results_table.setCellWidget(row, 0, thumb_label)
        
        self.results_table.setItem(row, 1, QTableWidgetItem(plate_text))
        
        conf_item = QTableWidgetItem(f"{confidence*100:.2f}%")
        self.results_table.setItem(row, 2, conf_item)
        
        self.results_table.setItem(row, 3, QTableWidgetItem(timestamp))
        
        if is_blacklisted:
            status_item = QTableWidgetItem("MATCHED")
            status_item.setBackground(QColor(255, 100, 100))
            status_item.setForeground(QColor(255, 255, 255))
            self.detected_vehicles.append({
                "plate": plate_text,
                "confidence": confidence,
                "timestamp": timestamp,
                "status": "MATCHED"
            })
        else:
            status_item = QTableWidgetItem("OK")
            status_item.setBackground(QColor(100, 255, 100))
            self.detected_vehicles.append({
                "plate": plate_text,
                "confidence": confidence,
                "timestamp": timestamp,
                "status": "OK"
            })
        
        self.results_table.setItem(row, 4, status_item)
        
        # Auto scroll to latest
        self.results_table.scrollToBottom()
        
        # Save cropped plate image
        try:
            plate_dir = Path("detected_plates").resolve()
            plate_dir.mkdir(exist_ok=True)
            safe_plate = "".join(c if c.isalnum() else "_" for c in plate_text)
            file_name = f"{safe_plate}_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(plate_dir / file_name), plate_region)
        except Exception as e:
            print(f"Failed to save plate image: {e}")
    
    def detection_finished(self):
        """Handle detection completion"""
        self.is_scanning = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.show_status(f"Detection Complete! Found {len(self.detected_vehicles)} blacklisted matches", "#00a86b")
    
    def on_detection_error(self, error_msg):
        """Handle detection error"""
        self.is_scanning = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "Detection Error", error_msg)
        self.show_status(f"Error: {error_msg}", "#ff5722")
    
    def export_results(self):
        """Export detection results with matched plate images"""
        if self.results_table.rowCount() == 0:
            QMessageBox.warning(self, "No Results", "No detection results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Audit Report", DETECTED_LOG, "CSV Files (*.csv);;JSON Files (*.json)"
        )
        if file_path:
            try:
                headers = ["Image", "Plate ID", "Confidence", "Timestamp", "Status"]
                base_path = Path(file_path)
                
                # Create images directory next to export file
                images_dir = base_path.parent / f"{base_path.stem}_images"
                images_dir.mkdir(exist_ok=True)
                
                exported_images = []
                data_rows = []
                plate_dir = Path("detected_plates").resolve()
                
                for row in range(self.results_table.rowCount()):
                    plate_item = self.results_table.item(row, 1)
                    conf_item = self.results_table.item(row, 2)
                    time_item = self.results_table.item(row, 3)
                    status_item = self.results_table.item(row, 4)
                    
                    plate_text = plate_item.text() if plate_item else ""
                    confidence = conf_item.text() if conf_item else ""
                    timestamp = time_item.text() if time_item else ""
                    status = status_item.text() if status_item else ""
                    
                    # Find saved plate image
                    image_filename = ""
                    if plate_text and plate_dir.exists():
                        safe_plate = "".join(c if c.isalnum() else "_" for c in plate_text)
                        matching_images = sorted(plate_dir.glob(f"{safe_plate}_*.jpg"))
                        if matching_images:
                            src_image = matching_images[0]
                            dst_image = images_dir / src_image.name
                            import shutil
                            shutil.copy2(src_image, dst_image)
                            image_filename = dst_image.name
                            exported_images.append(dst_image)
                    
                    row_data = {
                        "Image": image_filename,
                        "Plate ID": plate_text,
                        "Confidence": confidence,
                        "Timestamp": timestamp,
                        "Status": status
                    }
                    data_rows.append(row_data)
                
                # Write data file
                if file_path.endswith('.json'):
                    import json
                    with open(file_path, 'w') as f:
                        json.dump(data_rows, f, indent=2)
                else:
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                        for row_data in data_rows:
                            writer.writerow([
                                row_data["Image"],
                                row_data["Plate ID"],
                                row_data["Confidence"],
                                row_data["Timestamp"],
                                row_data["Status"]
                            ])
                
                self.show_status(f"Exported {len(data_rows)} results with {len(exported_images)} images", "#00a86b")
                QMessageBox.information(self, "Success", 
                                      f"Exported {len(data_rows)} results to:\n{file_path}\n\n"
                                      f"Images saved to:\n{images_dir}\n"
                                      f"({len(exported_images)} images)")
            except Exception as e:
                self.show_status(f"Export failed: {str(e)}", "#ff5722")
                QMessageBox.critical(self, "Export Error", str(e))
    
    def export_matched_images(self):
        """Export only matched plate images to a selected folder"""
        if self.results_table.rowCount() == 0:
            QMessageBox.warning(self, "No Results", "No detection results to export")
            return
        
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Folder for Images")
        if not export_dir:
            return
        
        try:
            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)
            
            exported_count = 0
            plate_dir = Path("detected_plates").resolve()
            
            for row in range(self.results_table.rowCount()):
                plate_item = self.results_table.item(row, 1)
                status_item = self.results_table.item(row, 4)
                
                if not plate_item or not status_item:
                    continue
                
                plate_text = plate_item.text()
                status = status_item.text()
                
                # Only export matched plates
                if status != "MATCHED":
                    continue
                
                # Find saved plate images
                if plate_dir.exists():
                    safe_plate = "".join(c if c.isalnum() else "_" for c in plate_text)
                    matching_images = sorted(plate_dir.glob(f"{safe_plate}_*.jpg"))
                    
                    for i, src_image in enumerate(matching_images):
                        dst_name = f"{safe_plate}_matched_{i+1}.jpg"
                        dst_path = export_path / dst_name
                        import shutil
                        shutil.copy2(src_image, dst_path)
                        exported_count += 1
            
            self.show_status(f"Exported {exported_count} matched images to {export_path}", "#00a86b")
            QMessageBox.information(self, "Success", 
                                  f"Exported {exported_count} matched plate images to:\n{export_path}")
        except Exception as e:
            self.show_status(f"Image export failed: {str(e)}", "#ff5722")
            QMessageBox.critical(self, "Export Error", f"Failed to export images:\n{str(e)}")
    
    def clear_results(self):
        """Clear all matched detection results"""
        if self.results_table.rowCount() == 0:
            return
        
        reply = QMessageBox.question(self, "Confirm Clear", 
                                    f"Clear all {self.results_table.rowCount()} matched plates?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.results_table.setRowCount(0)
            self.detected_vehicles = []
            self.progress_bar.setValue(0)
            self.show_status("Matched plates cleared", "#ff9800")
    
    def show_status(self, message, color="#00a86b"):
        """Show status message"""
        self.statusBar().showMessage(f"  {message}")
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: #2d2d2d;
                color: {color};
                border-top: 1px solid #404040;
                font-weight: bold;
            }}
        """)


# ==================== MAIN ====================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LicensePlateDetectorApp()
    window.show()
    sys.exit(app.exec_())