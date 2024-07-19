import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QTextEdit, QCheckBox, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt
from datetime import datetime
from options import DenoiseOptions, DeblurOptions
from util.video_to_numpy_array import get_video_frames

# Default directory for data
default_directory = "../data"
fixed_size = (640, 480)  # Fixed size for displayed images

class VideoEnhancementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Enhancement Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.source_path = ""
        self.source_data = []
        self.displayed_image = None

        self.denoise_options = DenoiseOptions(self.log_message)
        self.deblur_options = DeblurOptions(self.log_message)

        self.init_ui()

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.main_tab = QWidget()
        self.log_tab = QWidget()

        self.tabs.addTab(self.main_tab, "Main")
        self.tabs.addTab(self.log_tab, "Log")

        self.main_layout = QVBoxLayout(self.main_tab)
        self.log_layout = QVBoxLayout(self.log_tab)

        # Main tab layout
        self.upload_btn = QPushButton("Browse", self)
        self.upload_btn.clicked.connect(self.upload_source)
        self.main_layout.addWidget(self.upload_btn)

        self.edit_video_btn = QPushButton("Video Editor", self)
        self.edit_video_btn.clicked.connect(self.edit_video)
        self.main_layout.addWidget(self.edit_video_btn)

        # Denoise options
        self.denoise_group = QVBoxLayout()
        self.denoise_label = QLabel("Select Denoise Methods:", self)
        self.denoise_label.setStyleSheet("font-weight: bold;")
        self.denoise_group.addWidget(self.denoise_label)

        self.denoise_vars = []
        self.denoise_checkboxes = []
        self.denoise_options_list = [DenoiseOptions.SCC, DenoiseOptions.ECC, DenoiseOptions.FEATURE_BASED,
                                     DenoiseOptions.OPTICAL_FLOW, DenoiseOptions.DENOISE_COT, DenoiseOptions.RV_CLASSIC, DenoiseOptions.DENOISE_YOV]

        for option in self.denoise_options_list:
            var = QCheckBox(option, self)
            self.denoise_vars.append(var)
            self.denoise_checkboxes.append(var)
            self.denoise_group.addWidget(var)

        self.main_layout.addLayout(self.denoise_group)

        # Deblur options
        self.deblur_group = QVBoxLayout()
        self.deblur_label = QLabel("Select Deblur Methods:", self)
        self.deblur_label.setStyleSheet("font-weight: bold;")
        self.deblur_group.addWidget(self.deblur_label)

        self.deblur_vars = []
        self.deblur_checkboxes = []
        self.deblur_options_list = [DeblurOptions.RV_OM, DeblurOptions.NAFNET, DeblurOptions.NUBKE,
                                    DeblurOptions.NUMBKE2WIN, DeblurOptions.UNSUPERWIN]

        for option in self.deblur_options_list:
            var = QCheckBox(option, self)
            self.deblur_vars.append(var)
            self.deblur_checkboxes.append(var)
            self.deblur_group.addWidget(var)

        self.main_layout.addLayout(self.deblur_group)

        self.enhance_btn = QPushButton("Perform enhancement", self)
        self.enhance_btn.clicked.connect(self.perform_enhancement)
        self.main_layout.addWidget(self.enhance_btn)

        # Image display area
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(fixed_size[0], fixed_size[1])
        self.image_label.setStyleSheet("background-color: #1C1C1C;")
        self.main_layout.addWidget(self.image_label)

        # Log tab layout
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #1C1C1C; color: white;")
        self.log_layout.addWidget(self.log_output)

        self.save_log_btn = QPushButton("Save log", self)
        self.save_log_btn.clicked.connect(self.save_log)
        self.log_layout.addWidget(self.save_log_btn)

    def upload_source(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", default_directory, "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
        if file_path:
            self.source_path = file_path
            self.source_data = get_video_frames(file_path)
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                self.displayed_image = frame  # Store the original image for further processing
                self.show_image(frame)
            cap.release()

    def show_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, fixed_size)
        height, width, channel = frame_resized.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def perform_enhancement(self):
        if not self.source_path:
            self.log_message("Please upload a source file before performing enhancements.", "danger")
            return

        selected_denoise = [option for checkbox, option in zip(self.denoise_checkboxes, self.denoise_options_list) if checkbox.isChecked()]
        selected_deblur = [option for checkbox, option in zip(self.deblur_checkboxes, self.deblur_options_list) if checkbox.isChecked()]

        if selected_denoise or selected_deblur:
            self.log_message(f"Enhancement selected: Denoise - {selected_denoise}, Deblur - {selected_deblur}, Source Path - {self.source_path}", "info")
        else:
            self.log_message("Please select at least one denoise or deblur method.", "warning")

        for method in selected_denoise:
            if method == DenoiseOptions.SCC:
                self.denoise_options.perform_SCC(self.source_path, self.source_data)
            if method == DenoiseOptions.ECC:
                self.denoise_options.perform_ECC(self.source_path, self.source_data)
            if method == DenoiseOptions.FEATURE_BASED:
                self.denoise_options.perform_FEATURE_BASED(self.source_path, self.source_data)
            if method == DenoiseOptions.OPTICAL_FLOW:
                self.denoise_options.perform_OPTICAL_FLOW(self.source_path, self.source_data)
            if method == DenoiseOptions.DENOISE_COT:
                self.denoise_options.perform_DENOISE_COT(self.source_path, self.source_data)
            if method == DenoiseOptions.RV_CLASSIC:
                self.denoise_options.perform_RV_CLASSIC(self.source_path, self.source_data)
            if method == DenoiseOptions.DENOISE_YOV:
                self.denoise_options.perform_DENOISE_YOV(self.source_path, self.source_data)

        for method in selected_deblur:
            if method == DeblurOptions.RV_OM:
                self.deblur_options.perform_RV_OM(self.source_path, self.source_data)
            if method == DeblurOptions.NAFNET:
                self.deblur_options.perform_NAFNET(self.source_path, self.source_data)
            if method == DeblurOptions.NUBKE:
                self.deblur_options.perform_NUBKE(self.source_path, self.source_data)
            if method == DeblurOptions.NUMBKE2WIN:
                self.deblur_options.perform_NUMBKE2WIN(self.source_path, self.source_data)
            if method == DeblurOptions.UNSUPERWIN:
                self.deblur_options.perform_UNSUPERWIN(self.source_path, self.source_data)

    def log_message(self, message, level="info"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        color = {
            "info": QColor("white"),
            "warning": QColor("yellow"),
            "danger": QColor("red"),
            "comment": QColor("blue")
        }.get(level, QColor("white"))
        self.log_output.setTextColor(color)
        self.log_output.append(f"{timestamp} - {message}")

    def edit_video(self):
        self.log_message("Edit Video function called.", "info")

    def save_log(self):
        self.log_message("Saving log...", "info")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Log", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            with open(file_path, 'w') as file:
                log_text = self.log_output.toPlainText()
                file.write(log_text)
            self.log_message("Log saved successfully.", "info")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoEnhancementApp()
    window.show()
    sys.exit(app.exec_())
