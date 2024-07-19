import sys
import cv2
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QTextEdit,
                             QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QSlider, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect, QPoint
from datetime import datetime
from options import DenoiseOptions, DeblurOptions
import logging
import traceback
from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QGroupBox

# Default directory for data
default_directory = "../data"
fixed_size = (640, 480)  # Fixed size for displayed images

def get_video_frames(video_path):
    """
    Extract frames from a video file.

    :param video_path: Path to the video file
    :return: List of frames as numpy arrays
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

class VideoEnhancementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Enhancement Tool")
        self.setGeometry(100, 100, 1280, 720)  # Set the application to a fixed size

        self.source_path = ""
        self.source_data = []
        self.displayed_image = None
        self.roi_bbox = None  # Variable to store ROI bounding box

        self.denoise_options = DenoiseOptions(self.log_message)
        self.deblur_options = DeblurOptions(self.log_message)

        self.selecting_roi = False
        self.roi_start = QPoint()
        self.roi_end = QPoint()

        self.init_ui()

    def init_ui(self):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_area.setWidget(self.scroll_content)
        self.setCentralWidget(self.scroll_area)

        self.main_layout = QVBoxLayout(self.scroll_content)

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.main_tab = QWidget()
        self.log_tab = QWidget()

        self.tabs.addTab(self.main_tab, "Main")
        self.tabs.addTab(self.log_tab, "Log")

        self.tab_main_layout = QVBoxLayout(self.main_tab)
        self.log_layout = QVBoxLayout(self.log_tab)

        # Main tab layout
        self.upload_btn = QPushButton("Browse", self)
        self.upload_btn.clicked.connect(self.upload_source)
        self.tab_main_layout.addWidget(self.upload_btn)

        self.edit_video_btn = QPushButton("Video Editor", self)
        self.edit_video_btn.clicked.connect(self.edit_video)
        self.tab_main_layout.addWidget(self.edit_video_btn)

        # Image display area
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(fixed_size[0], fixed_size[1])
        self.image_label.setStyleSheet("background-color: #1C1C1C;")
        self.image_label.mousePressEvent = self.image_mouse_press
        self.image_label.mouseMoveEvent = self.image_mouse_move
        self.image_label.mouseReleaseEvent = self.image_mouse_release
        self.tab_main_layout.addWidget(self.image_label)

        # ROI selection button
        self.select_roi_btn = QPushButton("Select ROI", self)
        self.select_roi_btn.clicked.connect(self.enable_roi_selection)
        self.select_roi_btn.setFixedSize(100, 30)  # Make the button smaller
        self.tab_main_layout.addWidget(self.select_roi_btn)

        # Video control section
        self.video_control_layout = QHBoxLayout()

        self.prev_frame_btn = QPushButton("Previous Frame")
        self.prev_frame_btn.clicked.connect(self.previous_frame)
        self.video_control_layout.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QPushButton("Next Frame")
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.video_control_layout.addWidget(self.next_frame_btn)

        self.delete_frame_btn = QPushButton("Delete Frame")
        self.delete_frame_btn.clicked.connect(self.delete_frame)
        self.video_control_layout.addWidget(self.delete_frame_btn)

        self.tab_main_layout.addLayout(self.video_control_layout)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_slider_change)
        self.tab_main_layout.addWidget(self.frame_slider)

        self.frame_counter = QLabel("Frame: 0 / 0")
        self.tab_main_layout.addWidget(self.frame_counter)

        self.enhancement_label = QLabel("Select Enhancement Method:", self)
        self.enhancement_label.setStyleSheet("font-weight: bold;")
        self.tab_main_layout.addWidget(self.enhancement_label)

        self.enhancement_button_group = QButtonGroup(self)

        # Denoise options
        denoise_box = QGroupBox("Denoise Methods")
        denoise_layout = QVBoxLayout()
        self.denoise_options_list = [DenoiseOptions.SCC, DenoiseOptions.ECC, DenoiseOptions.FEATURE_BASED,
                                     DenoiseOptions.OPTICAL_FLOW, DenoiseOptions.DENOISE_COT, DenoiseOptions.RV_CLASSIC,
                                     DenoiseOptions.DENOISE_YOV]

        for i, option in enumerate(self.denoise_options_list):
            radio_btn = QRadioButton(option, self)
            self.enhancement_button_group.addButton(radio_btn, i)
            denoise_layout.addWidget(radio_btn)
        denoise_box.setLayout(denoise_layout)
        self.tab_main_layout.addWidget(denoise_box)

        # Deblur options
        deblur_box = QGroupBox("Deblur Methods")
        deblur_layout = QVBoxLayout()
        self.deblur_options_list = [DeblurOptions.RV_OM, DeblurOptions.NAFNET, DeblurOptions.NUBKE,
                                    DeblurOptions.NUMBKE2WIN, DeblurOptions.UNSUPERWIN]

        for i, option in enumerate(self.deblur_options_list, start=len(self.denoise_options_list)):
            radio_btn = QRadioButton(option, self)
            self.enhancement_button_group.addButton(radio_btn, i)
            deblur_layout.addWidget(radio_btn)
        deblur_box.setLayout(deblur_layout)
        self.tab_main_layout.addWidget(deblur_box)

        self.enhance_btn = QPushButton("Perform enhancement", self)
        self.enhance_btn.clicked.connect(self.perform_enhancement)
        self.tab_main_layout.addWidget(self.enhance_btn)

        # Log tab layout
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #1C1C1C; color: white;")
        self.log_layout.addWidget(self.log_output)

        self.save_log_btn = QPushButton("Save log", self)
        self.save_log_btn.clicked.connect(self.save_log)
        self.log_layout.addWidget(self.save_log_btn)

        # Initially disable video control buttons
        self.prev_frame_btn.setEnabled(False)
        self.next_frame_btn.setEnabled(False)
        self.delete_frame_btn.setEnabled(False)
        self.frame_slider.setEnabled(False)
        self.select_roi_btn.setEnabled(False)  # Disable ROI button initially

    def upload_source(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", default_directory,
                                                   "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
        if file_path:
            self.source_path = file_path
            print(f"Video uploaded: {self.source_path}")
            self.source_data = get_video_frames(file_path)
            self.current_frame = 0
            if self.source_data:
                self.show_image(self.source_data[0])
                self.update_slider()
                self.enable_video_controls()  # New method to enable controls
            self.log_message(f"Video loaded: {len(self.source_data)} frames", "info")

    def show_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, fixed_size)
        height, width, channel = frame_resized.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Draw the ROI if it exists
        if self.roi_bbox:
            painter = QPainter(q_img)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.roi_bbox)
            painter.end()

        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def perform_enhancement(self):
        if not self.source_path:
            self.log_message("Please upload a source file before performing enhancements.", "danger")
            return

        selected_option = self.enhancement_button_group.checkedButton()

        if selected_option:
            method = selected_option.text()
            self.log_message(f"Enhancement selected: {method}, Source Path - {self.source_path}", "info")
        else:
            self.log_message("Please select an enhancement method.", "warning")
            return

        # Denoise methods
        if method == DenoiseOptions.SCC:
            self.denoise_options.perform_SCC(self.source_path, self.source_data)
        elif method == DenoiseOptions.ECC:
            self.denoise_options.perform_ECC(self.source_path, self.source_data)
        elif method == DenoiseOptions.FEATURE_BASED:
            self.denoise_options.perform_FEATURE_BASED(self.source_path, self.source_data)
        elif method == DenoiseOptions.OPTICAL_FLOW:
            self.denoise_options.perform_OPTICAL_FLOW(self.source_path, self.source_data)
        elif method == DenoiseOptions.DENOISE_COT:
            self.denoise_options.perform_DENOISE_COT(self.source_path, self.source_data)
        elif method == DenoiseOptions.RV_CLASSIC:
            self.denoise_options.perform_RV_CLASSIC(self.source_path, self.source_data)
        elif method == DenoiseOptions.DENOISE_YOV:
            self.denoise_options.perform_DENOISE_YOV(self.source_path, self.source_data)
        # Deblur methods
        elif method == DeblurOptions.RV_OM:
            self.deblur_options.perform_RV_OM(self.source_path, self.source_data)
        elif method == DeblurOptions.NAFNET:
            self.deblur_options.perform_NAFNET(self.source_path, self.source_data)
        elif method == DeblurOptions.NUBKE:
            self.deblur_options.perform_NUBKE(self.source_path, self.source_data)
        elif method == DeblurOptions.NUMBKE2WIN:
            self.deblur_options.perform_NUMBKE2WIN(self.source_path, self.source_data)
        elif method == DeblurOptions.UNSUPERWIN:
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
        try:
            if not self.source_path:
                self.log_message("Please upload a video before using the Video Editor.", "warning")
                return

            self.log_message("Initializing Video Editor...", "info")
            from VideoEditor.Video_Editor import VideoEditor  # Import here to avoid circular import
            self.video_editor = VideoEditor()  # Create without parent
            self.log_message("Setting video in Video Editor...", "info")
            self.video_editor.set_video(self.source_path)

            if hasattr(self.video_editor, 'editing_finished'):
                self.log_message("Connecting signals...", "info")
                self.video_editor.editing_finished.connect(self.update_video)
            else:
                self.log_message("Warning: editing_finished signal not found", "warning")

            self.log_message("Showing Video Editor...", "info")
            self.video_editor.show()

        except ImportError as ie:
            self.log_message(f"Error importing VideoEditor: {str(ie)}", "danger")
            logging.error(f"Import error in edit_video: {str(ie)}")
            logging.error(traceback.format_exc())
        except Exception as e:
            self.log_message(f"Error in edit_video: {str(e)}", "danger")
            logging.error(f"Error in edit_video: {str(e)}")
            logging.error(traceback.format_exc())

    def set_video(self, video_path, parent=None):
        if parent:
            self.setParent(parent)
        self.video_path = video_path
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.frames = []
        self.trimmed_frames = []
        self.processed_frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames.append(frame)
        self.cap.release()

        self.current_frame = 0
        self.start_trim_slider.setRange(0, len(self.frames) - 1)
        self.end_trim_slider.setRange(0, len(self.frames) - 1)
        self.end_trim_slider.setValue(len(self.frames) - 1)
        self.frame_slider.setRange(0, len(self.frames) - 1)
        self.update_frame()
        self.enable_video_controls()

    def save_log(self):
        self.log_message("Saving log...", "info")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Log", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            with open(file_path, 'w') as file:
                log_text = self.log_output.toPlainText()
                file.write(log_text)
            self.log_message("Log saved successfully.", "info")

    def update_video(self, new_frames):
        try:
            logging.info(f"Updating video with {len(new_frames)} new frames")
            self.source_data = new_frames
            self.current_frame = 0
            if self.source_data:
                self.show_image(self.source_data[0])
                self.update_slider()
            self.log_message("Video updated with edited version.", "info")
        except Exception as e:
            logging.error(f"Error in update_video: {str(e)}")
            logging.error(traceback.format_exc())
            self.log_message(f"Error updating video: {str(e)}", "danger")

    def update_frame(self):
        try:
            if self.source_data and 0 <= self.current_frame < len(self.source_data):
                frame = self.source_data[self.current_frame]
                self.show_image(frame)
        except Exception as e:
            logging.error(f"Error in update_frame: {str(e)}")
            logging.error(traceback.format_exc())
            self.log_message(f"Error updating frame: {str(e)}", "danger")

    def previous_frame(self):
        if self.source_data and self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame()

    def next_frame(self):
        if self.source_data and self.current_frame < len(self.source_data) - 1:
            self.current_frame += 1
            self.update_frame()

    def delete_frame(self):
        if self.source_data and self.current_frame < len(self.source_data):
            del self.source_data[self.current_frame]
            if self.current_frame >= len(self.source_data):
                self.current_frame = len(self.source_data) - 1
            self.update_frame()
            self.update_slider()

    def on_slider_change(self):
        if self.source_data:
            self.current_frame = self.frame_slider.value()
            self.update_frame()

    def update_slider(self):
        if self.source_data:
            self.frame_slider.setRange(0, len(self.source_data) - 1)
            self.frame_slider.setValue(self.current_frame)
            self.frame_counter.setText(f"Frame: {self.current_frame + 1} / {len(self.source_data)}")

    def update_frame(self):
        if self.source_data and 0 <= self.current_frame < len(self.source_data):
            frame = self.source_data[self.current_frame]
            self.show_image(frame)
            self.update_slider()

    def enable_video_controls(self):
        self.prev_frame_btn.setEnabled(True)
        self.next_frame_btn.setEnabled(True)
        self.delete_frame_btn.setEnabled(True)
        self.frame_slider.setEnabled(True)
        self.select_roi_btn.setEnabled(True)  # Enable ROI button

    def enable_roi_selection(self):
        self.selecting_roi = True
        self.log_message("ROI selection enabled. Click and drag to select ROI.", "info")

    def image_mouse_press(self, event):
        if self.selecting_roi:
            self.roi_start = event.pos()
            self.roi_end = event.pos()

    def image_mouse_move(self, event):
        if self.selecting_roi:
            self.roi_end = event.pos()
            self.update_frame()

    def image_mouse_release(self, event):
        if self.selecting_roi:
            self.roi_end = event.pos()
            self.selecting_roi = False
            self.roi_bbox = QRect(self.roi_start, self.roi_end).normalized()
            self.log_message(f"ROI selected: {self.roi_bbox.getRect()}", "info")
            self.update_frame()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoEnhancementApp()
    window.show()
    sys.exit(app.exec_())
