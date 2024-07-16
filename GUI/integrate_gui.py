import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QTextEdit, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QSlider, QScrollArea, QRadioButton, QButtonGroup, QGroupBox
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt
from datetime import datetime
from options import DenoiseOptions, DeblurOptions
from Video_Editor import VideoEditor
import logging
import traceback
import numpy as np


from RapidBase.import_all import *
from Utils import *


default_directory = "../data"
fixed_size = (640, 480)

def get_video_frames(video_path):
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
        self.setGeometry(100, 100, 1800, 900)

        self.source_path = ""
        self.source_data = []
        self.previous_result_data = []
        self.current_result_data = []

        self.current_frame = 0
        self.previous_result_frame = 0
        self.current_result_frame = 0

        self.denoise_options = DenoiseOptions(self.log_message)
        self.deblur_options = DeblurOptions(self.log_message)

        self.init_ui()

    def init_ui(self):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_area_widget)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.main_tab = QWidget()
        self.log_tab = QWidget()

        self.tabs.addTab(self.main_tab, "Main")
        self.tabs.addTab(self.log_tab, "Log")

        self.main_layout = QVBoxLayout(self.scroll_area_widget)
        self.log_layout = QVBoxLayout(self.log_tab)

        self.main_tab_layout = QVBoxLayout(self.main_tab)
        self.main_tab_layout.addWidget(self.scroll_area)

        self.upload_btn = QPushButton("Browse", self)
        self.upload_btn.clicked.connect(self.upload_source)
        self.main_layout.addWidget(self.upload_btn)

        self.edit_video_btn = QPushButton("Video Editor", self)
        self.edit_video_btn.clicked.connect(self.edit_video)
        self.main_layout.addWidget(self.edit_video_btn)

        self.video_display_layout = QHBoxLayout()

        self.original_video_layout = self.create_video_layout("Original")
        self.video_display_layout.addLayout(self.original_video_layout[0])

        self.previous_result_layout = self.create_video_layout("Previous Result")
        self.video_display_layout.addLayout(self.previous_result_layout[0])

        self.current_result_layout = self.create_video_layout("Current Result")
        self.video_display_layout.addLayout(self.current_result_layout[0])

        self.main_layout.addLayout(self.video_display_layout)

        self.connect_video_controls(self.original_video_layout, self.navigate_original_video)
        self.connect_video_controls(self.previous_result_layout, self.navigate_previous_result)
        self.connect_video_controls(self.current_result_layout, self.navigate_current_result)

        self.enhancement_group = QVBoxLayout()
        self.enhancement_label = QLabel("Select Enhancement Method:", self)
        self.enhancement_label.setStyleSheet("font-weight: bold;")
        self.enhancement_group.addWidget(self.enhancement_label)

        self.enhancement_button_group = QButtonGroup(self)

        denoise_box = QGroupBox("Denoise Methods")
        denoise_layout = QVBoxLayout()
        self.denoise_options_list = ["SCC", "ECC", "FEATURE_BASED", "OPTICAL_FLOW", "DENOISE_COT", "RV_CLASSIC", "DENOISE_YOV"]

        for i, option in enumerate(self.denoise_options_list):
            radio_btn = QRadioButton(option, self)
            self.enhancement_button_group.addButton(radio_btn, i)
            denoise_layout.addWidget(radio_btn)
        denoise_box.setLayout(denoise_layout)
        self.enhancement_group.addWidget(denoise_box)

        deblur_box = QGroupBox("Deblur Methods")
        deblur_layout = QVBoxLayout()
        self.deblur_options_list = ["RV_OM", "NAFNET", "NUBKE", "NUMBKE2WIN", "UNSUPERWIN"]

        for i, option in enumerate(self.deblur_options_list, start=len(self.denoise_options_list)):
            radio_btn = QRadioButton(option, self)
            self.enhancement_button_group.addButton(radio_btn, i)
            deblur_layout.addWidget(radio_btn)
        deblur_box.setLayout(deblur_layout)
        self.enhancement_group.addWidget(deblur_box)

        self.main_layout.addLayout(self.enhancement_group)

        self.advanced_options_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout()

        self.video_selection_group = QButtonGroup(self)
        self.original_video_radio = QRadioButton("Use Original Video")
        self.previous_result_radio = QRadioButton("Use Previously Enhanced Video")
        self.current_result_radio = QRadioButton("Use Current Result Video")
        self.original_video_radio.setChecked(True)
        self.video_selection_group.addButton(self.original_video_radio)
        self.video_selection_group.addButton(self.previous_result_radio)
        self.video_selection_group.addButton(self.current_result_radio)

        advanced_layout.addWidget(self.original_video_radio)
        advanced_layout.addWidget(self.previous_result_radio)
        advanced_layout.addWidget(self.current_result_radio)
        self.advanced_options_group.setLayout(advanced_layout)

        self.main_layout.addWidget(self.advanced_options_group)

        self.enhance_btn = QPushButton("Perform enhancement", self)
        self.enhance_btn.clicked.connect(self.perform_enhancement)
        self.main_layout.addWidget(self.enhance_btn)

        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #1C1C1C; color: white;")
        self.log_layout.addWidget(self.log_output)

        self.save_log_btn = QPushButton("Save log", self)
        self.save_log_btn.clicked.connect(self.save_log)
        self.log_layout.addWidget(self.save_log_btn)

        self.disable_all_video_controls()

    def create_video_layout(self, title):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(title))

        image_label = QLabel(self)
        image_label.setFixedSize(fixed_size[0], fixed_size[1])
        image_label.setStyleSheet("background-color: #1C1C1C;")
        layout.addWidget(image_label)

        control_layout = QHBoxLayout()
        prev_frame_btn = QPushButton("Previous Frame")
        next_frame_btn = QPushButton("Next Frame")
        control_layout.addWidget(prev_frame_btn)
        control_layout.addWidget(next_frame_btn)
        layout.addLayout(control_layout)

        frame_slider = QSlider(Qt.Horizontal)
        layout.addWidget(frame_slider)

        frame_counter = QLabel("Frame: 0 / 0")
        layout.addWidget(frame_counter)

        return layout, image_label, prev_frame_btn, next_frame_btn, frame_slider, frame_counter

    def connect_video_controls(self, video_layout, navigation_function):
        _, _, prev_btn, next_btn, slider, _ = video_layout
        prev_btn.clicked.connect(lambda: navigation_function(-1))
        next_btn.clicked.connect(lambda: navigation_function(1))
        slider.valueChanged.connect(lambda value: navigation_function(value, is_slider=True))

    def upload_source(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", default_directory,
                                                   "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
        if file_path:
            try:
                self.source_path = file_path
                self.log_message(f"Video uploaded: {self.source_path}", "info")
                self.source_data = get_video_frames(file_path)
                self.current_frame = 0

                self.previous_result_data = []
                self.previous_result_frame = 0
                self.current_result_data = []
                self.current_result_frame = 0

                if self.source_data:
                    self.update_all_video_windows()
                    self.disable_result_video_controls()

                self.log_message(f"Video loaded: {len(self.source_data)} frames", "info")

                self.previous_result_radio.setEnabled(False)
                self.current_result_radio.setEnabled(False)
                self.original_video_radio.setChecked(True)
            except Exception as e:
                self.log_message(f"Error loading video: {str(e)}", "danger")
                logging.error(f"Error in upload_source: {str(e)}")
                logging.error(traceback.format_exc())

    def disable_result_video_controls(self):
        self.disable_video_controls(*self.previous_result_layout[2:5])
        self.disable_video_controls(*self.current_result_layout[2:5])

    def show_image(self, frame, label):
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, fixed_size)
            height, width, channel = frame_resized.shape
            bytes_per_line = 3 * width
            dtype = frame_resized.dtype
            if dtype == 'float32' and frame_resized.max() < 5:
                frame_resized = np.clip(frame_resized*255, 0, 255).astype(np.uint8)
            q_img = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(q_img))
        else:
            label.clear()

    def perform_enhancement(self):
        try:
            if not self.source_path:
                self.log_message("Please upload a source file before performing enhancements.", "danger")
                return

            selected_option = self.enhancement_button_group.checkedButton()

            if not selected_option:
                self.log_message("Please select an enhancement method.", "warning")
                return

            method = selected_option.text()
            self.log_message(f"Enhancement selected: {method}, Source Path - {self.source_path}", "info")

            if self.original_video_radio.isChecked():
                input_frames = self.source_data
                self.log_message("Using original video for enhancement", "info")
            elif self.previous_result_radio.isChecked():
                if isinstance(self.previous_result_data, np.ndarray) and self.previous_result_data.size > 0:
                    input_frames = self.previous_result_data
                    self.log_message("Using previous result for enhancement", "info")
                else:
                    self.log_message("No previous result available. Using original video.", "warning")
                    input_frames = self.source_data
            elif self.current_result_radio.isChecked():
                if isinstance(self.current_result_data, np.ndarray) and self.current_result_data.size > 0:
                    input_frames = self.current_result_data
                    self.log_message("Using current result for enhancement", "info")
                else:
                    self.log_message("No current result available. Using original video.", "warning")
                    input_frames = self.source_data
            else:
                raise ValueError("No video source selected")

            self.log_message(f"Starting enhancement: {method}", "info")

            if method in self.denoise_options_list:
                enhanced_frames = getattr(self.denoise_options, f"perform_{method}")(self.source_path, input_frames[0:8])
            elif method in self.deblur_options_list:
                enhanced_frames = getattr(self.deblur_options, f"perform_{method}")(self.source_path, input_frames[0:8])
            else:
                raise ValueError(f"Unknown method: {method}")

            if enhanced_frames is None:
                raise ValueError("Enhancement method returned None")

            if isinstance(enhanced_frames, list):
                enhanced_frames = list_to_numpy(enhanced_frames)
            elif not isinstance(enhanced_frames, np.ndarray):
                raise TypeError(f"Enhancement method returned unexpected type: {type(enhanced_frames)}")

            if enhanced_frames.size == 0:
                raise ValueError("Enhancement method returned empty array")

            self.previous_result_data = self.current_result_data if isinstance(self.current_result_data,
                                                                               np.ndarray) and self.current_result_data.size > 0 else self.source_data
            self.current_result_data = enhanced_frames

            self.previous_result_frame = 0
            self.current_result_frame = 0

            self.update_all_video_windows()
            self.enable_all_video_controls()

            self.log_message(f"Enhancement completed: {len(enhanced_frames)} frames processed", "info")

            self.previous_result_radio.setEnabled(True)
            self.current_result_radio.setEnabled(True)

        except AttributeError as ae:
            self.log_message(f"AttributeError in perform_enhancement: {str(ae)}", "danger")
            logging.error(f"AttributeError in perform_enhancement: {str(ae)}")
            logging.error(traceback.format_exc())
        except ValueError as ve:
            self.log_message(f"ValueError in perform_enhancement: {str(ve)}", "danger")
            logging.error(f"ValueError in perform_enhancement: {str(ve)}")
            logging.error(traceback.format_exc())
        except TypeError as te:
            self.log_message(f"TypeError in perform_enhancement: {str(te)}", "danger")
            logging.error(f"TypeError in perform_enhancement: {str(te)}")
            logging.error(traceback.format_exc())
        except Exception as e:
            self.log_message(f"Unexpected error in perform_enhancement: {str(e)}", "danger")
            logging.error(f"Unexpected error in perform_enhancement: {str(e)}")
            logging.error(traceback.format_exc())
        finally:
            self.log_message("Enhancement process finished", "info")

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
            from VideoEditor.Video_Editor import VideoEditor
            self.video_editor = VideoEditor()
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
                self.update_original_frame()
                self.update_original_slider()
            self.log_message("Video updated with edited version.", "info")
        except Exception as e:
            logging.error(f"Error in update_video: {str(e)}")
            logging.error(traceback.format_exc())
            self.log_message(f"Error updating video: {str(e)}", "danger")

    def update_video_window(self, frames, current_frame, image_label, prev_btn, next_btn, slider, counter):
        if isinstance(frames, np.ndarray):
            frames_exist = frames.size > 0
        else:
            frames_exist = bool(frames)

        if frames_exist and 0 <= current_frame < len(frames):
            self.show_image(frames[current_frame], image_label)
            slider.setRange(0, len(frames) - 1)
            slider.setValue(current_frame)
            counter.setText(f"Frame: {current_frame + 1} / {len(frames)}")
            prev_btn.setEnabled(current_frame > 0)
            next_btn.setEnabled(current_frame < len(frames) - 1)
            slider.setEnabled(True)
        else:
            image_label.clear()
            slider.setRange(0, 0)
            counter.setText("Frame: 0 / 0")
            prev_btn.setEnabled(False)
            next_btn.setEnabled(False)
            slider.setEnabled(False)

    def update_all_video_windows(self):
        self.update_video_window(self.source_data, self.current_frame, *self.original_video_layout[1:])
        self.update_video_window(self.previous_result_data, self.previous_result_frame,
                                 *self.previous_result_layout[1:])
        self.update_video_window(self.current_result_data, self.current_result_frame, *self.current_result_layout[1:])

    def enable_video_controls(self, prev_btn, next_btn, slider):
        prev_btn.setEnabled(True)
        next_btn.setEnabled(True)
        slider.setEnabled(True)

    def enable_all_video_controls(self):
        self.enable_video_controls(*self.original_video_layout[2:5])
        self.enable_video_controls(*self.previous_result_layout[2:5])
        self.enable_video_controls(*self.current_result_layout[2:5])

    def disable_video_controls(self, prev_btn, next_btn, slider):
        prev_btn.setEnabled(False)
        next_btn.setEnabled(False)
        slider.setEnabled(False)

    def disable_all_video_controls(self):
        self.disable_video_controls(*self.original_video_layout[2:5])
        self.disable_video_controls(*self.previous_result_layout[2:5])
        self.disable_video_controls(*self.current_result_layout[2:5])

    def navigate_original_video(self, value, is_slider=False):
        if is_slider:
            self.current_frame = value
        else:
            self.current_frame = max(0, min(self.current_frame + value, len(self.source_data) - 1))
        self.update_video_window(self.source_data, self.current_frame, *self.original_video_layout[1:])

    def navigate_previous_result(self, value, is_slider=False):
        if is_slider:
            self.previous_result_frame = value
        else:
            self.previous_result_frame = max(0, min(self.previous_result_frame + value,
                                                    len(self.previous_result_data) - 1))
        self.update_video_window(self.previous_result_data, self.previous_result_frame,
                                 *self.previous_result_layout[1:])

    def navigate_current_result(self, value, is_slider=False):
        if is_slider:
            self.current_result_frame = value
        else:
            self.current_result_frame = max(0,
                                            min(self.current_result_frame + value, len(self.current_result_data) - 1))
        self.update_video_window(self.current_result_data, self.current_result_frame, *self.current_result_layout[1:])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoEnhancementApp()
    window.show()
    sys.exit(app.exec_())
