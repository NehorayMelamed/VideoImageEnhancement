import os
import sys


# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Add the script directory to the Python path
sys.path.append(script_dir)


### Using nice ui terminal intro
# from welcome.app import display_banner
# display_banner()

from Segmentation.sam import get_mask_from_bbox
import PARAMETER
import sys
import cv2
import glob
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QTextEdit, QCheckBox,
                             QLineEdit, QDialogButtonBox, QRadioButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QSlider, QMessageBox, QComboBox, QLabel,
                             QProgressDialog, QDialog, QScrollArea,
                             QButtonGroup, QGroupBox)
from PyQt5.QtCore import Qt, QPoint, QCoreApplication, Qt, QSize, QThread, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QImage, QCursor, QIcon, QMovie, QColor, QPainter, QPen, QMouseEvent, QWheelEvent, \
    QTransform

from datetime import datetime
from options import DenoiseOptions, DeblurOptions
import logging
import traceback
import numpy as np
from functools import partial
from multiprocessing import Process, Queue
import ctypes
import pickle




# Default directory for data
default_directory = "../data"
fixed_size = (640, 480)  # Fixed size for displayed images
# Configure logging
logging.basicConfig(filename='video_enhancement.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


def get_video_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames  # This is already a list of numpy arrays


class VideoEnhancementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" AlgoLightLTD - Video Enhancement")
        self.setWindowIcon(QIcon('icons/logo.png'))
        self.setGeometry(100, 100, 1800, 900)  # Increased width to accommodate three videos
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F8F8F8;
            }
            QPushButton {
                background-color: #333333;
                color: white;
                border: none;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
            QLabel {
                font-size: 17px;
                color: #333333;
            }
            QSlider::groove:horizontal {
                border: 1px solid #CCCCCC;
                background: white;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #333333;
                border: 1px solid #333333;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #CCCCCC;
                border-radius: 5px;
            }
            QSlider::add-page:horizontal {
                background: #FFFFFF;
                border-radius: 5px;
            }
            QComboBox, QLineEdit {
                padding: 6px;
                border: 1px solid #CCCCCC;
                border-radius: 15px;
                background-color: white;
                font-size: 16px; /* Increased font size */
            }
            QLineEdit {
                padding: 2px;
                border: 1px solid #CCCCCC;
                border-radius: 15px;
                background-color: white;
                font-size: 16px; /* Increased font size */
            }
            QTabWidget::pane {
                border: none;
            }
            QTabBar::tab {
                background: #333333;
                color: white;
                padding: 10px 20px;
                min-height: 18px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 2px;
                font-size: 16px; /* Increased font size */
                min-width: 60px;
            }
            QTabBar::tab:selected {
                background: #555555;
                font-weight: bold;
            }
            QTabBar::tab:!selected {
                background: #777777;
            }
            QTabBar::tab:hover {
                background: #555555;
            }
            QScrollArea {
                border: none;
                border-radius: 15px;
            }
            QScrollBar:vertical {
                background: #F8F8F8;
                width: 16px;
                margin: 0px 3px 0px 3px;
                border-radius: 8px;
            }
            QScrollBar::handle:vertical {
                background: #333333;
                min-height: 20px;
                border-radius: 8px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;
                border: none;
            }
            QScrollBar:horizontal {
                background: #F8F8F8;
                height: 16px;
                margin: 3px 0px 3px 0px;
                border-radius: 8px;
            }
            QScrollBar::handle:horizontal {
                background: #333333;
                min-width: 20px;
                border-radius: 8px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: none;
                border: none;
            }
            QRadioButton, QCheckBox {
                font-size: 16px; /* Increased font size */
                spacing: 5px;
                padding: 5px;
                border-radius: 10px;
            }
            QGroupBox {
                margin-top: 15px;
                padding: 10px;
                border: 1px solid #CCCCCC;
                border-radius: 10px;
            }
            QLabel[videoFrame="true"] {
                background-color: #1C1C1C;
                border-radius: 15px;
            }
        """)

        self.source_path = ""
        self.source_data = []
        self.previous_result_data = []
        self.current_result_data = []

        self.current_frame = 0
        self.previous_result_frame = 0
        self.current_result_frame = 0

        self.enhancement_thread = None

        self.denoise_options = DenoiseOptions(self.log_message)
        self.deblur_options = DeblurOptions(self.log_message)

        self.deblur_options_list = list(DeblurOptions.OptionsServices.DICT.keys())
        self.denoise_options_list = list(DenoiseOptions.OptionsServices.DICT.keys())

        # Store the original default parameters
        self.default_denoise_params = {option: params.copy() for option, params in
                                       DenoiseOptions.OptionsServices.DICT.items()}
        self.default_deblur_params = {option: params.copy() for option, params in
                                      DeblurOptions.OptionsServices.DICT.items()}

        self.current_result_type = "frames"
        self.additional_results = {}
        self.result_types = ["frames"]
        self.enhanced_frames = None

        self.init_ui()

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.main_tab = QWidget()
        self.log_tab = QWidget()

        self.tabs.addTab(self.main_tab, "Main")
        self.tabs.addTab(self.log_tab, "Log")

        # Create an enhanced scroll area for the main tab
        scroll_area = EnhancedScrollArea()
        scroll_content = QWidget()
        self.main_layout = QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)

        # Set the scroll area as the layout for the main tab
        main_tab_layout = QVBoxLayout(self.main_tab)
        main_tab_layout.addWidget(scroll_area)

        self.log_layout = QVBoxLayout(self.log_tab)

        # Main tab layout
        self.upload_btn = QPushButton("Browse", self)
        self.upload_btn.clicked.connect(self.upload_source)
        self.main_layout.addWidget(self.upload_btn)

        # Add dropdown for video selection
        self.video_choice_dropdown = QComboBox()
        self.video_choice_dropdown.addItems(["Original Video", "Previous Result Video", "Current Result Video"])
        self.main_layout.addWidget(self.video_choice_dropdown)

        self.edit_video_btn = QPushButton("Video Editor", self)
        self.edit_video_btn.clicked.connect(self.edit_video)
        self.main_layout.addWidget(self.edit_video_btn)

        # Video display layout
        self.video_display_layout = QHBoxLayout()

        # Original video
        self.original_video_layout = self.create_video_layout("Original")
        self.video_display_layout.addLayout(self.original_video_layout[0])

        # Previous result video
        self.previous_result_layout = self.create_video_layout("Previous Result")
        self.video_display_layout.addLayout(self.previous_result_layout[0])

        # Current result video
        current_result_container = QVBoxLayout()
        self.current_result_layout = self.create_video_layout("Current Result")
        current_result_container.addLayout(self.current_result_layout[0])

        # Add result navigation controls
        self.result_navigation_layout = self.create_result_navigation()
        current_result_container.addLayout(self.result_navigation_layout)

        self.video_display_layout.addLayout(current_result_container)

        self.main_layout.addLayout(self.video_display_layout)

        self.connect_video_controls(self.original_video_layout, self.navigate_original_video)
        self.connect_video_controls(self.previous_result_layout, self.navigate_previous_result)
        self.connect_video_controls(self.current_result_layout, self.navigate_current_result)

        # Connect flip buttons
        self.original_video_layout[7].clicked.connect(self.flip_original_video)
        self.previous_result_layout[7].clicked.connect(self.flip_previous_result_video)
        self.current_result_layout[7].clicked.connect(self.flip_current_result_video)

        # Enhancement options
        self.enhancement_group = QVBoxLayout()
        self.enhancement_label = QLabel("Select Enhancement Method:", self)
        self.enhancement_label.setStyleSheet("font-weight: bold;")
        self.enhancement_group.addWidget(self.enhancement_label)

        self.enhancement_button_group = QButtonGroup(self)

        # Create a horizontal layout for denoise and deblur options
        options_layout = QHBoxLayout()

        # Denoise options
        denoise_box = QGroupBox("Denoise Methods")
        denoise_layout = QVBoxLayout()
        for i, option in enumerate(self.denoise_options_list):
            option_layout = QHBoxLayout()

            if DenoiseOptions.OptionsServices.DICT[option]:
                settings_btn = QPushButton()
                settings_btn.setIcon(QIcon("icons/settings.png"))
                settings_btn.setFixedSize(20, 20)
                settings_btn.setStyleSheet("background-color: transparent; border: none;")
                settings_btn.clicked.connect(
                    partial(self.open_parameter_settings, option, DenoiseOptions.OptionsServices.DICT[option]))
                option_layout.addWidget(settings_btn)
            else:
                placeholder = QWidget()
                placeholder.setFixedSize(20, 20)
                option_layout.addWidget(placeholder)

            radio_btn = QRadioButton(option, self)
            self.enhancement_button_group.addButton(radio_btn, i)
            option_layout.addWidget(radio_btn)

            denoise_layout.addLayout(option_layout)
        denoise_box.setLayout(denoise_layout)
        options_layout.addWidget(denoise_box)

        # Deblur options
        deblur_box = QGroupBox("Deblur Methods")
        deblur_layout = QVBoxLayout()
        for i, option in enumerate(self.deblur_options_list, start=len(self.denoise_options_list)):
            option_layout = QHBoxLayout()

            if DeblurOptions.OptionsServices.DICT[option]:
                settings_btn = QPushButton()
                settings_btn.setIcon(QIcon("icons/settings.png"))
                settings_btn.setFixedSize(20, 20)
                settings_btn.setStyleSheet("background-color: transparent; border: none;")
                settings_btn.clicked.connect(
                    partial(self.open_parameter_settings, option, DeblurOptions.OptionsServices.DICT[option]))
                option_layout.addWidget(settings_btn)
            else:
                placeholder = QWidget()
                placeholder.setFixedSize(20, 20)
                option_layout.addWidget(placeholder)

            radio_btn = QRadioButton(option, self)
            self.enhancement_button_group.addButton(radio_btn, i)
            option_layout.addWidget(radio_btn)

            deblur_layout.addLayout(option_layout)
        deblur_box.setLayout(deblur_layout)
        options_layout.addWidget(deblur_box)

        # Add the options layout to the enhancement group
        self.enhancement_group.addLayout(options_layout)

        self.main_layout.addLayout(self.enhancement_group)

        # Advanced options
        self.advanced_options_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout()

        self.video_selection_group = QButtonGroup(self)
        self.original_video_radio = QRadioButton("Use Original Video")
        self.previous_result_radio = QRadioButton("Use Previously Enhanced Video")
        self.current_result_radio = QRadioButton("Use Current Result Video")
        self.original_video_radio.setChecked(True)  # Default to original video
        self.video_selection_group.addButton(self.original_video_radio)
        self.video_selection_group.addButton(self.previous_result_radio)
        self.video_selection_group.addButton(self.current_result_radio)

        advanced_layout.addWidget(self.original_video_radio)
        advanced_layout.addWidget(self.previous_result_radio)
        advanced_layout.addWidget(self.current_result_radio)
        self.advanced_options_group.setLayout(advanced_layout)

        self.main_layout.addWidget(self.advanced_options_group)

        # Add the additional data dropdown after the advanced options
        self.main_layout.addWidget(QLabel("Additional Data:"))
        self.main_layout.addWidget(self.create_additional_data_dropdown())

        self.original_video_radio.toggled.connect(self.update_additional_data_dropdown)
        self.previous_result_radio.toggled.connect(self.update_additional_data_dropdown)
        self.current_result_radio.toggled.connect(self.update_additional_data_dropdown)

        # Replace the single checkbox with a group of radio buttons
        self.drawing_options_group = QGroupBox("Drawing Options")
        self.drawing_options_layout = QVBoxLayout()
        self.drawing_options_group.setLayout(self.drawing_options_layout)

        self.drawing_button_group = QButtonGroup(self)
        self.no_draw_radio = QRadioButton("No drawing")
        self.draw_polygon_radio = QRadioButton("Draw polygon")
        self.draw_box_radio = QRadioButton("Draw box")
        self.draw_polygon_mask_radio = QRadioButton("Draw box and use mask")

        self.drawing_button_group.addButton(self.no_draw_radio)
        self.drawing_button_group.addButton(self.draw_polygon_radio)
        self.drawing_button_group.addButton(self.draw_box_radio)
        self.drawing_button_group.addButton(self.draw_polygon_mask_radio)

        self.drawing_options_layout.addWidget(self.no_draw_radio)
        self.drawing_options_layout.addWidget(self.draw_polygon_radio)
        self.drawing_options_layout.addWidget(self.draw_box_radio)
        self.drawing_options_layout.addWidget(self.draw_polygon_mask_radio)

        self.no_draw_radio.setChecked(True)  # Set "No drawing" as default

        # Add the new drawing options group
        self.main_layout.addWidget(self.drawing_options_group)

        self.enhance_btn = QPushButton("Perform enhancement", self)
        self.enhance_btn.clicked.connect(self.perform_enhancement)
        self.main_layout.addWidget(self.enhance_btn)

        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.clicked.connect(self.stop_enhancement)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF4136;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    text-align: center;
                    text-decoration: none;
                    font-size: 12px;
                    margin: 4px 2px;
                    border-radius: 12px;
                }
                QPushButton:hover {
                    background-color: #FF7166;
                }
                QPushButton:disabled {
                    background-color: #FFCCC9;
                }
            """)
        self.stop_btn.setFixedSize(130, 50)  # Set a fixed size for the button

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.enhance_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.setAlignment(self.stop_btn, Qt.AlignRight)  # Align the stop button to the right

        self.main_layout.addLayout(button_layout)

        # Log tab layout
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #1C1C1C; color: white;")
        self.log_layout.addWidget(self.log_output)

        self.save_log_btn = QPushButton("Save log", self)
        self.save_log_btn.clicked.connect(self.save_log)
        self.log_layout.addWidget(self.save_log_btn)

        # Initially disable video control buttons
        self.disable_all_video_controls()

        # Add stretch to align content to the top
        self.main_layout.addStretch(1)

    def create_video_layout(self, title):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(title))

        image_label = QLabel(self)
        image_label.setFixedSize(fixed_size[0], fixed_size[1])
        image_label.setProperty("videoFrame", True)
        layout.addWidget(image_label)

        control_layout = QHBoxLayout()
        prev_frame_btn = QPushButton("Previous Frame")
        next_frame_btn = QPushButton("Next Frame")
        expand_btn = QPushButton("Expand")
        flip_btn = QPushButton("Flip Video")
        control_layout.addWidget(prev_frame_btn)
        control_layout.addWidget(next_frame_btn)
        control_layout.addWidget(expand_btn)
        control_layout.addWidget(flip_btn)
        layout.addLayout(control_layout)

        frame_slider = QSlider(Qt.Horizontal)
        layout.addWidget(frame_slider)

        frame_counter = QLabel("Frame: 0 / 0")
        layout.addWidget(frame_counter)

        # Connect expand button
        expand_btn.clicked.connect(lambda: self.expand_video(image_label, title))

        # We don't connect the flip button here because it's connected in init_ui
        # for each specific video layout

        return layout, image_label, prev_frame_btn, next_frame_btn, frame_slider, frame_counter, expand_btn, flip_btn

    def connect_video_controls(self, video_layout, navigation_function):
        _, _, prev_btn, next_btn, slider, counter, expand_btn, flip_btn = video_layout
        prev_btn.clicked.connect(lambda: navigation_function(-1))
        next_btn.clicked.connect(lambda: navigation_function(1))
        slider.valueChanged.connect(lambda value: navigation_function(value, is_slider=True))

    def upload_source(self):
        try:
            # Ask the user whether they want to select a file or a directory
            choice = QMessageBox.question(self, "Upload Source",
                                          "Do you want to upload a video file(yes) or a directory(no) of frames?",
                                          QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                                          QMessageBox.Yes)

            if choice == QMessageBox.Cancel:
                return

            if choice == QMessageBox.Yes:
                # User wants to upload a video file
                file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                           "Video Files (*.mp4 *.avi);;All Files (*)")
                if not file_path:
                    return  # User canceled the file dialog

                # Load video file
                self.source_path = file_path
                self.source_data = get_video_frames(file_path)
                self.log_message(f"Video uploaded: {self.source_path}", "info")

            elif choice == QMessageBox.No:
                # User wants to upload a directory of frames
                directory = QFileDialog.getExistingDirectory(self, "Select Directory with Frames", "")

                if not directory:
                    return  # User canceled the directory dialog

                # Load frames from the directory
                image_files = sorted(glob.glob(os.path.join(directory, "*.png")) +
                                     glob.glob(os.path.join(directory, "*.jpg")) +
                                     glob.glob(os.path.join(directory, "*.jpeg")))

                if not image_files:
                    raise ValueError("No valid image files found in the directory.")

                self.source_data = [cv2.imread(img) for img in image_files]
                self.source_path = directory
                self.log_message(f"Frames loaded from directory: {directory}", "info")

            # Convert source data to list if necessary
            if isinstance(self.source_data, np.ndarray):
                self.source_data = list(self.source_data)

            self.current_frame = 0

            # Reset previous and current result videos
            self.previous_result_data = []
            self.previous_result_frame = 0
            self.current_result_data = []
            self.current_result_frame = 0

            if self.source_data:
                self.update_all_video_windows()
                self.disable_result_video_controls()

            self.log_message(f"Video loaded: {len(self.source_data)} frames", "info")

            # Disable the options to use previously enhanced or current result video
            self.previous_result_radio.setEnabled(False)
            self.current_result_radio.setEnabled(False)
            self.original_video_radio.setChecked(True)

        except Exception as e:
            self.log_message(f"Error loading video or frames: {str(e)}", "danger")
            self.show_error_message("Error loading video or frames", str(e))
            logging.error(f"Error in upload_source: {str(e)}")
            logging.error(traceback.format_exc())

    def expand_video(self, image_label, title):
        original_frame = image_label.property("original_frame")
        if original_frame is not None:
            try:
                # Convert the frame to RGB if it's not already
                if len(original_frame.shape) == 2:  # Grayscale
                    original_frame = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2RGB)
                elif original_frame.shape[2] == 4:  # RGBA
                    original_frame = cv2.cvtColor(original_frame, cv2.COLOR_RGBA2RGB)

                dialog = ExpandedVideoDialog(original_frame, self)
                dialog.setWindowTitle(f"Expanded View - {title}")
                dialog.resize(800, 600)  # Set a default size, adjust as needed
                dialog.exec_()
            except Exception as e:
                self.log_message(f"Error in expand_video: {str(e)}", "danger")
                self.show_error_message("Error Expanding Video", f"An error occurred: {str(e)}")
                logging.error(f"Error in expand_video: {str(e)}")
                logging.error(traceback.format_exc())
        else:
            self.log_message("No frame to expand", "warning")
            self.show_error_message("No Frame", "There is no frame to expand.")

    def disable_result_video_controls(self):
        self.disable_video_controls(*self.previous_result_layout[2:])
        self.disable_video_controls(*self.current_result_layout[2:])

    def show_image(self, frame, label):
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, fixed_size)
            height, width, channel = frame_resized.shape
            bytes_per_line = 3 * width
            dtype = frame_resized.dtype
            if dtype == 'float32' and frame_resized.max() < 5:
                frame_resized = np.clip(frame_resized * 255, 0, 255).astype(np.uint8)
            q_img = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(pixmap)
            label.setProperty("original_frame", frame_rgb)  # Store original frame for expanding
        else:
            label.clear()

    def get_minimal_bounding_box(self, polygon):
        """
        Calculate the minimal bounding box of a given polygon.

        PARAMETER:
        polygon (list): A list of (x, y) tuples representing the polygon vertices.

        Returns:
        tuple: A tuple containing the minimal bounding box (orig_x, orig_y, orig_w, orig_h) in pixel coordinates.
        """
        x_coords, y_coords = zip(*polygon)
        min_x, min_y = min(x_coords), min(y_coords)
        max_x, max_y = max(x_coords), max(y_coords)

        orig_x = min_x
        orig_y = min_y
        orig_w = max_x - min_x
        orig_h = max_y - min_y

        return int(orig_x), int(orig_y), int(orig_w), int(orig_h)

    def perform_enhancement(self):
        try:
            self.log_message("Starting enhancement process...", "info")
            current_additional_results = self.additional_results.copy()  # Store current additional results
            self.log_message(f"Current additional results: {current_additional_results.keys()}", "info")

            self.enhanced_frames = None
            self.additional_results = {}
            self.result_types = ["frames"]
            self.current_result_type = "frames"

            if not self.source_path:
                self.log_message("Please upload a source file before performing enhancements.", "danger")
                self.show_error_message("No Source File", "Please upload a source file before performing enhancements.")
                return

            selected_option = self.enhancement_button_group.checkedButton()

            if not selected_option:
                self.log_message("Please select an enhancement method.", "warning")
                self.show_error_message("No Enhancement Method", "Please select an enhancement method.")
                return

            method = selected_option.text()
            self.log_message(f"Enhancement selected: {method}, Source Path - {self.source_path}", "info")

            try:
                input_frames = self.get_input_frames(current_additional_results)
                if isinstance(input_frames, tuple):
                    input_frames, additional_data = input_frames
                    self.log_message(f"Additional data received: {type(additional_data)}", "info")
                else:
                    additional_data = None
                    self.log_message("No additional data received", "info")
            except Exception as e:
                raise ValueError(f"Error getting input frames: {str(e)}")

            self.log_message(f"Input frames type: {type(input_frames)}, Length: {len(input_frames)}", "info")

            # Ensure input_frames is a list of numpy arrays
            if isinstance(input_frames, np.ndarray):
                if input_frames.ndim == 4:
                    input_frames = list(input_frames)
                elif input_frames.ndim == 3:
                    input_frames = [input_frames]  # Single frame
                elif input_frames.ndim == 2:
                    input_frames = [input_frames]  # Single grayscale frame
                else:
                    raise ValueError(f"Unexpected input_frames shape: {input_frames.shape}")
            elif not isinstance(input_frames, list):
                raise ValueError(f"Invalid input frames format: {type(input_frames)}")

            input_frames = [np.array(frame) if not isinstance(frame, np.ndarray) else frame for frame in input_frames]

            self.log_message(f"First frame shape: {input_frames[0].shape}, dtype: {input_frames[0].dtype}", "info")

            polygon, mask, box, input_method, user_input, params = None, None, None, None, None, None

            drawing_option = self.drawing_button_group.checkedButton().text()

            if drawing_option != "No drawing":
                self.log_message(f"Drawing option selected: {drawing_option}", "info")
                first_frame = input_frames[0].copy()
                if first_frame.ndim == 2:
                    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
                elif first_frame.ndim == 3 and first_frame.shape[2] == 4:
                    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGBA2BGR)

                dialog = ModernDrawingDialog(first_frame, drawing_option)
                if dialog.exec_() == QDialog.Accepted:
                    if drawing_option == "Draw polygon":
                        polygon = dialog.get_polygon()
                        box = self.get_minimal_bounding_box(polygon)
                        input_method = PARAMETER.InputMethodForServices.polygon
                        user_input = polygon
                    elif drawing_option == "Draw box":
                        polygon = dialog.get_polygon()  # This is actually a box
                        box = self.get_minimal_bounding_box(polygon)
                        input_method = PARAMETER.InputMethodForServices.BB
                        user_input = box
                    elif drawing_option == "Draw box and use mask":
                        polygon = dialog.get_polygon()
                        box = self.get_minimal_bounding_box(polygon)
                        mask = dialog.get_mask()
                        input_method = PARAMETER.InputMethodForServices.segmentation
                        user_input = mask

                    self.log_message(
                        f"Drawing completed: polygon={polygon}, box={box}, mask={'obtained' if mask is not None else 'not used'}",
                        "info")

            self.show_loading_indicator()

            QApplication.processEvents()  # This line ensures the loading indicator is displayed

            # Perform the selected enhancement method
            self.log_message(f"Starting enhancement: {method}", "info")

            if not self.current_result_radio.isChecked():
                current_additional_results = None

            input_dict = {PARAMETER.DictInput.frames: input_frames,
                          PARAMETER.DictInput.input_method: input_method,
                          PARAMETER.DictInput.user_input: user_input,
                          PARAMETER.DictInput.params: params,
                          "added_data": additional_data,
                          "previous_results": current_additional_results}

            self.log_message(f"Input dict created. Keys: {input_dict.keys()}", "info")
            self.log_message(f"Added data in input_dict: {type(input_dict['added_data'])}", "info")

            if method in self.denoise_options_list:
                params = DenoiseOptions.OptionsServices.DICT[method].copy()
            elif method in self.deblur_options_list:
                params = DeblurOptions.OptionsServices.DICT[method].copy()
            else:
                raise ValueError(f"Unknown method: {method}")

            input_dict[PARAMETER.DictInput.params] = params

            options_type = 'denoise' if method in self.denoise_options_list else 'deblur'

            self.enhancement_thread = EnhancementThread(method, input_dict, options_type)
            self.enhancement_thread.finished.connect(self.on_enhancement_finished)
            self.enhancement_thread.error.connect(self.on_enhancement_error)
            self.enhancement_thread.start()

            self.enhance_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

        except Exception as e:
            self.hide_loading_indicator()
            self.log_message(f"Error in perform_enhancement: {str(e)}", "danger")
            self.show_error_message("Error in Enhancement", f"An error occurred during enhancement: {str(e)}")
            logging.error(f"Error in perform_enhancement: {str(e)}")
            logging.error(traceback.format_exc())
        # finally:
        #     self.hide_loading_indicator()
        #     self.log_message("Enhancement process finished", "info")

    def stop_enhancement(self):
        if self.enhancement_thread and self.enhancement_thread.isRunning():
            self.enhancement_thread.stop()
            self.enhancement_thread.wait()
            self.log_message("Enhancement process stopped.", "info")
            self.hide_loading_indicator()
            self.enhance_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def on_enhancement_finished(self, enhancement_result):
        self.hide_loading_indicator()
        self.enhance_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_message(f"Enhancement completed. Result type: {type(enhancement_result)}", "info")
        try:
            new_frames, additional_results = self.process_enhancement_result(enhancement_result)
            self.log_message(f"Processed frames: {len(new_frames)}", "info")
            self.update_ui_with_results(new_frames, additional_results)
        except Exception as e:
            self.log_message(f"Error processing enhancement result: {str(e)}", "danger")
            self.show_error_message("Error Processing Result", f"An error occurred: {str(e)}")
        finally:
            self.log_message("Enhancement process finished", "info")

    def on_enhancement_error(self, error_message):
        self.hide_loading_indicator()
        self.enhance_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_message(f"Error in enhancement: {error_message}", "danger")
        self.show_error_message("Error in Enhancement", f"An error occurred during enhancement: {error_message}")

    def update_ui_with_results(self, new_frames, additional_results):
        self.previous_result_data = self.current_result_data if isinstance(self.current_result_data,
                                                                           list) else self.source_data
        self.current_result_data = new_frames
        self.enhanced_frames = new_frames
        self.additional_results = additional_results

        self.log_message(f"Updated additional_results: {self.additional_results.keys()}", "info")

        self.result_types = ["frames"] + list(additional_results.keys())
        self.current_result_type = "frames"

        self.result_type_dropdown.clear()
        self.result_type_dropdown.addItems(self.result_types)
        self.result_type_dropdown.setCurrentIndex(0)

        self.update_additional_data_dropdown()

        self.log_message(f"Updated result types: {self.result_types}", "info")
        self.log_message(f"Current result type set to: '{self.current_result_type}'", "info")

        self.previous_result_frame = 0
        self.current_result_frame = 0

        self.update_all_video_windows()
        self.enable_all_video_controls()
        self.update_result_navigation()

        self.previous_result_radio.setEnabled(True)
        self.current_result_radio.setEnabled(True)

    def process_enhancement_result(self, enhancement_result):
        if isinstance(enhancement_result, dict):
            if "frames" not in enhancement_result:
                raise ValueError("Enhancement method returned a dict without 'frames' key")
            new_frames = enhancement_result["frames"]
            additional_results = {k: v for k, v in enhancement_result.items() if k != "frames"}
        elif isinstance(enhancement_result, (list, np.ndarray)):
            new_frames = enhancement_result
            additional_results = {}
        else:
            raise ValueError(f"Unexpected enhancement result type: {type(enhancement_result)}")

        # Ensure new_frames is a list of numpy arrays
        if isinstance(new_frames, np.ndarray):
            if new_frames.ndim == 4:
                new_frames = list(new_frames)
            elif new_frames.ndim == 3:
                new_frames = [new_frames]
            elif new_frames.ndim == 2:
                new_frames = [new_frames]
            else:
                raise ValueError(f"Unexpected new_frames shape: {new_frames.shape}")
        elif not isinstance(new_frames, list):
            new_frames = [new_frames]

        return new_frames, additional_results

    def get_input_frames(self, current_additional_results):
        self.log_message(
            f"Getting input frames. Radio button states: Original={self.original_video_radio.isChecked()}, Previous={self.previous_result_radio.isChecked()}, Current={self.current_result_radio.isChecked()}",
            "info")
        self.log_message(f"Current additional_results keys: {current_additional_results.keys()}", "info")

        if self.original_video_radio.isChecked():
            return self.source_data
        elif self.previous_result_radio.isChecked() and self.previous_result_data:
            return self.previous_result_data
        elif self.current_result_radio.isChecked():
            if self.current_result_type == "frames":
                main_data = self.current_result_data
            else:
                main_data = self.additional_results.get(self.current_result_type, [])

            additional_data_key = self.additional_data_dropdown.currentText()
            self.log_message(f"Additional data key selected: {additional_data_key}", "info")

            if additional_data_key != "None":
                result_type, _, key = additional_data_key.partition(": ")
                self.log_message(f"Parsed result type: {result_type}, key: {key}", "info")

                if result_type in current_additional_results:
                    if isinstance(current_additional_results[result_type], dict):
                        additional_data = current_additional_results[result_type].get(key, None)
                    else:
                        additional_data = current_additional_results[result_type]

                    self.log_message(f"Additional data found: {type(additional_data)}", "info")
                    return main_data, additional_data
                else:
                    self.log_message(f"Result type {result_type} not found in additional_results", "warning")

            return main_data
        else:
            return self.source_data

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

    def show_error_message(self, title, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        detailed_text = traceback.format_exc()
        msg.setDetailedText(detailed_text)

        copy_button = QPushButton('Copy to Clipboard')
        msg.addButton(copy_button, QMessageBox.ActionRole)

        def copy_to_clipboard():
            QApplication.clipboard().setText(f"{message}\n\nDetailed Error:\n{detailed_text}")

        copy_button.clicked.connect(copy_to_clipboard)

        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def edit_video(self):
        try:
            choice = self.video_choice_dropdown.currentIndex()

            if choice == 0:
                video_data = self.source_data
                video_name = "Original Video"
            elif choice == 1:
                video_data = self.previous_result_data
                video_name = "Previous Result Video"
            elif choice == 2:
                if self.current_result_type == "frames":
                    video_data = self.current_result_data
                else:
                    video_data = self.additional_results.get(self.current_result_type, [])
                video_name = f"Current Result Video ({self.current_result_type})"
            else:
                raise ValueError("Invalid choice")

            if video_data is None or (isinstance(video_data, (list, np.ndarray)) and len(video_data) == 0):
                self.log_message(f"No {video_name} data available for editing.", "warning")
                self.show_error_message("No Video Data", f"Please ensure {video_name} exists before editing.")
                return

            self.log_message(f"Initializing Video Editor for {video_name}...", "info")
            from video_editor_5 import VideoEditor
            self.video_editor = VideoEditor()
            self.log_message(f"Setting {video_name} in Video Editor...", "info")

            # Ensure video_data is a list of frames
            if isinstance(video_data, np.ndarray):
                if video_data.ndim == 4:
                    video_data = list(video_data)
                elif video_data.ndim == 3:
                    video_data = [video_data]
                elif video_data.ndim == 2:
                    video_data = [video_data]
                else:
                    raise ValueError(f"Unexpected video data shape: {video_data.shape}")
            elif not isinstance(video_data, list):
                video_data = [video_data]

            # Create a temporary video file
            temp_video_path = f"temp_{video_name.lower().replace(' ', '_')}.mp4"
            self.save_frames_to_video(video_data, temp_video_path)

            self.video_editor.set_video(temp_video_path)

            # Ensure undo/redo stacks are initialized
            self.video_editor.undo_stack = []
            self.video_editor.redo_stack = []

            # Add initial state to undo stack
            self.video_editor.add_undo_state()

            if hasattr(self.video_editor, 'editing_finished'):
                self.log_message("Connecting signals...", "info")
                self.video_editor.editing_finished.connect(
                    lambda new_frames: self.on_editing_finished(new_frames, choice))
            else:
                self.log_message("Warning: editing_finished signal not found", "warning")

            self.log_message("Showing Video Editor...", "info")
            self.video_editor.show()

        except ImportError as ie:
            self.log_message(f"Error importing VideoEditor: {str(ie)}", "danger")
            self.show_error_message("Import Error", f"Error importing VideoEditor: {str(ie)}")
            logging.error(f"Import error in edit_video: {str(ie)}")
            logging.error(traceback.format_exc())
        except Exception as e:
            self.log_message(f"Error in edit_video: {str(e)}", "danger")
            self.show_error_message("Error in Video Editor", f"An error occurred: {str(e)}")
            logging.error(f"Error in edit_video: {str(e)}")
            logging.error(traceback.format_exc())

    def on_editing_finished(self, new_frames, choice):
        self.update_edited_video(new_frames, choice)
        self.refresh_ui()

    def save_frames_to_video(self, frames, output_path):
        if not frames:
            raise ValueError("No frames to save")

        # Ensure frames is a list
        if not isinstance(frames, list):
            frames = [frames]

        # Get the shape of the first frame
        first_frame = frames[0]
        if isinstance(first_frame, np.ndarray):
            height, width = first_frame.shape[:2]
        else:
            raise ValueError(f"Unsupported frame type: {type(first_frame)}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        for frame in frames:
            if isinstance(frame, np.ndarray):
                if frame.ndim == 2:  # If the frame is grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:  # If the frame has an alpha channel
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] != 3:
                    raise ValueError(f"Unexpected frame shape: {frame.shape}")

                # Ensure the frame is uint8
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)

                out.write(frame)
            else:
                raise ValueError(f"Unsupported frame type: {type(frame)}")

        out.release()

        # Ensure the video was created successfully
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError(f"Failed to create video at {output_path}")

    def update_edited_video(self, new_frames, choice):
        try:
            logging.info(f"Updating video with {len(new_frames)} new frames")
            if choice == 0:  # Original Video
                self.source_data = new_frames
                self.current_frame = 0
                self.log_message("Original video updated with edited version.", "info")
            elif choice == 1:  # Previous Result Video
                self.previous_result_data = new_frames
                self.previous_result_frame = 0
                self.log_message("Previous result video updated with edited version.", "info")
            elif choice == 2:  # Current Result Video
                if self.current_result_type == "frames":
                    self.current_result_data = new_frames
                    self.enhanced_frames = new_frames
                else:
                    self.additional_results[self.current_result_type] = new_frames
                self.current_result_frame = 0
                self.log_message(f"Current result video ({self.current_result_type}) updated with edited version.",
                                 "info")

        except Exception as e:
            logging.error(f"Error in update_edited_video: {str(e)}")
            logging.error(traceback.format_exc())
            self.log_message(f"Error updating video: {str(e)}", "danger")
            self.show_error_message("Error Updating Video", f"An error occurred while updating the video: {str(e)}")

    def save_log(self):
        self.log_message("Saving log...", "info")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Log", "", "Text Files (*.txt);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    log_text = self.log_output.toPlainText()
                    file.write(log_text)
                self.log_message("Log saved successfully.", "info")
            except Exception as e:
                self.log_message(f"Error saving log: {str(e)}", "danger")
                self.show_error_message("Error Saving Log", f"An error occurred while saving the log: {str(e)}")
                logging.error(f"Error in save_log: {str(e)}")
                logging.error(traceback.format_exc())

    def update_video_window(self, frames, current_frame, image_label, prev_btn, next_btn, slider, counter, expand_btn,
                            flip_btn):
        if isinstance(frames, np.ndarray):
            if frames.ndim == 4:
                frames = list(frames)
            elif frames.ndim == 3:
                frames = [frames]
            elif frames.ndim == 2:
                frames = [frames]
            else:
                raise ValueError(f"Unexpected frames shape: {frames.shape}")
        elif not isinstance(frames, list):
            # Handle single value results
            image_label.setText(str(frames))
            prev_btn.setEnabled(False)
            next_btn.setEnabled(False)
            slider.setEnabled(False)
            expand_btn.setEnabled(False)
            flip_btn.setEnabled(False)
            counter.setText("Single Value Result")
            return

        frames_exist = frames and len(frames) > 0

        if frames_exist and 0 <= current_frame < len(frames):
            self.show_image(frames[current_frame], image_label)
            slider.setRange(0, len(frames) - 1)
            slider.setValue(current_frame)
            counter.setText(f"Frame: {current_frame + 1} / {len(frames)}")
            prev_btn.setEnabled(current_frame > 0)
            next_btn.setEnabled(current_frame < len(frames) - 1)
            slider.setEnabled(True)
            expand_btn.setEnabled(True)
            flip_btn.setEnabled(True)
        else:
            image_label.clear()
            slider.setRange(0, 0)
            counter.setText("Frame: 0 / 0")
            prev_btn.setEnabled(False)
            next_btn.setEnabled(False)
            slider.setEnabled(False)
            expand_btn.setEnabled(False)
            flip_btn.setEnabled(False)

    def update_all_video_windows(self):
        self.update_video_window(self.source_data, self.current_frame, *self.original_video_layout[1:])
        self.update_video_window(self.previous_result_data, self.previous_result_frame,
                                 *self.previous_result_layout[1:])

        if self.current_result_type == "frames":
            current_result_data = self.current_result_data
        else:
            current_result_data = self.additional_results.get(self.current_result_type, [])

        self.update_video_window(current_result_data, self.current_result_frame, *self.current_result_layout[1:])

    def enable_video_controls(self, prev_btn, next_btn, slider, counter, expand_btn, flip_btn):
        prev_btn.setEnabled(True)
        next_btn.setEnabled(True)
        slider.setEnabled(True)
        expand_btn.setEnabled(True)
        flip_btn.setEnabled(True)

    def enable_all_video_controls(self):
        self.enable_video_controls(*self.original_video_layout[2:])
        self.enable_video_controls(*self.previous_result_layout[2:])
        self.enable_video_controls(*self.current_result_layout[2:])

    def disable_video_controls(self, prev_btn, next_btn, slider, counter, expand_btn, flip_btn):
        prev_btn.setEnabled(False)
        next_btn.setEnabled(False)
        slider.setEnabled(False)
        expand_btn.setEnabled(False)
        flip_btn.setEnabled(False)
        # Note: We don't typically disable the counter as it's just a label

    def disable_all_video_controls(self):
        self.disable_video_controls(*self.original_video_layout[2:])
        self.disable_video_controls(*self.previous_result_layout[2:])
        self.disable_video_controls(*self.current_result_layout[2:])

    def navigate_original_video(self, value, is_slider=False):
        try:
            if is_slider:
                self.current_frame = value
            else:
                self.current_frame = max(0, min(self.current_frame + value, len(self.source_data) - 1))
            self.update_video_window(self.source_data, self.current_frame, *self.original_video_layout[1:])
        except Exception as e:
            self.log_message(f"Error navigating original video: {str(e)}", "danger")
            self.show_error_message("Error Navigating Video", f"An error occurred: {str(e)}")
            logging.error(f"Error in navigate_original_video: {str(e)}")
            logging.error(traceback.format_exc())

    def navigate_previous_result(self, value, is_slider=False):
        try:
            if is_slider:
                self.previous_result_frame = value
            else:
                self.previous_result_frame = max(0, min(self.previous_result_frame + value,
                                                        len(self.previous_result_data) - 1))
            self.update_video_window(self.previous_result_data, self.previous_result_frame,
                                     *self.previous_result_layout[1:])
        except Exception as e:
            self.log_message(f"Error navigating previous result video: {str(e)}", "danger")
            self.show_error_message("Error Navigating Video", f"An error occurred: {str(e)}")
            logging.error(f"Error in navigate_previous_result: {str(e)}")
            logging.error(traceback.format_exc())

    def navigate_current_result(self, value, is_slider=False):
        try:
            if is_slider:
                self.current_result_frame = value
            else:
                self.current_result_frame = max(0,
                                                min(self.current_result_frame + value,
                                                    len(self.current_result_data) - 1))
            self.update_video_window(self.current_result_data, self.current_result_frame,
                                     *self.current_result_layout[1:])
        except Exception as e:
            self.log_message(f"Error navigating current result video: {str(e)}", "danger")
            self.show_error_message("Error Navigating Video", f"An error occurred: {str(e)}")
            logging.error(f"Error in navigate_current_result: {str(e)}")
            logging.error(traceback.format_exc())

    def flip_video(self, video_data, current_frame, layout):
        try:
            if isinstance(video_data, np.ndarray):
                if video_data.ndim == 4:
                    video_data = list(video_data)
                elif video_data.ndim == 3:
                    video_data = [video_data]
                else:
                    raise ValueError(f"Unexpected video_data shape: {video_data.shape}")

            if video_data and len(video_data) > 0:
                flipped_data = video_data[::-1]  # Reverse the list of frames
                new_current_frame = len(flipped_data) - 1 - current_frame
                self.update_video_window(flipped_data, new_current_frame, *layout[1:])
                return flipped_data, new_current_frame
            return video_data, current_frame
        except Exception as e:
            self.log_message(f"Error flipping video: {str(e)}", "danger")
            self.show_error_message("Error Flipping Video", f"An error occurred: {str(e)}")
            logging.error(f"Error in flip_video: {str(e)}")
            logging.error(traceback.format_exc())

    def flip_original_video(self):
        self.source_data, self.current_frame = self.flip_video(self.source_data, self.current_frame,
                                                               self.original_video_layout)

    def flip_previous_result_video(self):
        self.previous_result_data, self.previous_result_frame = self.flip_video(self.previous_result_data,
                                                                                self.previous_result_frame,
                                                                                self.previous_result_layout)

    def flip_current_result_video(self):
        self.current_result_data, self.current_result_frame = self.flip_video(self.current_result_data,
                                                                              self.current_result_frame,
                                                                              self.current_result_layout)

    def create_result_navigation(self):
        self.result_navigation_layout = QHBoxLayout()

        self.result_type_dropdown = QComboBox()
        self.result_type_dropdown.addItem("frames")
        self.result_type_dropdown.currentIndexChanged.connect(self.on_result_type_changed)

        self.result_navigation_layout.addWidget(QLabel("Result Type:"))
        self.result_navigation_layout.addWidget(self.result_type_dropdown)

        return self.result_navigation_layout

    def on_result_type_changed(self, index):
        self.current_result_type = self.result_type_dropdown.currentText()
        self.log_message(f"Result type changed to: '{self.current_result_type}'", "info")
        self.update_result_display()
        # self.update_additional_data_dropdown()

    def update_result_navigation(self):
        current_index = self.result_type_dropdown.findText(self.current_result_type)
        if current_index >= 0:
            self.result_type_dropdown.setCurrentIndex(current_index)
        self.result_type_dropdown.setEnabled(len(self.result_types) > 1)
        self.update_additional_data_dropdown()

    def update_result_display(self):
        self.log_message(f"Updating result display. Current result type: '{self.current_result_type}'", "info")
        self.log_message(f"Available result types: {self.result_types}", "info")
        self.log_message(f"Additional results keys: {list(self.additional_results.keys())}", "info")

        if not self.current_result_type:
            self.log_message("Current result type is empty. Defaulting to 'frames'", "warning")
            self.current_result_type = "frames"

        if self.current_result_type == "frames":
            self.current_result_data = self.enhanced_frames
        elif self.current_result_type in self.additional_results:
            self.current_result_data = self.additional_results[self.current_result_type]
        else:
            self.log_message(f"Invalid result type: '{self.current_result_type}'. Defaulting to 'frames'", "warning")
            self.current_result_type = "frames"
            self.current_result_data = self.enhanced_frames

        self.current_result_frame = 0
        self.refresh_ui()  # Add this line to update the additional data dropdown

    def get_default_parameters(self, option):
        if option in self.denoise_options_list:
            return self.default_denoise_params[option]
        elif option in self.deblur_options_list:
            return self.default_deblur_params[option]
        else:
            return {}

    def open_parameter_settings(self, option, params):
        if option in self.denoise_options_list:
            current_params = DenoiseOptions.OptionsServices.DICT[option]
        elif option in self.deblur_options_list:
            current_params = DeblurOptions.OptionsServices.DICT[option]
        else:
            current_params = params

        default_params = self.get_default_parameters(option)

        settings_window = ParameterSettingsWindow(option, current_params, default_params, self)
        if settings_window.exec_() == QDialog.Accepted:
            new_params = settings_window.get_parameters()
            if option in self.denoise_options_list or option in self.deblur_options_list:
                for param, value in new_params.items():
                    if isinstance(current_params[param], list):
                        # If the original parameter was a list, update it by moving the selected value to the front
                        current_list = current_params[param]
                        current_list.remove(value)
                        current_list.insert(0, value)
                    else:
                        # For non-list parameters, simply update the value
                        current_params[param] = value
            else:
                # For other cases, update the entire parameter dictionary
                params.update(new_params)

            print(f"New parameters for {option}: {current_params}")

    def show_loading_indicator(self):
        if not hasattr(self, 'loading_overlay'):
            self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.resize(self.size())
        self.loading_overlay.show()
        QApplication.processEvents()

    def hide_loading_indicator(self):
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide()
        QApplication.processEvents()

    def create_additional_data_dropdown(self):
        self.additional_data_dropdown = QComboBox()
        self.additional_data_dropdown.addItem("None")
        self.additional_data_dropdown.setEnabled(False)
        return self.additional_data_dropdown

    def update_additional_data_dropdown(self):
        self.additional_data_dropdown.clear()
        self.additional_data_dropdown.addItem("None")

        self.log_message(
            f"Updating additional data dropdown. Current result radio checked: {self.current_result_radio.isChecked()}",
            "info")
        self.log_message(f"Additional results keys: {self.additional_results.keys()}", "info")

        if self.current_result_radio.isChecked() and self.additional_results:
            for result_type, data in self.additional_results.items():
                if result_type != "frames":
                    if isinstance(data, dict):
                        for key in data.keys():
                            self.additional_data_dropdown.addItem(f"{result_type}: {key}")
                    else:
                        self.additional_data_dropdown.addItem(result_type)

            self.additional_data_dropdown.setEnabled(True)
        else:
            self.additional_data_dropdown.setEnabled(False)

        self.log_message(
            f"Updated additional data dropdown. Items: {[self.additional_data_dropdown.itemText(i) for i in range(self.additional_data_dropdown.count())]}",
            "info")

    def refresh_ui(self):
        self.update_all_video_windows()
        self.update_result_navigation()
        self.update_additional_data_dropdown()
        self.log_message("UI refreshed", "info")


class ZoomableImage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap()
        self.scale = 1.0
        self.pos = QPoint(0, 0)
        self.dragging = False
        self.last_pos = QPoint(0, 0)
        self.setMouseTracking(True)

    def setPixmap(self, pixmap):
        if pixmap.isNull():
            logging.error("Attempted to set null pixmap in ZoomableImage")
            return
        self.pixmap = pixmap
        self.pos = QPoint(0, 0)
        self.scale = 1.0
        self.update()

    def paintEvent(self, event):
        if self.pixmap.isNull():
            logging.warning("Attempting to paint null pixmap in ZoomableImage")
            return

        painter = QPainter(self)
        painter.translate(self.pos)
        painter.scale(self.scale, self.scale)
        painter.drawPixmap(0, 0, self.pixmap)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            zoom_factor = 1.1
            if event.angleDelta().y() < 0:
                zoom_factor = 1 / zoom_factor

            mouse_pos = event.pos()
            old_pos = (mouse_pos - self.pos) / self.scale
            self.scale *= zoom_factor
            new_pos = mouse_pos - old_pos * self.scale
            self.pos = QPoint(int(new_pos.x()), int(new_pos.y()))  # Ensure integer values

            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging:
            delta = event.pos() - self.last_pos
            self.pos += delta
            self.last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def centerImage(self):
        if self.pixmap.isNull():
            logging.warning("Attempting to center null pixmap in ZoomableImage")
            return
        self.scale = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        scaled_width = self.pixmap.width() * self.scale
        scaled_height = self.pixmap.height() * self.scale
        self.pos = QPoint(int(max(0, (self.width() - scaled_width) / 2)),
                          int(max(0, (self.height() - scaled_height) / 2)))
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self.centerImage()
        except Exception as e:
            logging.error(f"Error in resizeEvent: {str(e)}")


class ExpandedVideoDialog(QDialog):
    def __init__(self, original_frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Expanded Video")
        self.layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea(self)
        self.layout.addWidget(self.scroll_area)

        self.image_widget = ZoomableImage(self)
        self.scroll_area.setWidget(self.image_widget)
        self.scroll_area.setWidgetResizable(True)

        self.center_button = QPushButton("Center Image", self)
        self.center_button.clicked.connect(self.image_widget.centerImage)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.center_button)
        button_layout.addStretch()

        self.layout.addLayout(button_layout)

        self.original_frame = original_frame
        self.update_image()

        self.setMinimumSize(800, 600)

    def update_image(self):
        if self.original_frame is None:
            logging.error("Original frame is None in ExpandedVideoDialog")
            return

        try:
            height, width = self.original_frame.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(self.original_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            if pixmap.isNull():
                logging.error("Created pixmap is null in ExpandedVideoDialog")
                return
            self.image_widget.setPixmap(pixmap)
            self.image_widget.resize(pixmap.size())
            self.image_widget.centerImage()
        except Exception as e:
            logging.error(f"Error in update_image: {str(e)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.image_widget.centerImage()


class DraggableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.drag_start_position = QPoint()
        self.setCursor(QCursor(Qt.OpenHandCursor))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            diff = event.pos() - self.drag_start_position
            new_pos = self.pos() + diff
            self.move(new_pos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setCursor(QCursor(Qt.OpenHandCursor))


class DrawingCanvas(QLabel):
    def __init__(self, frame, drawing_option):
        super().__init__()
        self.original_frame = frame
        self.drawing_option = drawing_option
        self.points = []
        self.mask = None
        self.temp_point = None
        self.setMouseTracking(True)

        # Set a fixed size for the canvas
        max_width, max_height = 800, 600
        self.orig_height, self.orig_width = frame.shape[:2]
        self.scale_factor = min(max_width / self.orig_width, max_height / self.orig_height)
        self.canvas_width = int(self.orig_width * self.scale_factor)
        self.canvas_height = int(self.orig_height * self.scale_factor)
        self.setFixedSize(self.canvas_width, self.canvas_height)

        self.update_image()

    def update_image(self):
        image = self.original_frame.copy()
        if self.points:
            if self.drawing_option in ["Draw box", "Draw box and use mask"]:
                if len(self.points) == 2:
                    cv2.rectangle(image, self.points[0], self.points[1], (0, 255, 0), 2)
                elif len(self.points) == 1 and self.temp_point:
                    cv2.rectangle(image, self.points[0], self.temp_point, (0, 255, 0), 2)
            else:  # Polygon drawing
                pts = np.array(self.points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                if self.temp_point:
                    cv2.line(image, self.points[-1], self.temp_point, (0, 255, 0), 2)

        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.mask is None:
            x = int(event.x() / self.scale_factor)
            y = int(event.y() / self.scale_factor)
            if self.drawing_option in ["Draw box", "Draw box and use mask"]:
                if len(self.points) < 2:
                    self.points.append((x, y))
                    if len(self.points) == 2:
                        self.update_image()
                else:
                    self.points = [(x, y)]
            else:  # Polygon drawing
                self.points.append((x, y))
            self.update_image()

    def mouseMoveEvent(self, event):
        if self.mask is None:
            x = int(event.x() / self.scale_factor)
            y = int(event.y() / self.scale_factor)
            self.temp_point = (x, y)
            self.update_image()

    def mouseReleaseEvent(self, event):
        self.temp_point = None
        self.update_image()

    def display_mask(self, mask):
        self.mask = mask
        masked_image = self.original_frame.copy()
        masked_image[mask == 1] = [0, 255, 0]  # Green overlay for the mask
        cv2.addWeighted(masked_image, 0.5, self.original_frame, 0.5, 0, masked_image)

        height, width = masked_image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(masked_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

    def clear_canvas(self):
        self.points = []
        self.mask = None
        self.update_image()


class ModernDrawingDialog(QDialog):
    def __init__(self, frame, drawing_option, parent=None):
        super().__init__(parent)
        self.frame = frame
        self.drawing_option = drawing_option
        self.points = []
        self.mask = None
        self.setWindowTitle(f"Draw {drawing_option.split()[-1].capitalize()}")

        self.layout = QVBoxLayout(self)
        self.canvas = DrawingCanvas(self.frame, self.drawing_option)
        self.layout.addWidget(self.canvas)

        self.button_layout = QHBoxLayout()
        self.finish_button = QPushButton("Finish Drawing")
        self.finish_button.clicked.connect(self.finish_drawing)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.redraw_button = QPushButton("Redraw")
        self.redraw_button.clicked.connect(self.redraw)
        self.redraw_button.setEnabled(False)

        self.button_layout.addWidget(self.finish_button)
        self.button_layout.addWidget(self.redraw_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.button_layout)

        if self.drawing_option == "Draw box and use mask":
            self.approve_mask_button = QPushButton("Approve Mask")
            self.approve_mask_button.clicked.connect(self.approve_mask)
            self.approve_mask_button.setEnabled(False)
            self.button_layout.addWidget(self.approve_mask_button)

        self.adjustSize()

    def finish_drawing(self):
        self.points = self.canvas.points
        if self.drawing_option == "Draw box and use mask":
            if len(self.points) == 2:  # Ensure we have a valid box
                self.generate_mask()
            else:
                QMessageBox.warning(self, "Invalid Box", "Please draw a complete box before generating the mask.")
        else:
            self.accept()

    def generate_mask(self):
        try:
            box = self.get_minimal_bounding_box(self.points)
            print(f"Bounding Box: {box}")
            print(f"Points: {self.points}")
            print(f"Frame shape: {self.frame.shape}")

            # Ensure the box coordinates are within the image dimensions
            if (box[0] < 0 or box[1] < 0 or box[2] > self.frame.shape[1] or box[3] > self.frame.shape[0]):
                raise ValueError("Bounding box coordinates are out of image bounds")

            # Initialize progress dialog
            progress_dialog = QProgressDialog("Generating mask...", "Cancel", 0, 0, self)
            progress_dialog.setWindowTitle("Please Wait")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setCancelButton(None)
            progress_dialog.show()

            # Simulate a long task (e.g., the SAM process)
            QCoreApplication.processEvents()
            # time.sleep(0.5)  # Simulate processing time

            self.mask = get_mask_from_bbox(self.frame, PARAMETER.SAM_CHECKPOINTS, bbox=box)
            progress_dialog.close()  # Close the progress dialog

            if self.mask is not None:
                self.canvas.display_mask(self.mask)
                self.approve_mask_button.setEnabled(True)
                self.redraw_button.setEnabled(True)
                self.finish_button.setEnabled(False)
                # self.plot_mask()  # Call the plotting function
            else:
                raise ValueError("Mask generation returned None")
        except Exception as e:
            QMessageBox.warning(self, "Mask Generation Failed", f"Failed to generate mask: {str(e)}")
            logging.error(f"Error in generate_mask: {str(e)}")
            logging.error(traceback.format_exc())
            print(f"Error in generate_mask: {str(e)}")

    def approve_mask(self):
        self.accept()

    def redraw(self):
        self.canvas.clear_canvas()
        self.approve_mask_button.setEnabled(False)
        self.redraw_button.setEnabled(False)
        self.finish_button.setEnabled(True)

    def get_polygon(self):
        return self.points

    def get_box(self):
        return self.get_minimal_bounding_box(self.points)

    def get_mask(self):
        return self.mask

    def get_minimal_bounding_box(self, polygon):
        """
        Calculate the minimal bounding box of a given polygon.

        Parameters:
        polygon (list): A list of (x, y) tuples representing the polygon vertices.

        Returns:
        tuple: A tuple containing the minimal bounding box (x_min, y_min, x_max, y_max) in pixel coordinates.
        """
        x_coords, y_coords = zip(*polygon)
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)

        return int(x_min), int(y_min), int(x_max), int(y_max)


from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QDialogButtonBox, QComboBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt
import logging


class ParameterSettingsWindow(QDialog):
    def __init__(self, option_name, current_params, default_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Parameters for {option_name}")
        self.option_name = option_name
        self.current_params = current_params.copy()  # Create a copy to avoid modifying the original
        self.default_params = default_params
        self.param_inputs = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        for param, current_value in self.current_params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(param))

            try:
                if isinstance(current_value, list):
                    combo = QComboBox()
                    combo.addItems([str(item) for item in current_value])
                    default_index = 0
                    if param in self.default_params:
                        try:
                            default_index = current_value.index(self.default_params[param])
                        except ValueError:
                            logging.warning(f"Default value for {param} not found in options. Using first value.")
                    combo.setCurrentIndex(default_index)
                    self.param_inputs[param] = combo
                    param_layout.addWidget(combo)
                elif isinstance(current_value, int):
                    spin = QSpinBox()
                    spin.setRange(-1000000, 1000000)
                    spin.setValue(current_value)
                    self.param_inputs[param] = spin
                    param_layout.addWidget(spin)
                elif isinstance(current_value, float):
                    spin = QDoubleSpinBox()
                    spin.setRange(-1000000, 1000000)
                    spin.setDecimals(4)
                    spin.setValue(current_value)
                    self.param_inputs[param] = spin
                    param_layout.addWidget(spin)
                else:
                    line_edit = QLineEdit(str(current_value))
                    self.param_inputs[param] = line_edit
                    param_layout.addWidget(line_edit)
            except Exception as e:
                logging.error(f"Error setting up parameter {param}: {str(e)}")
                line_edit = QLineEdit(str(current_value))
                self.param_inputs[param] = line_edit
                param_layout.addWidget(line_edit)

            layout.addLayout(param_layout)

        reset_button = QPushButton("Reset to Default")
        reset_button.clicked.connect(self.reset_to_default)
        layout.addWidget(reset_button)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def reset_to_default(self):
        for param, default_value in self.default_params.items():
            if param in self.param_inputs:
                input_widget = self.param_inputs[param]
                try:
                    if isinstance(input_widget, QComboBox):
                        index = input_widget.findText(str(default_value))
                        if index >= 0:
                            input_widget.setCurrentIndex(index)
                    elif isinstance(input_widget, (QSpinBox, QDoubleSpinBox)):
                        input_widget.setValue(default_value)
                    else:
                        input_widget.setText(str(default_value))
                except Exception as e:
                    logging.error(f"Error resetting parameter {param}: {str(e)}")

    def get_parameters(self):
        params = {}
        for param, input_widget in self.param_inputs.items():
            try:
                if isinstance(input_widget, QComboBox):
                    # For dropdowns, return only the selected value
                    selected_item = input_widget.currentText()
                    # Convert the selected item to its original type
                    original_type = type(self.current_params[param][0])
                    params[param] = original_type(selected_item) if original_type != str else selected_item
                elif isinstance(input_widget, (QSpinBox, QDoubleSpinBox)):
                    params[param] = input_widget.value()
                else:
                    params[param] = input_widget.text()
            except Exception as e:
                logging.error(f"Error getting parameter {param}: {str(e)}")
                params[param] = self.current_params[param]  # Fallback to original value
        return params


class EnhancementProcess(Process):
    def __init__(self, method, input_dict, options_type, result_queue):
        super().__init__()
        self.method = method
        self.input_dict = input_dict
        self.options_type = options_type
        self.result_queue = result_queue

    def run(self):
        try:
            print(f"Enhancement process started. Input dict keys: {self.input_dict.keys()}")
            print(f"Added data in input_dict: {type(self.input_dict.get('added_data'))}")

            if self.options_type == 'denoise':
                from options import DenoiseOptions
                options = DenoiseOptions(print)
            else:
                from options import DeblurOptions
                options = DeblurOptions(print)

            enhancement_result = getattr(options, f"perform_{self.method}")(self.input_dict)

            pickled_result = pickle.dumps(enhancement_result)
            self.result_queue.put(("success", pickled_result))
        except Exception as e:
            self.result_queue.put(("error", str(e)))
        finally:
            self.result_queue.put(("finished", None))  # Signal that the process has finished


class EnhancementThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, method, input_dict, options_type):
        super().__init__()
        self.method = method
        self.input_dict = input_dict
        self.options_type = options_type
        self.process = None
        self.result_queue = Queue()
        self.stopped = False

    def run(self):
        self.process = EnhancementProcess(self.method, self.input_dict, self.options_type, self.result_queue)
        self.process.start()

        while True:
            if not self.result_queue.empty():
                result_type, result_data = self.result_queue.get()
                if result_type == "success":
                    unpickled_result = pickle.loads(result_data)
                    self.finished.emit(unpickled_result)
                    break
                elif result_type == "error":
                    self.error.emit(result_data)
                    break
                elif result_type == "finished":
                    if not self.stopped:
                        self.error.emit("Process finished without returning a result")
                    break

            if self.stopped:
                self.process.terminate()
                self.error.emit("Process was stopped by user")
                break

            self.process.join(timeout=0.1)
            if not self.process.is_alive():
                if not self.stopped:
                    self.error.emit("Process ended unexpectedly")
                break

        self.process.join()  # Ensure the process is fully terminated

    def stop(self):
        self.stopped = True


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint)

        self.label = QLabel(self)
        self.movie = QMovie("icons/loading.gif")
        self.label.setMovie(self.movie)
        self.movie.setScaledSize(QSize(50, 50))

        layout = QVBoxLayout(self)
        layout.addWidget(self.label, alignment=Qt.AlignCenter)

    def showEvent(self, event):
        super().showEvent(event)
        self.movie.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.movie.stop()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))


class EnhancedScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)

    def wheelEvent(self, event: QWheelEvent):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            # Horizontal scrolling
            delta = event.angleDelta().y()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta
            )
        else:
            # Default vertical scrolling
            super().wheelEvent(event)


if __name__ == '__main__':
    myappid = 'com.shaback.videoenhancement.v1'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icons/logo.png'))
    window = VideoEnhancementApp()
    window.show()
    sys.exit(app.exec_())
