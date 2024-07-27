from Segmentation.sam import get_mask_from_bbox
import sys
import cv2
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QTextEdit, QCheckBox,
                             QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QSlider, QMessageBox, QComboBox)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt
from datetime import datetime
from options import DenoiseOptions, DeblurOptions
import PARAMETER
import logging
import traceback
from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QGroupBox, QScrollArea
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QSlider, QScrollArea, QLabel
from PyQt5.QtCore import Qt, QPoint,QCoreApplication
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import QProgressDialog


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
        self.setWindowTitle("Video Enhancement Tool")
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
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QLabel {
                font-size: 14px;
                color: #333333;
            }
            QSlider::groove:horizontal {
                border: 1px solid #CCCCCC;
                background: white;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #333333;
                border: 1px solid #333333;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QComboBox, QLineEdit {
                padding: 6px;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                background-color: white;
            }
        """)

        self.source_path = ""
        self.source_data = []
        self.previous_result_data = []
        self.current_result_data = []

        self.current_frame = 0
        self.previous_result_frame = 0
        self.current_result_frame = 0

        self.denoise_options = DenoiseOptions(self.log_message)
        self.deblur_options = DeblurOptions(self.log_message)

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

        # Create a scroll area for the main tab
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
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

        # Denoise options
        denoise_box = QGroupBox("Denoise Methods")
        denoise_layout = QVBoxLayout()
        self.denoise_options_list = [DenoiseOptions.OptionsServices.SCC, DenoiseOptions.OptionsServices.ECC,
                                     DenoiseOptions.OptionsServices.FEATURE_BASED,
                                     DenoiseOptions.OptionsServices.OPTICAL_FLOW,
                                     DenoiseOptions.OptionsServices.DENOISE_COT,
                                     DenoiseOptions.OptionsServices.RV_CLASSIC,
                                     DenoiseOptions.OptionsServices.DENOISE_YOV]

        for i, option in enumerate(self.denoise_options_list):
            radio_btn = QRadioButton(option, self)
            self.enhancement_button_group.addButton(radio_btn, i)
            denoise_layout.addWidget(radio_btn)
        denoise_box.setLayout(denoise_layout)
        self.enhancement_group.addWidget(denoise_box)

        # Deblur options
        deblur_box = QGroupBox("Deblur Methods")
        deblur_layout = QVBoxLayout()
        self.deblur_options_list = [DeblurOptions.OptionsServices.RV_OM,
                                    DeblurOptions.OptionsServices.NAFNET,
                                    DeblurOptions.OptionsServices.NUBKE,
                                    DeblurOptions.OptionsServices.NUMBKE2WIN,
                                    DeblurOptions.OptionsServices.UNSUPERWIN,
                                    DeblurOptions.OptionsServices.STRETCH,
                                    DeblurOptions.OptionsServices.SHARPEN,
                                    DeblurOptions.OptionsServices.MEAN]

        for i, option in enumerate(self.deblur_options_list, start=len(self.denoise_options_list)):
            radio_btn = QRadioButton(option, self)
            self.enhancement_button_group.addButton(radio_btn, i)
            deblur_layout.addWidget(radio_btn)
        deblur_box.setLayout(deblur_layout)
        self.enhancement_group.addWidget(deblur_box)

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
        image_label.setStyleSheet("background-color: #1C1C1C;")
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
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", default_directory,
                                                   "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
        if file_path:
            try:
                self.source_path = file_path
                self.log_message(f"Video uploaded: {self.source_path}", "info")
                self.source_data = get_video_frames(file_path)
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
                self.log_message(f"Error loading video: {str(e)}", "danger")
                self.show_error_message("Error loading video", str(e))
                logging.error(f"Error in upload_source: {str(e)}")
                logging.error(traceback.format_exc())

    def expand_video(self, image_label, title):
        original_frame = image_label.property("original_frame")
        if original_frame is not None:
            dialog = ExpandedVideoDialog(original_frame, self)
            dialog.resize(800, 600)  # Set a default size, adjust as needed
            dialog.exec_()

    def disable_result_video_controls(self):
        self.disable_video_controls(*self.previous_result_layout[2:])
        self.disable_video_controls(*self.current_result_layout[2:])

    def zoom_image(self, label, zoom_factor):
        original_pixmap = label.pixmap()
        scaled_pixmap = original_pixmap.scaled(
            original_pixmap.size() * zoom_factor / 100,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

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
                input_frames = self.get_input_frames()
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

            polygon, mask, box, input_method, user_input = None, None, None, None, None

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

            # Perform the selected enhancement method
            self.log_message(f"Starting enhancement: {method}", "info")

            input_dict = {PARAMETER.DictInput.frames: input_frames,
                          PARAMETER.DictInput.input_method: input_method,
                          PARAMETER.DictInput.user_input: user_input}
            if method in self.denoise_options_list:
                enhancement_result = getattr(self.denoise_options, f"perform_{method}")(input_dict)
            elif method in self.deblur_options_list:
                enhancement_result = getattr(self.deblur_options, f"perform_{method}")(input_dict)
            else:
                raise ValueError(f"Unknown method: {method}")

            self.log_message(f"Enhancement completed. Result type: {type(enhancement_result)}", "info")

            # Step 5: Process enhancement result
            try:
                new_frames, additional_results = self.process_enhancement_result(enhancement_result)
            except Exception as e:
                raise ValueError(f"Error processing enhancement result: {str(e)}")

            self.log_message(f"Processed frames: {len(new_frames)}", "info")

            try:
                self.update_ui_with_results(new_frames, additional_results)
            except Exception as e:
                raise ValueError(f"Error updating UI with results: {str(e)}")

        except Exception as e:
            self.log_message(f"Error in perform_enhancement: {str(e)}", "danger")
            self.show_error_message("Error in Enhancement", f"An error occurred during enhancement: {str(e)}")
            logging.error(f"Error in perform_enhancement: {str(e)}")
            logging.error(traceback.format_exc())
        finally:
            self.log_message("Enhancement process finished", "info")

    def update_ui_with_results(self, new_frames, additional_results):
        self.previous_result_data = self.current_result_data if isinstance(self.current_result_data,
                                                                           list) else self.source_data
        self.current_result_data = new_frames
        self.enhanced_frames = new_frames
        self.additional_results = additional_results

        self.result_types = ["frames"] + list(additional_results.keys())
        self.current_result_type = "frames"

        self.result_type_dropdown.clear()
        self.result_type_dropdown.addItems(self.result_types)
        self.result_type_dropdown.setCurrentIndex(0)

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

    def get_input_frames(self):
        if self.original_video_radio.isChecked():
            return self.source_data
        elif self.previous_result_radio.isChecked() and self.previous_result_data:
            return self.previous_result_data
        elif self.current_result_radio.isChecked() and self.current_result_data:
            return self.current_result_data
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
                video_data = self.current_result_data
                video_name = "Current Result Video"
            else:
                raise ValueError("Invalid choice")

            if video_data is None or (isinstance(video_data, (list, np.ndarray)) and len(video_data) == 0):
                self.log_message(f"No {video_name} data available for editing.", "warning")
                self.show_error_message("No Video Data", f"Please ensure {video_name} exists before editing.")
                return

            self.log_message(f"Initializing Video Editor for {video_name}...", "info")
            from Video_Editor_3 import VideoEditor
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
                    lambda new_frames: self.update_edited_video(new_frames, choice))
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
                self.current_result_data = new_frames
                self.current_result_frame = 0
                self.log_message("Current result video updated with edited version.", "info")

            self.update_all_video_windows()

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

    def update_video(self, new_frames):
        try:
            logging.info(f"Updating video with {len(new_frames)} new frames")
            self.source_data = new_frames
            self.current_frame = 0
            self.previous_result_data = []
            self.current_result_data = []
            self.previous_result_frame = 0
            self.current_result_frame = 0

            if self.source_data:
                self.update_all_video_windows()
                self.enable_original_video_controls()
                self.disable_result_video_controls()

                # Reset and disable the advanced options
                self.previous_result_radio.setEnabled(False)
                self.current_result_radio.setEnabled(False)
                self.original_video_radio.setChecked(True)

            self.log_message(f"Video updated with edited version. New frame count: {len(new_frames)}", "info")
        except Exception as e:
            logging.error(f"Error in update_video: {str(e)}")
            logging.error(traceback.format_exc())
            self.log_message(f"Error updating video: {str(e)}", "danger")
            self.show_error_message("Error Updating Video", f"An error occurred while updating the video: {str(e)}")

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
        self.update_video_window(self.current_result_data, self.current_result_frame, *self.current_result_layout[1:])

    def update_slider(self):
        if self.source_data:
            self.frame_slider.setRange(0, len(self.source_data) - 1)
            self.frame_slider.setValue(self.current_frame)
            self.frame_counter.setText(f"Frame: {self.current_frame + 1} / {len(self.source_data)}")

    def enable_video_controls(self, prev_btn, next_btn, slider, counter, expand_btn, flip_btn):
        prev_btn.setEnabled(True)
        next_btn.setEnabled(True)
        slider.setEnabled(True)
        expand_btn.setEnabled(True)
        flip_btn.setEnabled(True)

    def enable_original_video_controls(self):
        self.enable_video_controls(*self.original_video_layout[2:])

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

    def choose_video_to_edit(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Choose Video to Edit")
        layout = QVBoxLayout(dialog)

        original_btn = QPushButton("Edit Original Video")
        previous_result_btn = QPushButton("Edit Previous Result Video")
        current_result_btn = QPushButton("Edit Current Result Video")

        original_btn.clicked.connect(lambda: dialog.done(1))
        previous_result_btn.clicked.connect(lambda: dialog.done(2))
        current_result_btn.clicked.connect(lambda: dialog.done(3))

        layout.addWidget(original_btn)
        layout.addWidget(previous_result_btn)
        layout.addWidget(current_result_btn)

        # Enable/disable buttons based on video availability
        original_btn.setEnabled(bool(self.source_data))
        previous_result_btn.setEnabled(bool(self.previous_result_data))
        current_result_btn.setEnabled(bool(self.current_result_data))

        return dialog.exec_()

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

    def update_result_navigation(self):
        current_index = self.result_type_dropdown.findText(self.current_result_type)
        if current_index >= 0:
            self.result_type_dropdown.setCurrentIndex(current_index)
        self.result_type_dropdown.setEnabled(len(self.result_types) > 1)

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
        self.update_all_video_windows()
        self.update_result_navigation()


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPainter, QPen, QColor


class DrawingDialog(QDialog):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.frame = frame
        self.points = []
        self.setWindowTitle("Draw Polygon/Box")
        self.setGeometry(100, 100, frame.shape[1], frame.shape[0])

        layout = QVBoxLayout()
        self.finish_button = QPushButton("Finish Drawing")
        self.finish_button.clicked.connect(self.accept)
        layout.addWidget(self.finish_button)
        self.setLayout(layout)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.frame.ndim == 2:  # Grayscale image
            image = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], self.frame.strides[0],
                           QImage.Format_Grayscale8)
        elif self.frame.shape[2] == 3:  # RGB image
            image = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], self.frame.strides[0],
                           QImage.Format_RGB888).rgbSwapped()
        elif self.frame.shape[2] == 4:  # RGBA image
            image = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], self.frame.strides[0],
                           QImage.Format_RGBA8888)
        else:
            raise ValueError(f"Unsupported image format: {self.frame.shape}")

        painter.drawImage(self.rect(), image)

        if len(self.points) > 0:
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            for i in range(len(self.points) - 1):
                painter.drawLine(self.points[i], self.points[i + 1])
            if len(self.points) > 2:
                painter.drawLine(self.points[-1], self.points[0])

    def mousePressEvent(self, event):
        self.points.append(QPoint(event.x(), event.y()))
        self.update()

    def get_polygon(self):
        return [(p.x(), p.y()) for p in self.points]


class ExpandedVideoDialog(QDialog):
    def __init__(self, original_frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Expanded Video")
        self.layout = QVBoxLayout(self)

        # Create a scroll area
        self.scroll_area = QScrollArea(self)
        self.layout.addWidget(self.scroll_area)

        # Create a widget to hold the image
        self.image_widget = DraggableLabel(self)
        self.image_widget.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_widget)

        # Set the original frame
        self.original_frame = original_frame
        self.current_zoom = 100
        self.update_image()

        # Add zoom slider
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(100, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickInterval(50)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.zoom_image)
        self.layout.addWidget(self.zoom_slider)

    def update_image(self):
        if self.original_frame is not None:
            height, width = self.original_frame.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(self.original_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(pixmap.size() * self.current_zoom / 100,
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
            self.image_widget.setPixmap(scaled_pixmap)
            self.image_widget.resize(scaled_pixmap.size())

    def zoom_image(self, value):
        self.current_zoom = value
        self.update_image()


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


import matplotlib.pyplot as plt
import time


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

        PARAMETER:
        polygon (list): A list of (x, y) tuples representing the polygon vertices.

        Returns:
        tuple: A tuple containing the minimal bounding box (x_min, y_min, x_max, y_max) in pixel coordinates.
        """
        x_coords, y_coords = zip(*polygon)
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)

        return int(x_min), int(y_min), int(x_max), int(y_max)

    def plot_mask(self):
        """ Plot the mask and the bounding box for debugging purposes. """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Original Frame
        ax[0].imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Original Frame")
        for point in self.points:
            ax[0].plot(point[0], point[1], 'ro')
        if len(self.points) > 1:
            rect = plt.Rectangle(self.points[0], self.points[1][0] - self.points[0][0],
                                 self.points[1][1] - self.points[0][1],
                                 linewidth=1, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)

        # Mask
        masked_image = self.frame.copy()
        masked_image[self.mask == 1] = [0, 255, 0]
        ax[1].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title("Mask Applied")

        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoEnhancementApp()
    window.show()
    sys.exit(app.exec_())
