from Segmentation.sam import get_mask_from_bbox
import PARAMETER
import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QTextEdit, QCheckBox, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QSlider
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt
from datetime import datetime
from options import DenoiseOptions, DeblurOptions
# Default directory for data
default_directory = "../data"
fixed_size = (640, 480)  # Fixed size for displayed images
import logging
import traceback
from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QGroupBox,QScrollArea
import numpy as np



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
        self.current_result_layout = self.create_video_layout("Current Result")
        self.video_display_layout.addLayout(self.current_result_layout[0])

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

        # Add checkbox for polygon drawing
        self.draw_polygon_checkbox = QCheckBox("Draw polygon/box for enhancement")
        self.main_layout.addWidget(self.draw_polygon_checkbox)

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

        Parameters:
        polygon (Polygon): A shapely Polygon object.

        Returns:
        tuple: A tuple containing the minimal bounding box (orig_x, orig_y, orig_w, orig_h) in pixel coordinates.
        """
        # Get the bounding box of the polygon
        min_x, min_y, max_x, max_y = polygon.bounds

        # Calculate width and height
        orig_w = max_x - min_x
        orig_h = max_y - min_y

        # Convert to integers
        orig_x, orig_y, orig_w, orig_h = map(int, [min_x, min_y, orig_w, orig_h])

        return orig_x, orig_y, orig_w, orig_h

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

            # Determine which video to use as input
            if self.original_video_radio.isChecked():
                input_frames = self.source_data
                self.log_message("Using original video for enhancement", "info")
            elif self.previous_result_radio.isChecked():
                if isinstance(self.previous_result_data, (list, np.ndarray)) and len(self.previous_result_data) > 0:
                    input_frames = self.previous_result_data
                    self.log_message("Using previous result for enhancement", "info")
                else:
                    self.log_message("No previous result available. Using original video.", "warning")
                    input_frames = self.source_data
            elif self.current_result_radio.isChecked():
                if isinstance(self.current_result_data, (list, np.ndarray)) and len(self.current_result_data) > 0:
                    input_frames = self.current_result_data
                    self.log_message("Using current result for enhancement", "info")
                else:
                    self.log_message("No current result available. Using original video.", "warning")
                    input_frames = self.source_data
            else:
                raise ValueError("No video source selected")

            self.log_message(f"Input frames type: {type(input_frames)}, Length: {len(input_frames)}", "info")

            # Ensure input_frames is a list of numpy arrays
            if isinstance(input_frames, np.ndarray):
                if input_frames.ndim == 4:
                    input_frames = list(input_frames)
                elif input_frames.ndim == 3:
                    input_frames = [input_frames]  # Single frame
                else:
                    raise ValueError(f"Unexpected input_frames shape: {input_frames.shape}")
            elif not isinstance(input_frames, list):
                raise ValueError(f"Invalid input frames format: {type(input_frames)}")

            input_frames = [np.array(frame) if not isinstance(frame, np.ndarray) else frame for frame in input_frames]

            self.log_message(f"First frame shape: {input_frames[0].shape}, dtype: {input_frames[0].dtype}", "info")

            polygon, mask, box, input_method, user_input = None, None, None, None, None

            if self.draw_polygon_checkbox.isChecked():
                first_frame = input_frames[0].copy()
                if first_frame.ndim == 2:
                    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
                elif first_frame.ndim == 3 and first_frame.shape[2] == 4:
                    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGBA2BGR)

                self.log_message(f"Frame for drawing dialog: shape {first_frame.shape}, dtype: {first_frame.dtype}",
                                 "info")

                dialog = DrawingDialog(first_frame)
                if dialog.exec_() == QDialog.Accepted:
                    polygon = dialog.get_polygon()
                    try:
                        try:
                            box = self.get_minimal_bounding_box(polygon)
                        except:
                            logging.error("Error calculating minimal bounding box")
                            box = cv2.boundingRect(np.array(polygon))
                        logging.debug("Generating mask...")
                        mask = get_mask_from_bbox(first_frame, PARAMETER.SAM_CHECKPOINTS, bbox=box)
                        logging.debug("Mask generated successfully")
                    except:
                        logging.error("Error generating mask")

                    self.log_message(f"Polygon drawn: {polygon}, converted to box: {box}", "info")

                    # Let the user choose what to send to the function
                    choice_dialog = QDialog(self)
                    choice_dialog.setWindowTitle("Choose what to send")
                    choice_layout = QVBoxLayout(choice_dialog)

                    mask_btn = QPushButton("Send Mask")
                    polygon_btn = QPushButton("Send Polygon")
                    box_btn = QPushButton("Send Box")

                    choice_layout.addWidget(mask_btn)
                    choice_layout.addWidget(polygon_btn)
                    choice_layout.addWidget(box_btn)

                    mask_btn.clicked.connect(lambda: choice_dialog.done(1))
                    polygon_btn.clicked.connect(lambda: choice_dialog.done(2))
                    box_btn.clicked.connect(lambda: choice_dialog.done(3))

                    choice = choice_dialog.exec_()

                    if choice == 1:
                        self.log_message("Sending mask to enhancement function", "info")
                        polygon, box = None, None
                        input_method = PARAMETER.InputMethodForServices.segmentation
                        user_input = mask
                    elif choice == 2:
                        self.log_message("Sending polygon to enhancement function", "info")
                        mask, box = None, None
                        input_method = PARAMETER.InputMethodForServices.polygon
                        user_input = polygon
                    elif choice == 3:
                        self.log_message("Sending box to enhancement function", "info")
                        mask, polygon = None, None
                        input_method = PARAMETER.InputMethodForServices.BB
                        user_input = box
                    else:
                        self.log_message("No selection made, using all available data", "info")
            else:
                input_method = None
                user_input = None

            # Perform the selected enhancement method
            self.log_message(f"Starting enhancement: {method}", "info")

            if method in self.denoise_options_list:
                enhanced_frames = getattr(self.denoise_options, f"perform_{method}")(self.source_path, input_frames,
                                                                                     input_method=input_method,
                                                                                     user_input=user_input)
            elif method in self.deblur_options_list:
                enhanced_frames = getattr(self.deblur_options, f"perform_{method}")(self.source_path, input_frames,
                                                                                    input_method=input_method,
                                                                                    user_input=user_input)
            else:
                raise ValueError(f"Unknown method: {method}")

            self.log_message(f"Enhancement completed. Result type: {type(enhanced_frames)}", "info")

            if enhanced_frames is None:
                raise ValueError("Enhancement method returned None")

            # Ensure enhanced_frames is a list of numpy arrays
            if isinstance(enhanced_frames, np.ndarray):
                if enhanced_frames.ndim == 4:
                    enhanced_frames = list(enhanced_frames)
                elif enhanced_frames.ndim == 3:
                    enhanced_frames = [enhanced_frames]
                else:
                    raise ValueError(f"Unexpected enhanced_frames shape: {enhanced_frames.shape}")
            elif not isinstance(enhanced_frames, list):
                raise ValueError(f"Invalid enhanced frames format: {type(enhanced_frames)}")

            self.log_message(f"Enhanced frames shape: {enhanced_frames[0].shape}, dtype: {enhanced_frames[0].dtype}",
                             "info")

            # Update video windows
            self.previous_result_data = self.current_result_data if self.current_result_data else self.source_data
            self.current_result_data = enhanced_frames

            self.previous_result_frame = 0
            self.current_result_frame = 0

            self.update_all_video_windows()
            self.enable_all_video_controls()

            self.log_message(f"Enhancement completed: {len(enhanced_frames)} frames processed", "info")

            # Enable the options to use previously enhanced or current result video
            self.previous_result_radio.setEnabled(True)
            self.current_result_radio.setEnabled(True)

        except Exception as e:
            self.log_message(f"Error in perform_enhancement: {str(e)}", "danger")
            logging.error(f"Error in perform_enhancement: {str(e)}")
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
            from Video_Editor import VideoEditor  # Import here to avoid circular import
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

    def update_video_window(self, frames, current_frame, image_label, prev_btn, next_btn, slider, counter, expand_btn,
                            flip_btn):
        if isinstance(frames, np.ndarray):
            if frames.ndim == 4:
                frames = list(frames)
            elif frames.ndim == 3:
                frames = [frames]
            else:
                raise ValueError(f"Unexpected frames shape: {frames.shape}")

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

    def flip_video(self, video_data, current_frame, layout):
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

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint

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

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QScrollArea, QLabel
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QCursor

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
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoEnhancementApp()
    window.show()
    sys.exit(app.exec_())