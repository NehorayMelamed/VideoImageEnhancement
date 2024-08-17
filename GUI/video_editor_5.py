from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QTimer, QThread, QSize
from PyQt5.QtGui import QImage, QPixmap, QPen, QMovie, QColor, QPainter, QMovie
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, \
    QLabel, QSlider, QComboBox, QLineEdit, QScrollArea, QDialog, QInputDialog, QRubberBand, QMessageBox, \
    QProgressDialog, QFrame, QProgressBar

import torch
import cv2
import numpy as np
import traceback
import logging
import sys

from Detection.yolo_world import get_bounding_boxes
from Detection.dino import detect_objects_dino, load_model

import PARAMETER
from PyQt5.QtGui import QIcon

from Tracking.croper_tracker import Tracker

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def frames_to_constant_format(frames, dtype_requested='uint8', range_requested=[0, 255], channels_requested=3,
                              threshold=5):
    """
    Process a list of frames to match the requested dtype, range, and number of channels.

    Args:
        frames (list of np.ndarray): List of input frames (numpy arrays).
        dtype_requested (str, optional): Requested data type ('uint8' or 'float'). Default is 'uint8'.
        range_requested (list, optional): Requested range ([0, 255] or [0, 1]). Default is [0, 255].
        channels_requested (int, optional): Requested number of channels (1 or 3). Default is 3.
        threshold (int, optional): Threshold for determining the input range. Default is 5.

    Returns:
        list of np.ndarray: List of processed frames matching the requested dtype, range, and number of channels.

    This function performs the following steps:
        1. Analyzes the first frame to determine the original number of channels, dtype, and range.
        2. Converts each frame to the requested number of channels using RGB2BW or BW2RGB if needed.
        3. Converts each frame to the requested range ([0, 255] or [0, 1]).
        4. Converts each frame to the requested dtype ('uint8' or 'float').
    """

    ### Analyze First Frame: ###
    first_frame = frames[0]  # Get the first frame for analysis
    original_dtype = first_frame.dtype  # Determine the original dtype of the first frame
    original_channels = first_frame.shape[2] if len(
        first_frame.shape) == 3 else 1  # Determine the original number of channels

    if original_dtype == np.uint8:  # Check if the original dtype is uint8
        original_range = [0, 255]  # Set original range to [0, 255]
    else:
        max_val = np.max(first_frame)  # Get the maximum value of the first frame
        original_range = [0, 255] if max_val > threshold else [0,
                                                               1]  # Determine the original range based on max value and threshold

    processed_frames = []  # Initialize list to store processed frames

    ### Process Each Frame: ###
    for frame in frames:  # Loop through each frame in the list

        ### Convert Number of Channels if Needed: ###
        if original_channels != channels_requested:  # Check if channel conversion is needed
            if channels_requested == 1:
                frame = RGB2BW(frame)  # Convert to grayscale
            else:
                frame = BW2RGB(frame)  # Convert to RGB

        ### Convert Range if Needed: ###
        if original_range != range_requested:  # Check if range conversion is needed
            if original_range == [0, 255] and range_requested == [0, 1]:
                frame = frame / 255.0  # Convert range from [0, 255] to [0, 1]
            elif original_range == [0, 1] and range_requested == [0, 255]:
                frame = frame * 255.0  # Convert range from [0, 1] to [0, 255]

        ### Convert Dtype if Needed: ###
        if original_dtype != dtype_requested:  # Check if dtype conversion is needed
            frame = frame.astype(dtype_requested)  # Convert dtype

        processed_frames.append(frame)  # Add the processed frame to the list

    return processed_frames  # Return the list of processed frames


def RGB2BW(input_image):
    if len(input_image.shape) == 2:
        return input_image

    if len(input_image.shape) == 3:
        if type(input_image) == torch.Tensor and input_image.shape[0] == 3:
            grayscale_image = 0.299 * input_image[0:1, :, :] + 0.587 * input_image[1:2, :, :] + 0.114 * input_image[2:3,
                                                                                                        :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1] + 0.587 * input_image[:, :, 1:2] + 0.114 * input_image[:,
                                                                                                        :, 2:3]
        else:
            grayscale_image = input_image

    elif len(input_image.shape) == 4:
        if type(input_image) == torch.Tensor and input_image.shape[1] == 3:
            grayscale_image = 0.299 * input_image[:, 0:1, :, :] + 0.587 * input_image[:, 1:2, :,
                                                                          :] + 0.114 * input_image[:, 2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, :, 0:1] + 0.587 * input_image[:, :, :,
                                                                          1:2] + 0.114 * input_image[:, :, :, 2:3]
        else:
            grayscale_image = input_image

    elif len(input_image.shape) == 5:
        if type(input_image) == torch.Tensor and input_image.shape[2] == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1, :, :] + 0.587 * input_image[:, :, 1:2, :,
                                                                             :] + 0.114 * input_image[:, :, 2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, :, :, 0:1] + 0.587 * input_image[:, :, :, :,
                                                                             1:2] + 0.114 * input_image[:, :, :, :, 2:3]
        else:
            grayscale_image = input_image

    return grayscale_image


def BW2RGB(input_image):
    ### For Both Torch Tensors and Numpy Arrays!: ###
    # Actually... we're not restricted to RGB....
    if len(input_image.shape) == 2:
        if type(input_image) == torch.Tensor:
            RGB_image = input_image.unsqueeze(0)
            RGB_image = torch.cat([RGB_image, RGB_image, RGB_image], 0)
        elif type(input_image) == np.ndarray:
            RGB_image = np.atleast_3d(input_image)
            RGB_image = np.concatenate([RGB_image, RGB_image, RGB_image], -1)
        return RGB_image

    if len(input_image.shape) == 3:
        if type(input_image) == torch.Tensor and input_image.shape[0] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 0)
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 1:
            RGB_image = np.concatenate([input_image, input_image, input_image], -1)
        else:
            RGB_image = input_image

    elif len(input_image.shape) == 4:
        if type(input_image) == torch.Tensor and input_image.shape[1] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 1)
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 1:
            RGB_image = np.concatenate([input_image, input_image, input_image], -1)
        else:
            RGB_image = input_image

    elif len(input_image.shape) == 5:
        if type(input_image) == torch.Tensor and input_image.shape[2] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 2)
        else:
            RGB_image = input_image

    return RGB_image


class VideoEditor(QMainWindow):
    editing_finished = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Editor")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("""
                QMainWindow {
                    background-color: #F8F8F8;  /* Soft white background */
                }
                QPushButton {
                    background-color: #333333;  /* Dark gray, almost black */
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
                    background-color: #555555;  /* Lighter gray for hover effect */
                }
                QLabel {
                    font-size: 14px;
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
                }
            """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.video_path = None
        self.cap = None
        self.current_frame = 0
        self.frames = []
        self.trimmed_frames = []
        self.processed_frames = []
        self.mode = None

        self.undo_stack = []
        self.redo_stack = []

        self.toolbar = self.addToolBar("Tools")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(24, 24))

        self.init_ui()
        self.progress_dialog = None
        self.progress_timer = None

    def init_ui(self):
        # Toolbar actions
        self.upload_action = self.toolbar.addAction(QIcon("icons/upload.png"), "Upload")
        self.upload_action.triggered.connect(self.upload_video)
        self.add_separator(self.toolbar, 12, line_before=True)

        self.trim_action = self.toolbar.addAction(QIcon("icons/trim.png"), "Trim")
        self.trim_action.triggered.connect(self.trim_video)

        self.crop_action = self.toolbar.addAction(QIcon("icons/crop.png"), "Crop")
        self.crop_action.triggered.connect(self.crop_video)
        self.add_separator(self.toolbar, 12, line_before=False)
        self.add_separator(self.toolbar, 12, add_line=False)

        self.rotate_action = self.toolbar.addAction(QIcon("icons/rotate.png"), "Rotate")
        self.rotate_action.triggered.connect(self.rotate_video)

        self.scale_action = self.toolbar.addAction(QIcon("icons/scale.png"), "Scale")
        self.scale_action.triggered.connect(self.scale_video)

        self.aspect_ratio_action = self.toolbar.addAction(QIcon("icons/aspect_ratio.png"), "Change Aspect Ratio")
        self.aspect_ratio_action.triggered.connect(self.change_aspect_ratio)
        self.toolbar.addSeparator()

        self.select_frame_action = self.toolbar.addAction(QIcon("icons/select_frame.png"), "Select Frame")
        self.select_frame_action.triggered.connect(self.select_single_frame)
        self.add_separator(self.toolbar, 12, add_line=False)
        self.add_separator(self.toolbar, 12, line_before=True)

        self.undo_action = self.toolbar.addAction(QIcon("icons/undo.png"), "Undo")
        self.undo_action.triggered.connect(self.undo)

        self.redo_action = self.toolbar.addAction(QIcon("icons/redo.png"), "Redo")
        self.redo_action.triggered.connect(self.redo)

        self.add_separator(self.toolbar, 12, line_before=False)
        self.save_action = self.toolbar.addAction(QIcon("icons/save.png"), "Save")
        self.save_action.triggered.connect(self.save_video)

        self.finish_action = self.toolbar.addAction(QIcon("icons/finish.png"), "Finish")
        self.finish_action.triggered.connect(self.finish_editing)

        # Main layout
        main_layout = QVBoxLayout()

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label)

        # Trimming controls
        trim_layout = QHBoxLayout()
        trim_layout.addWidget(QLabel("Start:"))
        self.start_trim_slider = QSlider(Qt.Horizontal)
        self.start_trim_slider.valueChanged.connect(self.update_trim_preview)
        trim_layout.addWidget(self.start_trim_slider)
        trim_layout.addWidget(QLabel("End:"))
        self.end_trim_slider = QSlider(Qt.Horizontal)
        self.end_trim_slider.valueChanged.connect(self.update_trim_preview)
        trim_layout.addWidget(self.end_trim_slider)
        main_layout.addLayout(trim_layout)

        # Mode selection and controls
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Manual", "Auto"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)

        self.draw_box_btn = QPushButton(QIcon("icons/draw_box.png"), "Draw Box")
        self.draw_box_btn.clicked.connect(self.draw_box)
        mode_layout.addWidget(self.draw_box_btn)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter prompt")
        mode_layout.addWidget(self.prompt_input)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["DINO", "YOLO-World"])
        mode_layout.addWidget(self.model_combo)

        self.detect_btn = QPushButton(QIcon("icons/detect.png"), "Detect Objects")
        self.detect_btn.clicked.connect(self.detect_objects)
        mode_layout.addWidget(self.detect_btn)

        main_layout.addLayout(mode_layout)

        # Frame controls
        frame_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_change)
        frame_layout.addWidget(self.frame_slider)

        self.frame_counter = QLabel("Frame: 0 / 0")
        frame_layout.addWidget(self.frame_counter)

        self.delete_frame_btn = QPushButton(QIcon("icons/delete.png"), "Delete Frame")
        self.delete_frame_btn.clicked.connect(self.delete_frame)
        frame_layout.addWidget(self.delete_frame_btn)

        main_layout.addLayout(frame_layout)

        # Set main layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Initial state
        self.disable_video_controls()
        self.on_mode_changed("Manual")

        # Set tooltips for better user guidance
        self.upload_action.setToolTip("Upload a new video")
        self.trim_action.setToolTip("Trim the current video")
        self.crop_action.setToolTip("Crop the current video")
        self.undo_action.setToolTip("Undo last action")
        self.redo_action.setToolTip("Redo last undone action")
        self.save_action.setToolTip("Save the edited video")
        self.finish_action.setToolTip("Finish editing and close")
        self.draw_box_btn.setToolTip("Draw a box around the object to track")
        self.detect_btn.setToolTip("Detect objects based on the prompt")
        self.delete_frame_btn.setToolTip("Delete the current frame")

        # Adjust sizes for better layout
        self.prompt_input.setMinimumWidth(150)
        self.model_combo.setMinimumWidth(100)
        self.frame_slider.setMinimumWidth(300)

    def add_separator(self, toolbar, width, line_before=True, add_line=True):
        separator_widget = QWidget()
        separator_widget.setFixedSize(width, 10)  # Width and height of the separator widget

        line = None
        if add_line:
            line = QFrame()
            line.setFrameShape(QFrame.VLine)
            line.setFrameShadow(QFrame.Sunken)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        if line_before and line:
            layout.addWidget(line)
            layout.addWidget(separator_widget)
        elif line:
            layout.addWidget(separator_widget)
            layout.addWidget(line)
        else:
            layout.addWidget(separator_widget)

        container = QWidget()
        container.setLayout(layout)
        toolbar.addWidget(container)

    def set_video(self, video_path):
        try:
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
            self.draw_box_btn.setEnabled(True)
            self.detect_btn.setEnabled(True)
            self.mode_combo.setEnabled(True)
            self.undo_action.setEnabled(True)  # Changed from undo_btn to undo_action
            self.redo_action.setEnabled(True)  # Changed from redo_btn to redo_action
            self.save_action.setEnabled(True)  # Changed from save_btn to save_action
            self.finish_action.setEnabled(True)  # Changed from finish_btn to finish_action
        except Exception as e:
            logging.error(f"Error in set_video: {str(e)}")
            logging.error(traceback.format_exc())
            self.show_error_message("Error", f"An error occurred: {str(e)}")

    def upload_video(self):
        try:
            self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video")
            if self.video_path:
                self.set_video(self.video_path)
                self.enable_video_controls()
                self.draw_box_btn.setEnabled(True)
                self.detect_btn.setEnabled(True)
                self.mode_combo.setEnabled(True)
                self.undo_action.setEnabled(True)  # Changed from undo_btn to undo_action
                self.redo_action.setEnabled(True)  # Changed from redo_btn to redo_action
                self.save_action.setEnabled(True)  # Changed from save_btn to save_action
                self.finish_action.setEnabled(True)  # Changed from finish_btn to finish_action
        except Exception as e:
            logging.error(f"Error in upload_video: {str(e)}")
            logging.error(traceback.format_exc())
            self.show_error_message("Error", f"An error occurred: {str(e)}")

    def update_frame(self):
        try:
            if self.processed_frames and 0 <= self.current_frame < len(self.processed_frames):
                frame = self.processed_frames[self.current_frame]
            elif self.frames and 0 <= self.current_frame < len(self.frames):
                frame = self.frames[self.current_frame]
            else:
                return
            self.display_frame(frame)
            self.frame_slider.setValue(self.current_frame)
            total_frames = len(self.processed_frames) if self.processed_frames else len(self.frames)
            self.frame_counter.setText(f"Frame: {self.current_frame + 1} / {total_frames}")
        except Exception as e:
            logging.error(f"Error in update_frame: {str(e)}")
            logging.error(traceback.format_exc())
            self.show_error_message("Error", f"An error occurred: {str(e)}")

    def display_frame(self, frame):
        try:
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.video_label.setPixmap(
                    pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            logging.error(f"Error in display_frame: {str(e)}")
            logging.error(traceback.format_exc())
            self.show_error_message("Error", f"An error occurred: {str(e)}")

    def trim_video(self):
        try:
            if not self.frames:
                QMessageBox.warning(self, "No Video", "Please upload a video first.")
                return

            start = self.start_trim_slider.value()
            end = self.end_trim_slider.value()

            if start >= end:
                QMessageBox.warning(self, "Invalid Trim", "Start frame must be before end frame.")
                return

            # Add undo state before modifying
            self.add_undo_state()

            self.trimmed_frames = self.frames[start:end + 1]
            if not self.trimmed_frames:
                QMessageBox.warning(self, "Trim Error", "Trimming resulted in no frames. Please adjust your selection.")
                return

            self.frame_slider.setRange(0, len(self.trimmed_frames) - 1)
            self.current_frame = 0
            self.frames = self.trimmed_frames  # Update the main frames list
            self.update_frame()
            QMessageBox.information(self, "Trim Complete", f"Video trimmed from frame {start} to {end}")
        except Exception as e:
            QMessageBox.critical(self, "Trim Error", f"An error occurred while trimming: {str(e)}")

    def draw_box(self):
        if self.trimmed_frames or self.frames:
            frame = (self.trimmed_frames or self.frames)[self.current_frame].copy()
            self.drawing_window = DrawingWindow(frame)
            self.drawing_window.boxDrawn.connect(self.process_drawn_box)
            self.drawing_window.show()
        else:
            QMessageBox.warning(self, "No Video", "Please upload a video first.")

    def process_drawn_box(self, bbox):
        if not bbox:
            logging.warning("No box drawn. Please try again.")
            QMessageBox.warning(self, "Warning", "No box drawn. Please try again.")
            return

        self.show_loading_indicator()
        self.process_box_worker = ProcessBoxWorker(self.frames, bbox)
        self.process_box_worker.finished.connect(self.on_process_box_finished)
        self.process_box_worker.error.connect(self.on_process_box_error)
        self.process_box_worker.start()

    def on_process_box_finished(self, formated_crops):
        self.hide_loading_indicator()
        self.add_undo_state()
        self.processed_frames = formated_crops
        self.current_frame = 0
        self.frame_slider.setRange(0, len(self.processed_frames) - 1)
        self.frame_slider.setValue(0)
        self.frames = self.processed_frames
        self.update_frame()
        QMessageBox.information(self, "Processing Complete", "Box processing completed successfully.")

    def on_process_box_error(self, error_msg):
        self.hide_loading_indicator()
        logging.error(f"An error occurred: {error_msg}")
        QMessageBox.critical(self, "Error", f"An error occurred: {error_msg}")

    def update_progress(self, value):
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.set_progress(value)

    def detect_objects(self):
        if self.trimmed_frames or self.frames:
            prompt = self.prompt_input.text()
            detector = self.model_combo.currentText()
            self.show_loading_indicator()
            self.detect_worker = DetectObjectsWorker(self.trimmed_frames or self.frames, prompt, detector)
            self.detect_worker.finished.connect(self.on_detect_objects_finished)
            self.detect_worker.error.connect(self.on_detect_objects_error)
            self.detect_worker.start()
        else:
            QMessageBox.warning(self, "No Video", "Please upload a video first.")

    def on_detect_objects_finished(self, boxes):
        self.hide_loading_indicator()
        if len(boxes) > 0:
            self.choose_box(boxes)
        else:
            QMessageBox.warning(self, "No Objects Detected", "No objects were detected with the given prompt.")

    def on_detect_objects_error(self, error_msg):
        self.hide_loading_indicator()
        logging.error(f"An error occurred during object detection: {error_msg}")
        QMessageBox.critical(self, "Error", f"An error occurred during object detection: {error_msg}")

    def delete_frame(self):
        frames_to_use = self.processed_frames if self.processed_frames else self.frames

        if not frames_to_use:
            QMessageBox.warning(self, "No Frames", "There are no frames to delete.")
            return

        self.add_undo_state()  # Add this line

        del frames_to_use[self.current_frame]

        if frames_to_use:
            if self.current_frame >= len(frames_to_use):
                self.current_frame = len(frames_to_use) - 1
            self.frame_slider.setRange(0, len(frames_to_use) - 1)
            self.update_frame()
        else:
            self.current_frame = 0
            self.frame_slider.setRange(0, 0)
            self.video_label.clear()
            self.frame_counter.setText("Frame: 0 / 0")

        if self.processed_frames:
            self.processed_frames = frames_to_use
        else:
            self.frames = frames_to_use
            self.trimmed_frames = frames_to_use  # Update trimmed_frames as well

    def crop_video(self):
        if not self.frames:
            QMessageBox.warning(self, "No Video", "Please upload a video first.")
            return

        frame = self.frames[self.current_frame].copy()
        crop_dialog = CropDialog(frame, self)
        if crop_dialog.exec_() == QDialog.Accepted:
            crop_rect = crop_dialog.get_crop_rect()
            if crop_rect:
                self.add_undo_state()  # Add this line
                self.apply_crop(crop_rect)
            else:
                QMessageBox.warning(self, "No Crop", "No area was selected for cropping.")

    def apply_crop(self, crop_rect):
        if crop_rect is None:
            return

        nx, ny, nw, nh = crop_rect
        self.show_loading_indicator()
        try:
            cropped_frames = []
            for frame in self.frames:
                h, w = frame.shape[:2]
                x = int(nx * w)
                y = int(ny * h)
                width = int(nw * w)
                height = int(nh * h)

                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                width = min(width, w - x)
                height = min(height, h - y)

                cropped_frame = frame[y:y + height, x:x + width]

                if cropped_frame.size == 0:
                    raise ValueError("Cropping resulted in an empty frame")

                cropped_frames.append(cropped_frame)

            if not cropped_frames:
                raise ValueError("No frames were cropped")

            self.frames = cropped_frames
            self.trimmed_frames = cropped_frames  # Update trimmed_frames as well
            self.processed_frames = []  # Clear processed frames
            self.current_frame = 0
            self.frame_slider.setRange(0, len(self.frames) - 1)
            self.start_trim_slider.setRange(0, len(self.frames) - 1)
            self.end_trim_slider.setRange(0, len(self.frames) - 1)
            self.end_trim_slider.setValue(len(self.frames) - 1)
            self.update_frame()
            QMessageBox.information(self, "Crop Complete", "Video has been cropped successfully.")
        except Exception as e:
            logging.error(f"Error during cropping: {str(e)}")
            logging.error(traceback.format_exc())
            self.show_error_message("Error", f"An error occurred: {str(e)}")
        finally:
            self.hide_loading_indicator()

    def add_undo_state(self):
        state = {
            'frames': [frame.copy() for frame in self.frames],
            'trimmed_frames': [frame.copy() for frame in self.trimmed_frames],
            'processed_frames': [frame.copy() for frame in self.processed_frames],
            'current_frame': self.current_frame,
            'start_trim': self.start_trim_slider.value(),
            'end_trim': self.end_trim_slider.value()
        }
        self.undo_stack.append(state)
        self.redo_stack.clear()  # Clear redo stack when a new action is performed

    def undo(self):
        if not self.undo_stack:
            QMessageBox.information(self, "Undo", "Nothing to undo.")
            return

        try:
            # Save current state to redo stack
            current_state = {
                'frames': [frame.copy() for frame in self.frames],
                'trimmed_frames': [frame.copy() for frame in self.trimmed_frames],
                'processed_frames': [frame.copy() for frame in self.processed_frames],
                'current_frame': self.current_frame,
                'start_trim': self.start_trim_slider.value(),
                'end_trim': self.end_trim_slider.value()
            }
            self.redo_stack.append(current_state)

            # Restore previous state
            state = self.undo_stack.pop()
            self.frames = state['frames']
            self.trimmed_frames = state['trimmed_frames']
            self.processed_frames = state['processed_frames']
            self.current_frame = state['current_frame']

            # Update UI
            self.start_trim_slider.setRange(0, len(self.frames) - 1)
            self.end_trim_slider.setRange(0, len(self.frames) - 1)
            self.start_trim_slider.setValue(state['start_trim'])
            self.end_trim_slider.setValue(state['end_trim'])
            self.frame_slider.setRange(0, len(self.frames) - 1)
            self.frame_slider.setValue(self.current_frame)

            self.update_frame()
            QMessageBox.information(self, "Undo", "Action undone successfully.")
        except Exception as e:
            logging.error(f"Error in undo: {str(e)}")
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Undo Error", f"An error occurred while undoing: {str(e)}")

    def redo(self):
        if not self.redo_stack:
            QMessageBox.information(self, "Redo", "Nothing to redo.")
            return

        try:
            # Save current state to undo stack
            current_state = {
                'frames': [frame.copy() for frame in self.frames],
                'trimmed_frames': [frame.copy() for frame in self.trimmed_frames],
                'processed_frames': [frame.copy() for frame in self.processed_frames],
                'current_frame': self.current_frame,
                'start_trim': self.start_trim_slider.value(),
                'end_trim': self.end_trim_slider.value()
            }
            self.undo_stack.append(current_state)

            # Restore next state
            state = self.redo_stack.pop()
            self.frames = state['frames']
            self.trimmed_frames = state['trimmed_frames']
            self.processed_frames = state['processed_frames']
            self.current_frame = state['current_frame']

            # Update UI
            self.start_trim_slider.setRange(0, len(self.frames) - 1)
            self.end_trim_slider.setRange(0, len(self.frames) - 1)
            self.start_trim_slider.setValue(state['start_trim'])
            self.end_trim_slider.setValue(state['end_trim'])
            self.frame_slider.setRange(0, len(self.frames) - 1)
            self.frame_slider.setValue(self.current_frame)

            self.update_frame()
            QMessageBox.information(self, "Redo", "Action redone successfully.")
        except Exception as e:
            logging.error(f"Error in redo: {str(e)}")
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Redo Error", f"An error occurred while redoing: {str(e)}")

    def save_video(self):
        try:
            if not self.processed_frames and not self.frames:
                QMessageBox.warning(self, "No Video", "There is no video to save.")
                return

            output_path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "MP4 files (*.mp4)")
            if not output_path:
                return

            frames_to_save = self.processed_frames if self.processed_frames else self.frames
            self.show_loading_indicator()
            self.save_worker = SaveVideoWorker(frames_to_save, output_path)
            self.save_worker.finished.connect(self.on_save_video_finished)
            self.save_worker.error.connect(self.on_save_video_error)
            self.save_worker.start()
        except Exception as e:
            self.hide_loading_indicator()
            logging.error(f"Error in save_video: {str(e)}")
            logging.error(traceback.format_exc())
            self.show_error_message("Error", f"An error occurred while saving the video: {str(e)}")

    def on_save_video_finished(self):
        self.hide_loading_indicator()
        QMessageBox.information(self, "Save Complete", "Video has been saved successfully.")

    def on_save_video_error(self, error_msg):
        self.hide_loading_indicator()
        logging.error(f"Error in save_video: {error_msg}")
        self.show_error_message("Error", f"An error occurred while saving the video: {error_msg}")

    def disable_video_controls(self):
        controls = [self.start_trim_slider, self.end_trim_slider, self.trim_action,
                    self.draw_box_btn, self.detect_btn, self.frame_slider,
                    self.delete_frame_btn, self.save_action, self.mode_combo,
                    self.prompt_input, self.model_combo, self.crop_action,
                    self.undo_action, self.redo_action, self.rotate_action,
                    self.scale_action, self.select_frame_action]  # Add select_frame_action to the list
        for control in controls:
            control.setEnabled(False)

    def enable_video_controls(self):
        controls = [self.start_trim_slider, self.end_trim_slider, self.trim_action,
                    self.frame_slider, self.delete_frame_btn, self.save_action,
                    self.mode_combo, self.prompt_input, self.model_combo,
                    self.crop_action, self.undo_action, self.redo_action,
                    self.rotate_action, self.scale_action,
                    self.select_frame_action]  # Add select_frame_action to the list
        for control in controls:
            control.setEnabled(True)

    def on_mode_changed(self, mode):
        if mode == "Manual":
            self.draw_box_btn.setVisible(True)
            self.prompt_input.setVisible(False)
            self.model_combo.setVisible(False)
            self.detect_btn.setVisible(False)
        else:
            self.draw_box_btn.setVisible(False)
            self.prompt_input.setVisible(True)
            self.model_combo.setVisible(True)
            self.detect_btn.setVisible(True)

    def update_trim_preview(self):
        start = self.start_trim_slider.value()
        end = self.end_trim_slider.value()
        if start <= end and start < len(self.frames):
            if self.sender() == self.start_trim_slider:
                self.current_frame = start
            else:
                self.current_frame = end
            self.update_frame()

    def finish_editing(self):
        try:
            logging.info("Finish editing button pressed")
            edited_video = self.get_edited_video()
            logging.info(f"Edited video frames: {len(edited_video)}")
            self.editing_finished.emit(edited_video)
            logging.info("Emitted editing_finished signal")
            self.close()
        except Exception as e:
            logging.error(f"Error in finish_editing: {str(e)}")
            logging.error(traceback.format_exc())
            self.show_error_message("Error", f"An error occurred: {str(e)}")

    def get_edited_video(self):
        if self.processed_frames:
            return self.processed_frames
        elif self.trimmed_frames:
            return self.trimmed_frames
        else:
            return self.frames

    def show_error_message(self, title, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def on_frame_slider_change(self):
        self.current_frame = self.frame_slider.value()
        self.update_frame()

    def choose_box(self, boxes):
        frame = self.trimmed_frames[0] if self.trimmed_frames else self.frames[0]
        frame_with_boxes = self.draw_boxes_on_frame(frame, boxes)

        rgb_frame = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        dialog = BoxSelectionDialog(self, q_img)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_box is not None:
            chosen_box = boxes[dialog.selected_box]
            self.process_auto_box(chosen_box)
        else:
            QMessageBox.information(self, "Cancelled", "Box selection cancelled.")

        self.update_frame()

    def draw_boxes_on_frame(self, frame, boxes):
        frame_with_boxes = frame.copy()
        height, width = frame.shape[:2]
        for i, box in enumerate(boxes):
            if len(box) >= 4:
                cx, cy, w, h = box[:4]
                x1 = int((cx - w / 2) * width)
                y1 = int((cy - h / 2) * height)
                x2 = int((cx + w / 2) * width)
                y2 = int((cy + h / 2) * height)
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_with_boxes, f'{i + 1}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame_with_boxes

    def process_auto_box(self, bbox):
        self.show_loading_indicator()
        try:
            frame = self.trimmed_frames[0] if self.trimmed_frames else self.frames[0]
            frames = self.trimmed_frames or self.frames
            height, width = frame.shape[:2]

            cx, cy, w, h = bbox
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            w = int(w * width)
            h = int(h * height)
            pixel_bbox = (x, y, w, h)

            self.process_box_worker = ProcessBoxWorker(frames, pixel_bbox)
            self.process_box_worker.finished.connect(self.on_auto_process_box_finished)
            self.process_box_worker.error.connect(self.on_process_box_error)
            self.process_box_worker.start()

        except Exception as e:
            self.hide_loading_indicator()
            logging.error(f"An error occurred: {str(e)}")
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def on_auto_process_box_finished(self, formated_crops):
        self.hide_loading_indicator()
        self.add_undo_state()
        self.processed_frames = formated_crops
        self.current_frame = 0
        self.frame_slider.setRange(0, len(self.processed_frames) - 1)
        self.frame_slider.setValue(0)
        self.frames = self.processed_frames
        self.update_frame()
        logging.debug(f"Processed {len(self.processed_frames)} frames")
        QMessageBox.information(self, "Processing Complete", "Auto box processing completed successfully.")

    def rotate_video(self):
        if not self.frames:
            QMessageBox.warning(self, "No Video", "Please upload a video first.")
            return

        frame = self.frames[self.current_frame]
        dialog = RotationDialog(self, frame)
        result = dialog.exec_()

        if result > 0:
            angle = dialog.angle
            self.add_undo_state()

            if result == 2:  # Apply to all frames
                self.rotate_all_frames(angle)
            else:  # Apply to current frame
                self.rotate_single_frame(angle)

            self.update_frame()
            QMessageBox.information(self, "Rotation Complete", f"Video rotated by {angle} degrees.")

    def rotate_all_frames(self, angle):
        rotated_frames = []
        for frame in self.frames:
            rotated_frame = self.rotate_frame(frame, angle)
            rotated_frames.append(rotated_frame)
        self.frames = rotated_frames
        if self.processed_frames:
            self.processed_frames = rotated_frames
        if self.trimmed_frames:
            self.trimmed_frames = rotated_frames

    def rotate_single_frame(self, angle):
        if 0 <= self.current_frame < len(self.frames):
            rotated_frame = self.rotate_frame(self.frames[self.current_frame], angle)
            self.frames[self.current_frame] = rotated_frame
            if self.processed_frames:
                self.processed_frames[self.current_frame] = rotated_frame
            if self.trimmed_frames:
                self.trimmed_frames[self.current_frame] = rotated_frame

    def rotate_frame(self, frame, angle):
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        return rotated_frame

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

    def scale_video(self):
        if not self.frames:
            QMessageBox.warning(self, "No Video", "Please upload a video first.")
            return

        frame = self.frames[self.current_frame]
        dialog = ScalingDialog(self, frame)
        result = dialog.exec_()

        if result > 0:
            scale_factor = dialog.scale_factor / 100
            self.add_undo_state()

            if result == 2:  # Apply to all frames
                self.scale_all_frames(scale_factor)
            else:  # Apply to current frame
                self.scale_single_frame(scale_factor)

            self.update_frame()
            QMessageBox.information(self, "Scaling Complete", f"Video scaled to {dialog.scale_factor}%.")

    def scale_all_frames(self, scale_factor):
        scaled_frames = []
        for frame in self.frames:
            scaled_frame = self.scale_frame(frame, scale_factor)
            scaled_frames.append(scaled_frame)
        self.frames = scaled_frames
        if self.processed_frames:
            self.processed_frames = scaled_frames
        if self.trimmed_frames:
            self.trimmed_frames = scaled_frames

    def scale_single_frame(self, scale_factor):
        if 0 <= self.current_frame < len(self.frames):
            scaled_frame = self.scale_frame(self.frames[self.current_frame], scale_factor)
            self.frames[self.current_frame] = scaled_frame
            if self.processed_frames:
                self.processed_frames[self.current_frame] = scaled_frame
            if self.trimmed_frames:
                self.trimmed_frames[self.current_frame] = scaled_frame

    def scale_frame(self, frame, scale_factor):
        try:
            height, width = frame.shape[:2]
            new_height = max(1, int(height * scale_factor))
            new_width = max(1, int(width * scale_factor))

            if scale_factor >= 1:  # Enlarging
                scaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                y_offset = max(0, (new_height - height) // 2)
                x_offset = max(0, (new_width - width) // 2)
                result_frame = scaled_frame[y_offset:y_offset + height, x_offset:x_offset + width]
            else:  # Shrinking
                scaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                result_frame = np.zeros((height, width, 3), dtype=np.uint8)
                y_offset = (height - new_height) // 2
                x_offset = (width - new_width) // 2
                result_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = scaled_frame

            return result_frame
        except Exception as e:
            print(f"Error in scale_frame: {str(e)}")
            return frame.copy()  # Return a copy of the original frame if scaling fails

    def select_single_frame(self):
        if not self.frames:
            QMessageBox.warning(self, "No Video", "Please upload a video first.")
            return

        current_frame = self.current_frame
        frame_count = len(self.frames)

        selected_frame, ok = QInputDialog.getInt(self, "Select Frame",
                                                 f"Enter frame number (1-{frame_count}):",
                                                 current_frame + 1, 1, frame_count)
        if ok:
            self.add_undo_state()
            selected_frame -= 1  # Convert to 0-based index
            self.frames = [self.frames[selected_frame]]
            self.processed_frames = [self.processed_frames[selected_frame]] if self.processed_frames else []
            self.trimmed_frames = [self.trimmed_frames[selected_frame]] if self.trimmed_frames else []
            self.current_frame = 0
            self.frame_slider.setRange(0, 0)
            self.start_trim_slider.setRange(0, 0)
            self.end_trim_slider.setRange(0, 0)
            self.update_frame()
            QMessageBox.information(self, "Frame Selected", f"Video reduced to frame {selected_frame + 1}.")

    def apply_aspect_ratio(self, frame, new_ratio):
        height, width = frame.shape[:2]
        current_ratio = width / height
        target_ratio = new_ratio[0] / new_ratio[1]

        if abs(current_ratio - target_ratio) < 0.01:  # If ratios are very close, return original frame
            return frame

        if current_ratio > target_ratio:
            # Current frame is wider, need to crop width
            new_width = int(height * target_ratio)
            start = (width - new_width) // 2
            modified_frame = frame[:, start:start + new_width]
        else:
            # Current frame is taller, need to crop height
            new_height = int(width / target_ratio)
            start = (height - new_height) // 2
            modified_frame = frame[start:start + new_height, :]

        # Ensure the modified frame is not empty
        if modified_frame.size == 0:
            print("Warning: Modified frame is empty. Returning original frame.")
            return frame

        return modified_frame

    def change_aspect_ratio(self):
        if not self.frames:
            QMessageBox.warning(self, "No Video", "Please upload a video first.")
            return

        frame = self.frames[self.current_frame]
        dialog = AspectRatioDialog(self, frame)
        result = dialog.exec_()

        if result > 0:
            new_ratio = dialog.get_aspect_ratio()
            self.add_undo_state()

            try:
                if result == 2:  # Apply to all frames
                    self.apply_aspect_ratio_all_frames(new_ratio)
                else:  # Apply to current frame
                    self.apply_aspect_ratio_single_frame(new_ratio)

                self.update_frame()
                QMessageBox.information(self, "Aspect Ratio Changed",
                                        f"Video aspect ratio changed to {new_ratio[0]}:{new_ratio[1]}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while changing aspect ratio: {str(e)}")

    def apply_aspect_ratio_all_frames(self, new_ratio):
        modified_frames = []
        for frame in self.frames:
            modified_frame = self.apply_aspect_ratio(frame, new_ratio)
            modified_frames.append(modified_frame)
        self.frames = modified_frames
        if self.processed_frames:
            self.processed_frames = modified_frames
        if self.trimmed_frames:
            self.trimmed_frames = modified_frames

    def apply_aspect_ratio_single_frame(self, new_ratio):
        if 0 <= self.current_frame < len(self.frames):
            modified_frame = self.apply_aspect_ratio(self.frames[self.current_frame], new_ratio)
            self.frames[self.current_frame] = modified_frame
            if self.processed_frames:
                self.processed_frames[self.current_frame] = modified_frame
            if self.trimmed_frames:
                self.trimmed_frames[self.current_frame] = modified_frame


class DrawingWindow(QWidget):
    boxDrawn = pyqtSignal(tuple)

    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        self.setWindowTitle("Draw Box")
        self.setGeometry(100, 100, frame.shape[1], frame.shape[0])
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
            }
        """)
        self.begin = QPoint()
        self.end = QPoint()
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0],
                                              QImage.Format_RGB888).rgbSwapped())

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        self.rubberBand.setGeometry(QRect(self.begin, self.end))
        self.rubberBand.show()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.rubberBand.setGeometry(QRect(self.begin, self.end).normalized())

    def mouseReleaseEvent(self, event):
        self.rubberBand.hide()
        bbox = (min(self.begin.x(), self.end.x()),
                min(self.begin.y(), self.end.y()),
                abs(self.begin.x() - self.end.x()),
                abs(self.begin.y() - self.end.y()))
        self.boxDrawn.emit(bbox)
        self.close()


class BoxSelectionDialog(QDialog):
    def __init__(self, parent, frame_with_boxes):
        super().__init__(parent)
        self.setWindowTitle("Select Box")
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0;
            }
            QLabel {
                    font-size: 14px;
                    color: #333333;
            }
            QPushButton {
                    background-color: #333333;  /* Dark gray, almost black */
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
                    background-color: #555555;  /* Lighter gray for hover effect */
            }
        """)
        layout = QVBoxLayout(self)

        self.label = QLabel()
        self.pixmap = QPixmap.fromImage(frame_with_boxes)
        self.label.setPixmap(self.pixmap)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.label)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        select_button = QPushButton("Select Box Number")
        select_button.clicked.connect(self.get_box_number)
        layout.addWidget(select_button)

        self.selected_box = None
        self.adjust_size()

    def adjust_size(self):
        screen = QApplication.primaryScreen().geometry()
        max_width = int(screen.width() * 0.8)
        max_height = int(screen.height() * 0.8)

        aspect_ratio = self.pixmap.width() / self.pixmap.height()

        if self.pixmap.width() > max_width or self.pixmap.height() > max_height:
            if aspect_ratio > 1:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
        else:
            new_width = self.pixmap.width()
            new_height = self.pixmap.height()

        new_width += 40
        new_height += 80

        self.setFixedSize(new_width, new_height)

    def get_box_number(self):
        number, ok = QInputDialog.getInt(self, "Choose Box", "Enter the number of the box you want to track:", 1, 1,
                                         100)
        if ok:
            self.selected_box = number - 1
            self.accept()


class CropDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.crop_rect = None
        self.start_point = None
        self.end_point = None

        self.setWindowTitle("Crop Image")
        self.setGeometry(100, 100, image.shape[1], image.shape[0])

        layout = QVBoxLayout()
        self.confirm_button = QPushButton("Confirm Crop")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)
        self.setLayout(layout)

    def paintEvent(self, event):
        painter = QPainter(self)
        scaled_image = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0],
                              QImage.Format_RGB888).rgbSwapped()
        scaled_image = scaled_image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawImage(self.rect(), scaled_image)

        if self.start_point and self.end_point:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.SolidLine))
            painter.drawRect(QRect(self.start_point, self.end_point))

    def mousePressEvent(self, event):
        self.start_point = event.pos()
        self.end_point = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        self.end_point = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.end_point = event.pos()
        self.crop_rect = (min(self.start_point.x(), self.end_point.x()),
                          min(self.start_point.y(), self.end_point.y()),
                          abs(self.start_point.x() - self.end_point.x()),
                          abs(self.start_point.y() - self.end_point.y()))
        self.update()

    def get_crop_rect(self):
        if self.crop_rect:
            x, y, w, h = self.crop_rect
            return (x / self.width(), y / self.height(), w / self.width(), h / self.height())
        return None


class RotationDialog(QDialog):
    def __init__(self, parent, frame):
        super().__init__(parent)
        self.setWindowTitle("Rotate Video")
        self.frame = frame
        self.angle = 0

        layout = QVBoxLayout(self)

        # Rotation preview
        self.preview_label = QLabel()
        layout.addWidget(self.preview_label)

        # Rotation slider
        slider_layout = QHBoxLayout()
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self.update_preview)
        slider_layout.addWidget(QLabel("Angle:"))
        slider_layout.addWidget(self.rotation_slider)
        self.angle_label = QLabel("0°")
        slider_layout.addWidget(self.angle_label)
        layout.addLayout(slider_layout)

        # Buttons
        button_layout = QHBoxLayout()
        apply_current_btn = QPushButton("Apply to Current Frame")
        apply_current_btn.clicked.connect(lambda: self.done(1))
        apply_all_btn = QPushButton("Apply to All Frames")
        apply_all_btn.clicked.connect(lambda: self.done(2))
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(apply_current_btn)
        button_layout.addWidget(apply_all_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        self.update_preview()

    def update_preview(self):
        self.angle = self.rotation_slider.value()
        self.angle_label.setText(f"{self.angle}°")
        rotated_frame = self.parent().rotate_frame(self.frame, self.angle)
        height, width = rotated_frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(rotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.preview_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class ScalingDialog(QDialog):
    def __init__(self, parent, frame):
        super().__init__(parent)
        self.setWindowTitle("Scale Video")
        self.frame = frame
        self.scale_factor = 100

        layout = QVBoxLayout(self)

        # Scaling preview
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(400, 300)
        layout.addWidget(self.preview_label)

        # Scaling slider
        slider_layout = QHBoxLayout()
        self.scaling_slider = QSlider(Qt.Horizontal)
        self.scaling_slider.setRange(10, 300)  # Allow scaling from 10% to 300%
        self.scaling_slider.setValue(100)
        self.scaling_slider.valueChanged.connect(self.update_preview)
        slider_layout.addWidget(QLabel("Scale:"))
        slider_layout.addWidget(self.scaling_slider)
        self.scale_label = QLabel("100%")
        slider_layout.addWidget(self.scale_label)
        layout.addLayout(slider_layout)

        # Buttons
        button_layout = QHBoxLayout()
        apply_current_btn = QPushButton("Apply to Current Frame")
        apply_current_btn.clicked.connect(lambda: self.done(1))
        apply_all_btn = QPushButton("Apply to All Frames")
        apply_all_btn.clicked.connect(lambda: self.done(2))
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(apply_current_btn)
        button_layout.addWidget(apply_all_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        self.update_preview()

    def update_preview(self):
        try:
            self.scale_factor = self.scaling_slider.value()
            self.scale_label.setText(f"{self.scale_factor}%")

            frame_copy = self.frame.copy()
            scaled_frame = self.parent().scale_frame(frame_copy, self.scale_factor / 100)

            if scaled_frame is not None and scaled_frame.size > 0:
                height, width = scaled_frame.shape[:2]
                bytes_per_line = 3 * width

                # Convert the numpy array to bytes
                frame_bytes = scaled_frame.tobytes()

                q_image = QImage(frame_bytes, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)
                self.preview_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                print("Error: Scaled frame is None or empty")
        except Exception as e:
            print(f"Error in update_preview: {str(e)}")


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)

        self.label = QLabel(self)
        self.movie = QMovie("icons/loading.gif")
        self.label.setMovie(self.movie)
        self.movie.setScaledSize(QSize(50, 50))
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


class AnimationThread(QThread):
    def __init__(self, movie):
        super().__init__()
        self.movie = movie
        self.is_running = True

    def run(self):
        self.movie.start()
        while self.is_running:
            self.msleep(100)  # Sleep for a short time to prevent high CPU usage

    def stop(self):
        self.is_running = False
        self.wait()
        self.movie.stop()


class ProcessBoxWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, frames, bbox):
        super().__init__()
        self.frames = frames
        self.bbox = bbox

    def run(self):
        try:
            logging.debug(f"Processing box: {self.bbox}")
            logging.debug("Running tracker...")

            crops = Tracker.align_crops_from_BB(frames=self.frames, initial_BB_XYWH=self.bbox)
            formated_crops = frames_to_constant_format(crops)

            logging.debug("Finish tracker...")
            self.finished.emit(formated_crops)
        except Exception as e:
            logging.error(f"Error in ProcessBoxWorker: {str(e)}")
            logging.error(traceback.format_exc())
            self.error.emit(str(e))


class DetectObjectsWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, frames, prompt, detector):
        super().__init__()
        self.frames = frames
        self.prompt = prompt
        self.detector = detector

    def run(self):
        try:
            frame = self.frames[0]
            if self.detector == "YOLO-World":
                boxes = get_bounding_boxes(frame, PARAMETER.yolo_world_checkpoint, [self.prompt])
            else:  # DINO
                model = load_model(PARAMETER.grounding_dino_config_SwinT_OGC, PARAMETER.grounding_dino_checkpoint)
                model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
                boxes, logits, phrases = detect_objects_dino(model, frame, self.prompt)
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().tolist()
            self.finished.emit(boxes)
        except Exception as e:
            self.error.emit(str(e))


class SaveVideoWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, frames, output_path):
        super().__init__()
        self.frames = frames
        self.output_path = output_path

    def run(self):
        try:
            height, width = self.frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, 30.0, (width, height))

            for frame in self.frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)

            out.release()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class AspectRatioDialog(QDialog):
    def __init__(self, parent, frame):
        super().__init__(parent)
        self.setWindowTitle("Change Aspect Ratio")
        self.frame = frame
        self.aspect_ratio = (16, 9)  # Default aspect ratio

        layout = QVBoxLayout(self)

        # Aspect ratio preview
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(400, 300)
        layout.addWidget(self.preview_label)

        # Aspect ratio selection
        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("Aspect Ratio:"))
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(["16:9", "4:3", "1:1", "21:9", "Custom"])
        self.ratio_combo.currentTextChanged.connect(self.on_ratio_changed)
        ratio_layout.addWidget(self.ratio_combo)
        layout.addLayout(ratio_layout)

        # Custom ratio inputs
        self.custom_layout = QHBoxLayout()
        self.width_input = QLineEdit()
        self.height_input = QLineEdit()
        self.custom_layout.addWidget(QLabel("Width:"))
        self.custom_layout.addWidget(self.width_input)
        self.custom_layout.addWidget(QLabel("Height:"))
        self.custom_layout.addWidget(self.height_input)
        self.custom_widget = QWidget()
        self.custom_widget.setLayout(self.custom_layout)
        layout.addWidget(self.custom_widget)
        self.custom_widget.hide()

        # Buttons
        button_layout = QHBoxLayout()
        apply_current_btn = QPushButton("Apply to Current Frame")
        apply_current_btn.clicked.connect(lambda: self.done(1))
        apply_all_btn = QPushButton("Apply to All Frames")
        apply_all_btn.clicked.connect(lambda: self.done(2))
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(apply_current_btn)
        button_layout.addWidget(apply_all_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        self.update_preview()

    def on_ratio_changed(self, text):
        if text == "Custom":
            self.custom_widget.show()
            self.adjustSize()
        else:
            self.custom_widget.hide()
            self.adjustSize()
            w, h = map(int, text.split(":"))
            self.aspect_ratio = (w, h)
        self.update_preview()

    def update_preview(self):
        try:
            modified_frame = self.parent().apply_aspect_ratio(self.frame, self.aspect_ratio)
            height, width = modified_frame.shape[:2]
            bytes_per_line = 3 * width

            # Convert numpy array to bytes
            rgb_image = cv2.cvtColor(modified_frame, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_image)
            self.preview_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            print(f"Error in update_preview: {str(e)}")
            traceback.print_exc()

    def get_aspect_ratio(self):
        if self.ratio_combo.currentText() == "Custom":
            try:
                w = int(self.width_input.text())
                h = int(self.height_input.text())
                return (w, h)
            except ValueError:
                return (16, 9)  # Default to 16:9 if invalid input
        return self.aspect_ratio


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = VideoEditor()
    editor.show()
    sys.exit(app.exec_())
