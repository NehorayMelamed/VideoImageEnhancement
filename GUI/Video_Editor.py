

from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import torch

from Detection.yolo_world import get_bounding_boxes
from Detection.dino import detect_objects_dino
from Segmentation.sam import get_mask_from_bbox
from Tracking.co_tracker.co_tracker import track_points_in_video_auto
import PARAMETER
from PyQt5.QtWidgets import QProgressDialog
from Detection.dino import load_model

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, \
    QLabel, QSlider, QComboBox, QLineEdit, QScrollArea, QDialog, QInputDialog, QRubberBand, QMessageBox
from PyQt5.QtCore import pyqtSignal, QTimer
import traceback
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class VideoEditor(QMainWindow):
    editing_finished = pyqtSignal(object)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Editor")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }
            QSlider {
                height: 15px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #fff;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #4CAF50;
                width: 18px;
                height: 18px;
                border-radius: 9px;
                margin: -5px 0;
            }
            QComboBox {
                padding: 5px;
            }
            QLineEdit {
                padding: 5px;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_change)
        self.layout.addWidget(self.frame_slider)

        self.video_path = None
        self.cap = None
        self.current_frame = 0
        self.frames = []
        self.trimmed_frames = []
        self.processed_frames = []
        self.mode = None
        # QTimer.singleShot(0, self.closeAllWindows)
        self.init_ui()
        self.progress_dialog = None
        self.progress_timer = None


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
        except Exception as e:
            logging.error(f"Error in set_video: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def init_ui(self):
        # Upload button
        self.upload_btn = QPushButton("Upload Video")
        self.upload_btn.clicked.connect(self.upload_video)
        self.layout.addWidget(self.upload_btn)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)  # Set a minimum size for the video display
        self.layout.addWidget(self.video_label)

        # Trimming controls
        self.trim_layout = QHBoxLayout()
        self.start_trim_slider = QSlider(Qt.Horizontal)
        self.end_trim_slider = QSlider(Qt.Horizontal)
        self.start_trim_slider.valueChanged.connect(self.update_trim_preview)
        self.end_trim_slider.valueChanged.connect(self.update_trim_preview)  # Add this line
        self.trim_layout.addWidget(QLabel("Start:"))
        self.trim_layout.addWidget(self.start_trim_slider)
        self.trim_layout.addWidget(QLabel("End:"))
        self.trim_layout.addWidget(self.end_trim_slider)
        self.layout.addLayout(self.trim_layout)

        # Trim button
        self.trim_btn = QPushButton("Trim Video")
        self.trim_btn.clicked.connect(self.trim_video)
        self.layout.addWidget(self.trim_btn)

        # Add this after the trim button
        self.crop_btn = QPushButton("Crop Video")
        self.crop_btn.clicked.connect(self.crop_video)
        self.layout.addWidget(self.crop_btn)

        # Mode selection
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Manual", "Auto"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        self.layout.addWidget(self.mode_combo)

        # Manual mode controls
        self.manual_layout = QHBoxLayout()
        self.draw_box_btn = QPushButton("Draw Box")
        self.draw_box_btn.clicked.connect(self.draw_box)
        self.manual_layout.addWidget(self.draw_box_btn)
        self.layout.addLayout(self.manual_layout)

        # Auto mode controls
        self.auto_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter prompt")
        self.auto_layout.addWidget(self.prompt_input)

        # Model selection dropdown
        self.model_combo = QComboBox()
        self.model_combo.addItems(["DINO", "YOLO-World"])
        self.auto_layout.addWidget(self.model_combo)

        self.detect_btn = QPushButton("Detect Objects")
        self.detect_btn.clicked.connect(self.detect_objects)
        self.auto_layout.addWidget(self.detect_btn)
        self.layout.addLayout(self.auto_layout)

        # Frame navigation
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_change)
        self.layout.addWidget(self.frame_slider)

        # Frame counter display
        self.frame_counter = QLabel("Frame: 0 / 0")
        self.layout.addWidget(self.frame_counter)

        # Delete frame button
        self.delete_frame_btn = QPushButton("Delete Frame")
        self.delete_frame_btn.clicked.connect(self.delete_frame)
        self.layout.addWidget(self.delete_frame_btn)

        # Finish and save button
        self.save_btn = QPushButton("Finish and Save")
        self.save_btn.clicked.connect(self.save_video)
        self.layout.addWidget(self.save_btn)

        # Set up main window
        self.setLayout(self.layout)
        self.setWindowTitle("Video Editor")
        self.setGeometry(100, 100, 800, 600)

        # Initially disable buttons that require a video
        self.disable_video_controls()

        # Initially set the mode to Manual and hide Auto controls
        self.on_mode_changed("Manual")
        self.finish_btn = QPushButton("Finish Editing")
        self.finish_btn.clicked.connect(self.finish_editing)
        self.layout.addWidget(self.finish_btn)

    def on_mode_changed(self, mode):
        if mode == "Manual":
            self.draw_box_btn.setVisible(True)
            self.prompt_input.setVisible(False)
            self.model_combo.setVisible(False)
            self.detect_btn.setVisible(False)
        else:  # Auto mode
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
            else:  # end_trim_slider
                self.current_frame = end
            self.update_frame()

    def closeAllWindows(self):
        cv2.destroyAllWindows()

    def upload_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.frames = []
            self.trimmed_frames = []  # Reset trimmed frames
            self.processed_frames = []  # Reset processed frames
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
            self.update_frame()
            self.enable_video_controls()

    def update_frame(self):
        if self.frames and 0 <= self.current_frame < len(self.frames):
            frame = self.frames[self.current_frame]
            self.display_frame(frame)
            self.frame_slider.setValue(self.current_frame)

    def display_frame(self, frame):
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(
                pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def trim_video(self):
        self.show_progress_dialog("Trimming video...")
        try:
            start = self.start_trim_slider.value()
            end = self.end_trim_slider.value()
            self.trimmed_frames = self.frames[start:end + 1]
            self.frame_slider.setRange(0, len(self.trimmed_frames) - 1)
            self.current_frame = 0
            self.frames = self.trimmed_frames  # Update the main frames list
            self.update_frame()
            QMessageBox.information(self, "Trim Complete", f"Video trimmed from frame {start} to {end}")
        finally:
            self.progress_dialog.close()

    def draw_box(self):
        if self.trimmed_frames:
            frame = self.trimmed_frames[self.current_frame].copy()
            self.drawing_window = DrawingWindow(frame)
            self.drawing_window.boxDrawn.connect(self.process_drawn_box)
            self.drawing_window.show()

    def process_drawn_box(self, bbox):
        self.show_progress_dialog("Processing drawn box...")
        try:
            if bbox:
                try:
                    frame = self.trimmed_frames[0].copy()
                    logging.debug(f"Processing box: {bbox}")

                    logging.debug("Generating mask...")
                    mask = get_mask_from_bbox(frame, PARAMETER.SAM_CHECKPOINTS, bbox=bbox)
                    logging.debug("Mask generated successfully")

                    logging.debug("Generating center point...")
                    points = self.generate_grid_points(mask, bbox)
                    logging.debug(f"Generated center point: {points}")

                    if not points:
                        raise ValueError("No valid points generated for tracking")

                    logging.debug("Running co-tracker...")
                    tracks, visibility, _ = track_points_in_video_auto(self.trimmed_frames, points=points)
                    logging.debug(
                        f"Co-tracker completed. Tracks shape: {tracks.shape}, Visibility shape: {visibility.shape}")

                    logging.debug("Processing frames...")
                    self.process_frames(bbox, tracks, visibility)
                    logging.debug("Frames processed successfully")

                except Exception as e:
                    logging.error(f"An error occurred: {str(e)}")
                    logging.error(traceback.format_exc())
                    QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            else:
                logging.warning("No box drawn. Please try again.")
                QMessageBox.warning(self, "Warning", "No box drawn. Please try again.")
        finally:
            self.progress_dialog.close()

    def generate_grid_points(self, mask, bbox):
        y, x = np.where(mask)
        if len(x) == 0 or len(y) == 0:
            # If the mask is empty, use the center of the bounding box
            center_x = bbox[0] + bbox[2] // 2
            center_y = bbox[1] + bbox[3] // 2
        else:
            center_x, center_y = np.mean(x), np.mean(y)
        return [(center_x, center_y)]

    def generate_grid_points_auto(self, mask, bbox):
        x, y, w, h = bbox
        mask_roi = mask[y:y + h, x:x + w]
        y_indices, x_indices = np.where(mask_roi)
        if len(x_indices) == 0 or len(y_indices) == 0:
            center_x, center_y = x + w // 2, y + h // 2
        else:
            center_x = x + int(np.mean(x_indices))
            center_y = y + int(np.mean(y_indices))
        return [(center_x / mask.shape[1], center_y / mask.shape[0])]  # Return normalized coordinates

    def detect_objects(self):
        if self.trimmed_frames:
            prompt = self.prompt_input.text()
            detector = self.model_combo.currentText()  # Get the selected model
            self.show_progress_dialog("Detecting objects...")
            try:
                if detector == "YOLO-World":
                    boxes = get_bounding_boxes(self.trimmed_frames[0], PARAMETER.YOLOWORLD_CHECKPOINTS, [prompt])
                else:  # DINO
                    model = load_model(PARAMETER.DINO_CONFIG, PARAMETER.DINO_CHECKPOINTS)
                    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
                    boxes, logits, phrases = detect_objects_dino(model, self.trimmed_frames[0], prompt)

                    # Convert boxes to a list of tuples if it's a tensor
                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes.cpu().tolist()

                logging.debug(f"Detected boxes: {boxes}")
                logging.debug(f"Number of boxes: {len(boxes)}")
                logging.debug(f"Box format: {type(boxes[0]) if boxes else 'No boxes'}")

                if len(boxes) > 0:
                    self.choose_box(boxes)
                else:
                    QMessageBox.warning(self, "No Objects Detected", "No objects were detected with the given prompt.")
            except Exception as e:
                logging.error(f"An error occurred during object detection: {str(e)}")
                logging.error(traceback.format_exc())
                QMessageBox.critical(self, "Error", f"An error occurred during object detection: {str(e)}")
            finally:
                self.close_progress_dialog()

    def update_mask(self, initial_mask, track, visible):
        # This is a simplified version. You might need a more sophisticated approach.
        new_mask = np.zeros_like(initial_mask)
        for point, is_visible in zip(track, visible):
            if is_visible:
                cv2.circle(new_mask, tuple(map(int, point)), 5, 255, -1)
        return new_mask

    def delete_frame(self):
        frames_to_use = self.processed_frames if self.processed_frames else self.frames

        if not frames_to_use:
            QMessageBox.warning(self, "No Frames", "There are no frames to delete.")
            return

        del frames_to_use[self.current_frame]

        if frames_to_use:
            if self.current_frame >= len(frames_to_use):
                self.current_frame = len(frames_to_use) - 1
            self.frame_slider.setRange(0, len(frames_to_use) - 1)
            self.update_frame()
        else:
            self.current_frame = 0
            self.frame_slider.setRange(0, 0)
            self.video_label.clear()  # Clear the video display
            self.frame_counter.setText("Frame: 0 / 0")

        # Update the appropriate frame list
        if self.processed_frames:
            self.processed_frames = frames_to_use
        else:
            self.frames = frames_to_use

    def process_frames_auto(self, original_bbox, tracks, visibility):
        self.show_progress_dialog("Processing frames...")
        try:
            self.processed_frames = []
            height, width = self.trimmed_frames[0].shape[:2]
            orig_x, orig_y, orig_w, orig_h = original_bbox

            logging.debug(f"Original bbox (pixels): x={orig_x}, y={orig_y}, w={orig_w}, h={orig_h}")

            for i, frame in enumerate(self.trimmed_frames):
                center_x, center_y = tracks[i][0]
                # The tracks are already in pixel coordinates, no need to convert

                new_x = int(center_x - orig_w / 2)
                new_y = int(center_y - orig_h / 2)

                # Ensure the box doesn't go out of frame
                new_x = max(0, min(new_x, width - orig_w))
                new_y = max(0, min(new_y, height - orig_h))

                logging.debug(f"Frame {i}: Center ({center_x}, {center_y}), New top-left: ({new_x}, {new_y})")

                cropped_frame = frame[new_y:new_y + orig_h, new_x:new_x + orig_w]
                self.processed_frames.append(cropped_frame)

            self.current_frame = 0
            self.frame_slider.setRange(0, len(self.processed_frames) - 1)
            self.frame_slider.setValue(0)
            self.frames = self.processed_frames
            self.update_frame()
            logging.debug(f"Processed {len(self.processed_frames)} frames")
        finally:
            self.close_progress_dialog()
    def choose_box(self, boxes):
        frame = self.trimmed_frames[0].copy()
        height, width = frame.shape[:2]

        # Draw bounding boxes on the frame
        for i, box in enumerate(boxes):
            if len(box) >= 4:
                # Assuming the format is [center_x, center_y, width, height] in normalized coordinates
                cx, cy, w, h = box[:4]
                # Convert to pixel coordinates and top-left, bottom-right format
                x1 = int((cx - w / 2) * width)
                y1 = int((cy - h / 2) * height)
                x2 = int((cx + w / 2) * width)
                y2 = int((cy + h / 2) * height)
                print(f"Box {i + 1}: ({x1}, {y1}) to ({x2}, {y2})")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{i + 1}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the frame to RGB for displaying with PyQt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Show dialog with frame and get user selection
        dialog = BoxSelectionDialog(self, q_img)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_box is not None:
            chosen_box = boxes[dialog.selected_box]
            self.process_auto_box(chosen_box)
        else:
            QMessageBox.information(self, "Cancelled", "Box selection cancelled.")

        # Restore the original frame display
        self.update_frame()

    def process_auto_box(self, bbox):
        self.show_progress_dialog("Processing frames...")
        try:
            frame = self.trimmed_frames[0].copy()
            height, width = frame.shape[:2]
            logging.debug(f"Processing normalized box: {bbox}")

            # Convert normalized bbox to pixel coordinates
            cx, cy, w, h = bbox
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            w = int(w * width)
            h = int(h * height)
            pixel_bbox = (x, y, w, h)
            logging.debug(f"Pixel bbox: {pixel_bbox}")

            logging.debug("Generating mask...")
            mask = get_mask_from_bbox(frame, PARAMETER.SAM_CHECKPOINTS, bbox=pixel_bbox)
            logging.debug("Mask generated successfully")

            logging.debug("Generating center point...")
            points = self.generate_grid_points(mask, pixel_bbox)
            logging.debug(f"Generated center point: {points}")

            if not points:
                raise ValueError("No valid points generated for tracking")

            logging.debug("Running co-tracker...")
            tracks, visibility, frames_tracks = track_points_in_video_auto(self.trimmed_frames, points=points)
            logging.debug(f"Co-tracker completed. Tracks shape: {tracks.shape}, Visibility shape: {visibility.shape}")

            logging.debug("Processing frames...")
            self.process_frames_auto(pixel_bbox, tracks, visibility)
            logging.debug("Frames processed successfully")

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
        finally:
            self.close_progress_dialog()

    def process_frames(self, original_bbox, tracks, visibility):
        self.processed_frames = []
        height, width = self.trimmed_frames[0].shape[:2]

        # Convert bbox to pixel coordinates if it's normalized
        if max(original_bbox) <= 1:
            orig_x, orig_y, orig_w, orig_h = [int(coord * (width if i % 2 == 0 else height)) for i, coord in
                                              enumerate(original_bbox)]
        else:
            orig_x, orig_y, orig_w, orig_h = map(int, original_bbox)

        logging.debug(f"Original bbox (pixels): x={orig_x}, y={orig_y}, w={orig_w}, h={orig_h}")

        for i, frame in enumerate(self.trimmed_frames):
            if visibility[i][0]:
                # Convert tracked point to pixel coordinates
                center_x, center_y = tracks[i][0]
                if max(center_x, center_y) <= 1:  # Normalized coordinates
                    center_x = int(center_x * width)
                    center_y = int(center_y * height)

                # Calculate new bounding box coordinates
                new_x = int(center_x - orig_w / 2)
                new_y = int(center_y - orig_h / 2)

                # Ensure the box doesn't go out of frame
                new_x = max(0, min(new_x, width - orig_w))
                new_y = max(0, min(new_y, height - orig_h))

                logging.debug(f"Frame {i}: Center ({center_x}, {center_y}), New top-left: ({new_x}, {new_y})")

                # Crop the frame
                cropped_frame = frame[new_y:new_y + orig_h, new_x:new_x + orig_w]

                # Resize if necessary
                if cropped_frame.shape[:2] != (orig_h, orig_w):
                    cropped_frame = cv2.resize(cropped_frame, (orig_w, orig_h))

                self.processed_frames.append(cropped_frame)
            else:
                logging.debug(f"Frame {i}: Point not visible")
                if self.processed_frames:
                    self.processed_frames.append(self.processed_frames[-1])
                else:
                    blank_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                    self.processed_frames.append(blank_frame)

        self.current_frame = 0
        self.frame_slider.setRange(0, len(self.processed_frames) - 1)
        self.frame_slider.setValue(0)
        self.frames = self.processed_frames  # Update the main frames list
        self.update_frame()
        logging.debug(f"Processed {len(self.processed_frames)} frames")

    def update_frame(self):
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

    def save_video(self):
        try:
            if not self.processed_frames and not self.frames:
                QMessageBox.warning(self, "No Video", "There is no video to save.")
                return

            output_path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "MP4 files (*.mp4)")
            if not output_path:
                return

            frames_to_save = self.processed_frames if self.processed_frames else self.frames
            height, width = frames_to_save[0].shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

            self.show_progress_dialog("Saving video...")
            for frame in frames_to_save:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)

            out.release()
            QMessageBox.information(self, "Save Complete", "Video has been saved successfully.")
        except Exception as e:
            logging.error(f"Error in save_video: {str(e)}")
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"An error occurred while saving the video: {str(e)}")
        finally:
            self.close_progress_dialog()

    def disable_video_controls(self):
        controls = [self.start_trim_slider, self.end_trim_slider, self.trim_btn,
                    self.draw_box_btn, self.detect_btn, self.frame_slider,
                    self.delete_frame_btn, self.save_btn, self.mode_combo,
                    self.prompt_input, self.model_combo, self.crop_btn]  # Add crop_btn
        for control in controls:
            control.setEnabled(False)

    def enable_video_controls(self):
        controls = [self.start_trim_slider, self.end_trim_slider, self.trim_btn,
                    self.draw_box_btn, self.detect_btn, self.frame_slider,
                    self.delete_frame_btn, self.save_btn, self.mode_combo,
                    self.prompt_input, self.model_combo, self.crop_btn]  # Add crop_btn
        for control in controls:
            control.setEnabled(True)

    def on_frame_slider_change(self):
        self.current_frame = self.frame_slider.value()
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
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def get_edited_video(self):
        if self.processed_frames:
            return self.processed_frames
        elif self.trimmed_frames:
            return self.trimmed_frames
        else:
            return self.frames

    def show_progress_dialog(self, message):
        if self.progress_dialog is None:
            self.progress_dialog = QProgressDialog(message, "Cancel", 0, 0, self)
            self.progress_dialog.setWindowTitle("Processing")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setValue(0)
            self.progress_dialog.setRange(0, 0)  # Indeterminate progress

        self.progress_dialog.setLabelText(message)
        self.progress_dialog.show()

        # Use a timer to keep the UI responsive
        if self.progress_timer is None:
            self.progress_timer = QTimer(self)
            self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(100)  # Update every 100 ms

    def update_progress(self):
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.setValue(self.progress_dialog.value() + 1)

    def close_progress_dialog(self):
        if self.progress_timer:
            self.progress_timer.stop()
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

    def crop_video(self):
        if not self.frames:
            QMessageBox.warning(self, "No Video", "Please upload a video first.")
            return

        frame = self.frames[self.current_frame].copy()
        crop_dialog = CropDialog(frame, self)
        if crop_dialog.exec_() == QDialog.Accepted:
            crop_rect = crop_dialog.get_crop_rect()
            if crop_rect:
                self.apply_crop(crop_rect)
            else:
                QMessageBox.warning(self, "No Crop", "No area was selected for cropping.")

    def apply_crop(self, crop_rect):
        if crop_rect is None:
            return

        nx, ny, nw, nh = crop_rect  # Normalized coordinates
        self.show_progress_dialog("Cropping video...")
        try:
            cropped_frames = []
            for frame in self.frames:
                h, w = frame.shape[:2]
                x = int(nx * w)
                y = int(ny * h)
                width = int(nw * w)
                height = int(nh * h)

                # Ensure the crop rectangle is within the frame bounds
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
            self.current_frame = 0
            self.update_frame()
            QMessageBox.information(self, "Crop Complete", "Video has been cropped successfully.")
        except Exception as e:
            self.log_message(f"Error during cropping: {str(e)}", "danger")
            QMessageBox.critical(self, "Cropping Error", f"An error occurred while cropping: {str(e)}")
        finally:
            self.close_progress_dialog()




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
        painter.drawImage(self.rect(), QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888).rgbSwapped())

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
                color: #333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout = QVBoxLayout(self)

        # Display the frame with boxes
        self.label = QLabel()
        self.pixmap = QPixmap.fromImage(frame_with_boxes)
        self.label.setPixmap(self.pixmap)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.label)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # Button to open input dialog
        select_button = QPushButton("Select Box Number")
        select_button.clicked.connect(self.get_box_number)
        layout.addWidget(select_button)

        self.selected_box = None

        # Set a fixed size for the dialog based on the image size
        self.adjust_size()

    def adjust_size(self):
        screen = QApplication.primaryScreen().geometry()
        max_width = int(screen.width() * 0.8)
        max_height = int(screen.height() * 0.8)

        # Calculate the aspect ratio of the image
        aspect_ratio = self.pixmap.width() / self.pixmap.height()

        # Determine the new size while maintaining aspect ratio
        if self.pixmap.width() > max_width or self.pixmap.height() > max_height:
            if aspect_ratio > 1:  # Wider than tall
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            else:  # Taller than wide
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
        else:
            new_width = self.pixmap.width()
            new_height = self.pixmap.height()

        # Add some padding for the button and dialog frame
        new_width += 40
        new_height += 80

        self.setFixedSize(new_width, new_height)

    def get_box_number(self):
        number, ok = QInputDialog.getInt(self, "Choose Box", "Enter the number of the box you want to track:", 1, 1,
                                         100)
        if ok:
            self.selected_box = number - 1
            self.accept()


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect

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
        scaled_image = QImage(self.image.data, self.image.shape[1], self.image.shape[0],
                              self.image.strides[0], QImage.Format_RGB888).rgbSwapped()
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
            return (x / self.width(), y / self.height(),
                    w / self.width(), h / self.height())
        return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = VideoEditor()
    editor.show()
    sys.exit(app.exec_())