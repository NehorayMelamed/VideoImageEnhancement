import os
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import time
import tkinter as tk
from tkinter import filedialog, messagebox

class UiManager:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Editor")

        # Input video path
        self.input_label = tk.Label(self.root, text="Input Video Path:")
        self.input_label.pack()
        self.input_entry = tk.Entry(self.root, width=50)
        self.input_entry.pack()
        self.input_browse_button = tk.Button(self.root, text="Browse", command=self.browse_input)
        self.input_browse_button.pack()

        # Output video path
        self.output_label = tk.Label(self.root, text="Output Video Path:")
        self.output_label.pack()
        self.output_entry = tk.Entry(self.root, width=50)
        self.output_entry.pack()
        self.output_browse_button = tk.Button(self.root, text="Browse", command=self.browse_output)
        self.output_browse_button.pack()

        # Start button
        self.start_button = tk.Button(self.root, text="Start", command=self.start_editing)
        self.start_button.pack()

        self.log_text = tk.Text(self.root, height=10, width=50)
        self.log_text.pack()

    def browse_input(self):
        input_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, input_path)

    def browse_output(self):
        output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        self.output_entry.delete(0, tk.END)
        self.output_entry.insert(0, output_path)

    def start_editing(self):
        input_path = self.input_entry.get()
        output_path = self.output_entry.get()

        if not input_path or not output_path:
            messagebox.showerror("Error", "Please select both input and output paths.")
            return

        editor = VideoEditor(input_path, output_path, UiManager=self)
        editor.process_video()

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def run(self):
        self.root.mainloop()

class VideoEditor:
    def __init__(self, input_video_path, output_video_path, UiManager=None):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.cap = cv2.VideoCapture(self.input_video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.is_playing = False
        self.last_moved = 'Navigate'
        self.UiManager = UiManager

        if not self.cap.isOpened():
            self.log("Error opening video stream or file")

        self.window_name = f'Video - {self.get_video_name()}'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Adjusting window size here
        self.panel_height = 50  # Adjust as needed
        cv2.resizeWindow(self.window_name, 800, 600 + self.panel_height)  # Adjust the size as desired

        cv2.createTrackbar('Start', self.window_name, 0, self.frame_count - 1, self.nothing)
        cv2.createTrackbar('End', self.window_name, 0, self.frame_count - 1, self.nothing)
        cv2.createTrackbar('Navigate', self.window_name, 0, self.frame_count - 1, self.nothing)
        cv2.resizeWindow(self.window_name, 800, 600)  # Adjust the size as desired

    def log(self, msg):
        if self.UiManager is None:
            print(msg)
        else:
            self.UiManager.log(msg)

    @staticmethod
    def nothing(x):
        pass

    def process_video(self):
        while True:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            start_frame = cv2.getTrackbarPos('Start', self.window_name)
            end_frame = cv2.getTrackbarPos('End', self.window_name)

            if self.is_playing:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame + 1 >= end_frame:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    current_frame = start_frame
                else:
                    current_frame = current_frame + 1
            else:
                if self.last_moved == 'Start':
                    current_frame = start_frame
                elif self.last_moved == 'End':
                    current_frame = end_frame
                else:  # last_moved == 'Navigate'
                    current_frame = cv2.getTrackbarPos('Navigate', self.window_name)

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

            ret, frame = self.cap.read()
            if not ret:
                break

            # Before displaying the frame:
            h, w, _ = frame.shape
            panel_height = 50  # Adjust as needed
            panel = np.ones((panel_height, w, 3), dtype=np.uint8) * 255  # White panel

            # Add the instruction panel
            h, w, _ = frame.shape
            panel = np.ones((self.panel_height, w, 3), dtype=np.uint8) * 255  # White panel
            instructions = "Please press the next key --- p: Play/Pause | s: Save (frames range)| g: " \
                           "Get the frames range |  q: Quit"

            # Bigger font size and adjusted text position
            font_size = 1
            text_thickness = 2
            cv2.putText(panel, instructions, (10, int(self.panel_height / 2 + 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), text_thickness, cv2.LINE_AA)

            display_frame = np.vstack((frame, panel))

            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKeyEx(1) & 0xFF  # Use waitKeyEx instead of waitKey

            if self.is_playing:
                cv2.setTrackbarPos('Navigate', self.window_name, current_frame)

            if key == ord('p'):
                self.is_playing = not self.is_playing

            elif key == ord('s'):
                self.save_video(start_frame, end_frame)
                break

            elif key == ord('g'):
                self.cap.release()
                cv2.destroyAllWindows()
                self.log(f"Returning start frame {start_frame} and end frame {end_frame}.")
                return start_frame, end_frame

            elif key == ord('q'):
                break

            if cv2.getTrackbarPos('Start', self.window_name) != start_frame:
                self.last_moved = 'Start'
            elif cv2.getTrackbarPos('End', self.window_name) != end_frame:
                self.last_moved = 'End'
            elif cv2.getTrackbarPos('Navigate', self.window_name) != current_frame:
                self.last_moved = 'Navigate'

            # Adding a delay based on FPS of video to maintain original speed
            if self.is_playing:
                time.sleep(1 / self.fps)

        self.cap.release()
        cv2.destroyAllWindows()

    def save_video(self, start_frame, end_frame):
        self.cap.release()
        cv2.destroyAllWindows()

        clip = VideoFileClip(self.input_video_path)
        start_sec = start_frame / self.fps
        end_sec = end_frame / self.fps
        subclip = clip.subclip(start_sec, end_sec)
        subclip.write_videofile(self.output_video_path, codec='libx264')
        self.log(f"Success to cut and save video - {self.output_video_path}")
        self.cap = cv2.VideoCapture(self.input_video_path)

    def get_video_name(self):
        return os.path.basename(self.input_video_path).split('.')[0]

if __name__ == '__main__':
    ui_manager = UiManager()
    ui_manager.run()
