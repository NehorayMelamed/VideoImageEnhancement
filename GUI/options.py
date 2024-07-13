import os
import sys
from pathlib import Path
import mimetypes
import tkinter as tk

import PARAMETER

### deblur
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer/to_neo")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer/RVRT_deblur_inference.py")



### Ben - NafNet





class InputType:
    DIR_IMAGES = "dir of images"
    VIDEO = "video"
    IMAGE = "image"
    OTHER = "other"


class InputChecker:
    @staticmethod
    def check_input_type(input_source):
        if os.path.isdir(input_source):
            # Check if the directory contains only images
            if all(mimetypes.guess_type(file)[0].startswith('image') for file in os.listdir(input_source)):
                return InputType.DIR_IMAGES
            else:
                return InputType.OTHER
        elif os.path.isfile(input_source):
            mime_type, _ = mimetypes.guess_type(input_source)
            if mime_type is not None:
                if mime_type.startswith('video'):
                    return InputType.VIDEO
                elif mime_type.startswith('image'):
                    return InputType.IMAGE
        return InputType.OTHER


class DenoiseOptions:
    RVRT = "Rvrt"
    DENOISE_YOAV = "denoise yoav"
    STABILIZE_ENTIRE_FRAME = "stabilize entire frame"
    STABLE_OBJECT_COTRACKER = "stable object 'with co-tracker'"
    STABLE_OBJECT_OPTICAL_FLOW = "stable object with optical flow tracker"
    STABLE_OBJECT_CLASSIC_TRACKER = "stable object 'classic tracker'"

    def __init__(self, log_function):
        self.log_function = log_function

    def perform_RVRT(self, input_source):
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DENOISE", service_name="RVRT",
                                                              input_name=input_name)
        self.log_function(f"Performing RVRT denoise on: {input_source}", "info")
        # Example function call (replace with actual function)
        # run_rvrt_denoise(input_source, path_to_directory_output)
        # save_results(self.log_function, path_to_directory_output,
        #              {"denoise_result.mp4": b"dummy data", "segmentation.pt": b"dummy data"})

    def perform_DENOISE_YOAV(self, input_source):
        DenoiseWindow(tk.Tk(), self.log_function, input_source)

    def perform_STABILIZE_ENTIRE_FRAME(self, input_source):
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DENOISE",
                                                              service_name="STABILIZE_ENTIRE_FRAME",
                                                              input_name=input_name)
        self.log_function(f"Performing stabilize entire frame on: {input_source}", "info")
        # Example function call (replace with actual function)
        # run_stabilize_entire_frame(input_source, path_to_directory_output)
        # save_results(self.log_function, path_to_directory_output,
        #              {"denoise_result.mp4": b"dummy data", "segmentation.pt": b"dummy data"})

    def perform_STABLE_OBJECT_COTRACKER(self, input_source):
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DENOISE",
                                                              service_name="STABLE_OBJECT_COTRACKER",
                                                              input_name=input_name)
        self.log_function(f"Performing stable object 'with co-tracker' on: {input_source}", "info")
        # Example function call (replace with actual function)
        # run_stable_object_cotracker(input_source, path_to_directory_output)
        # save_results(self.log_function, path_to_directory_output,
        #              {"denoise_result.mp4": b"dummy data", "segmentation.pt": b"dummy data"})

    def perform_STABLE_OBJECT_OPTICAL_FLOW(self, input_source):
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DENOISE",
                                                              service_name="STABLE_OBJECT_OPTICAL_FLOW",
                                                              input_name=input_name)
        self.log_function(f"Performing stable object with optical flow tracker on: {input_source}", "info")
        # Example function call (replace with actual function)
        # run_stable_object_optical_flow(input_source, path_to_directory_output)
        # save_results(self.log_function, path_to_directory_output,
        #              {"denoise_result.mp4": b"dummy data", "segmentation.pt": b"dummy data"})

    def perform_STABLE_OBJECT_CLASSIC_TRACKER(self, input_source):
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DENOISE",
                                                              service_name="STABLE_OBJECT_CLASSIC_TRACKER",
                                                              input_name=input_name)
        self.log_function(f"Performing stable object 'classic tracker' on: {input_source}", "info")
        # Example function call (replace with actual function)
        # run_stable_object_classic_tracker(input_source, path_to_directory_output)
        # save_results(self.log_function, path_to_directory_output,
        #              {"denoise_result.mp4": b"dummy data", "segmentation.pt": b"dummy data"})


class DeblurOptions:
    RVRT = "Rvrt"
    REALBASICVSR = "RealBasicVSR"
    RVRT_OMER = "RvrtOmer"
    NAFNET = "NafNet"
    BLUR_KERNEL_DEBLUR = "Blur kernel and deblur"

    def __init__(self, log_function):
        self.log_function = log_function

    def perform_RVRT(self, input_source):
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DEBLUR", service_name="RVRT",
                                                              input_name=input_name)
        self.log_function(f"Performing RVRT deblur on: {input_source}", "info")
        # Example function call (replace with actual function)
        # run_rvrt_deblur(input_source, path_to_directory_output)
        # save_results(self.log_function, path_to_directory_output,
        #              {"deblur_result.mp4": b"dummy data", "segmentation.pt": b"dummy data"})

    def perform_REALBASICVSR(self, input_source):
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DEBLUR", service_name="REALBASICVSR",
                                                              input_name=input_name)
        self.log_function(f"Performing RealBasicVSR deblur on: {input_source}", "info")
        # Example function call (replace with actual function)
        # run_realbasicvsr_deblur(input_source, path_to_directory_output)
        # save_results(self.log_function, path_to_directory_output,
        #              {"deblur_result.mp4": b"dummy data", "segmentation.pt": b"dummy data"})

    def perform_RVRT_OMER(self, input_source):
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DEBLUR", service_name="RVRT_OMER",
                                                              input_name=input_name)
        self.log_function(f"Performing RvrtOmer deblur on: {input_source}", "info")
        from Omer.to_neo.main_deblur import main_deblur

        # Example function call (replace with actual function)
        if main_deblur(video_path=input_source, use_roi=False, save_videos=True,
                    blur_video_mp4=os.path.join(path_to_directory_output, "blur_video.mp4"),
                    deblur_video_mp4=os.path.join(path_to_directory_output, "deblur_video.mp4")
                    ) is False:
            self.log_function('Failed to perform deblur omer')
            return False


    def perform_NAFNET(self, input_source):
        from ben_deblur.ImageDeBlur.deblur_functions import main_nafnet_deblur

        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DEBLUR", service_name="NAFNET",
                                                              input_name=input_name)
        self.log_function(f"Performing NafNet deblur on: {input_source}", "info")

        if main_nafnet_deblur(video_path=input_source, output_folder=path_to_directory_output) is False:
            self.log_function("failed to perform nafnet")



    def perform_BLUR_KERNEL_DEBLUR(self, input_source):
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DEBLUR",
                                                              service_name="BLUR_KERNEL_DEBLUR", input_name=input_name)
        self.log_function(f"Performing Blur kernel and deblur on: {input_source}", "info")
        # Example function call (replace with actual function)
        # run_blur_kernel_deblur(input_source, path_to_directory_output)
        # save_results(self.log_function, path_to_directory_output,
        #              {"deblur_result.mp4": b"dummy data", "segmentation.pt": b"dummy data"})


def save_results(log_function, output_directory, results):
    for result_name, result_data in results.items():
        result_path = output_directory / result_name
        with open(result_path, 'wb') as result_file:
            result_file.write(result_data)

    log_function(f"Results saved in {output_directory}", "info")


def build_directory_to_service(base_directory, service_name, input_name):
    FULL_PATH_OUTPUT = Path("OUTPUT")
    if not FULL_PATH_OUTPUT.exists():
        FULL_PATH_OUTPUT.mkdir()

    path_to_output_directory_base = FULL_PATH_OUTPUT / base_directory
    if not path_to_output_directory_base.exists():
        path_to_output_directory_base.mkdir()

    path_to_output_directory_service = path_to_output_directory_base / service_name
    if not path_to_output_directory_service.exists():
        path_to_output_directory_service.mkdir()

    path_to_output_directory_service_with_input_name = path_to_output_directory_service / input_name
    if not path_to_output_directory_service_with_input_name.exists():
        path_to_output_directory_service_with_input_name.mkdir()

    return path_to_output_directory_service_with_input_name



class DenoiseWindow:
    def __init__(self, master, logger, input_source):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Denoise")
        self.logger = logger
        self.video_path = input_source

        # Description
        self.description_label = tk.Label(self.window,
                                          text="This is a denoise section for removing noise from the video")
        self.description_label.pack()

        # Radio button for selecting ROI or entire video
        self.use_roi_var = tk.BooleanVar(value=False)  # Default selection is not using ROI
        self.roi_radio = tk.Radiobutton(self.window, text="Use ROI", variable=self.use_roi_var, value=True)
        self.roi_radio.pack()

        self.entire_video_radio = tk.Radiobutton(self.window, text="Use Entire Video", variable=self.use_roi_var,
                                                 value=False)
        self.entire_video_radio.pack()

        # Labels and entry boxes for window_size_temporal and stride parameters
        self.window_size_label = tk.Label(self.window, text="window_size_temporal (Default: 4):")
        self.window_size_label.pack()
        self.window_size_temporal_entry = tk.Entry(self.window)
        self.window_size_temporal_entry.insert(tk.END, "4")  # Default value for window_size_temporal
        self.window_size_temporal_entry.pack()

        self.stride_label = tk.Label(self.window, text="stride (Default: 1):")
        self.stride_label.pack()
        self.stride_entry = tk.Entry(self.window)
        self.stride_entry.insert(tk.END, "1")  # Default value for stride
        self.stride_entry.pack()

        # Button to perform denoising
        self.apply_button = tk.Button(self.window, text="Apply", command=self.perform_denoising)
        self.apply_button.pack()

    def perform_denoising(self):
        from Yoav_denoise_new.denois.main_denoise import main_denoise


        if self.video_path:
            use_roi = self.use_roi_var.get()

            # Get window_size_temporal and stride values
            window_size_temporal = int(self.window_size_temporal_entry.get())
            stride = int(self.stride_entry.get())

            base_video_name = os.path.basename(self.video_path).split('.')[0]
            path_to_directory_output = build_directory_to_service(base_directory="DENOISE", service_name="DENOISE_YOAV",
                                                                  input_name=base_video_name)

            full_output_video_path = path_to_directory_output / f"denoise_{base_video_name}.mp4"
            full_input_for_output_video_path = path_to_directory_output / f"input_noise_{base_video_name}.mp4"

            # Example function call (replace with actual function)
            if main_denoise(window_size_temporal=window_size_temporal, stride=stride, video_path=self.video_path,
                            use_roi=use_roi,
                            noise_save_video_file_name=full_input_for_output_video_path,
                            denoise_save_video_file_name=full_output_video_path):
                self.logger(f"Denoise process finished successfully")
                self.logger(f"Data saved into {full_input_for_output_video_path}")
            else:
                self.logger(f"Denoise process encountered an error")

        else:
            self.logger("Please select a video file.")
        self.window.destroy()

