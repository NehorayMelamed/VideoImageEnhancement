import sys
import mimetypes
import tkinter as tk
import PARAMETER
from WienerDeconvolution import WienerDeconvolution

# Update sys.path to include required directories for deblur
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer/to_neo")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer/RVRT_deblur_inference.py")

from AlignClass import *


# Updated options for denoise and deblur methods
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
    SCC = "SCC"
    ECC = "ECC"
    FEATURE_BASED = "FeatureBased"
    OPTICAL_FLOW = "Optical Flow"
    DENOISE_COT = "DenoiseCoT"
    RV_CLASSIC = "RV-Classic"
    DENOISE_YOV = "Denoise-Yov"

    def __init__(self, log_function):
        self.log_function = log_function

    def perform_SCC(self, input_source, source_data):
        self.log_function(f"Performing SCC denoise on: {input_source}", "info")
        # Add implementation here

    def perform_ECC(self, input_source, source_data):
        self.log_function(f"Performing ECC denoise on: {input_source}", "info")
        # Add implementation here

    def perform_FEATURE_BASED(self, input_source, source_data):
        self.log_function(f"Performing FeatureBased denoise on: {input_source}", "info")
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DENOISE",
                                                              service_name=DenoiseOptions.FEATURE_BASED,
                                                              input_name=input_name)
        aligned_crops, average_crop = AlignClass.align_and_average_frames_using_FeatureBased(frames=source_data)
        save_results(aligned_crops=aligned_crops, average_crop=average_crop, output_directory=path_to_directory_output)
        # Add implementation here

    def perform_OPTICAL_FLOW(self, input_source, source_data):
        self.log_function(f"Performing Optical Flow denoise on: {input_source}", "info")
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DENOISE",
                                                              service_name=DenoiseOptions.OPTICAL_FLOW,
                                                              input_name=input_name)
        aligned_crops, average_crop = AlignClass.align_and_average_frames_using_OpticalFlow(source_data)
        save_results(aligned_crops=aligned_crops, average_crop=average_crop, output_directory=path_to_directory_output)

        # Add implementation here

    def perform_DENOISE_COT(self, input_source, source_data):
        self.log_function(f"Performing denoise co tracker denoise on: {input_source}", "info")
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DENOISE",
                                                              service_name=DenoiseOptions.DENOISE_COT,
                                                              input_name=input_name)
        aligned_crops, average_crop = AlignClass.align_and_average_frames_using_CoTracker(frames=source_data)
        save_results(aligned_crops=aligned_crops, average_crop=average_crop, output_directory=path_to_directory_output)
        # Add implementation here

    def perform_RV_CLASSIC(self, input_source, source_data):
        self.log_function(f"Performing RV-Classic denoise on: {input_source}", "info")
        # Add implementation here

    def perform_DENOISE_YOV(self, input_source, source_data):
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DENOISE", service_name="DENOISE_YOV",
                                                              input_name=input_name)
        self.log_function(f"Performing Denoise-Yov denoise on: {input_source}", "info")
        # Example function call (replace with actual function)
        # run_denoise_yov(source_data, path_to_directory_output)


class DeblurOptions:
    RV_OM = "RV-Om"
    NAFNET = "NafNet"
    NUBKE = "NubKe"
    NUMBKE2WIN = "Numbke2Win"
    UNSUPERWIN = "UnsuperWin"

    def __init__(self, log_function):
        self.log_function = log_function

    def perform_RV_OM(self, input_source, source_data):
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DEBLUR", service_name="RV-Om",
                                                              input_name=input_name)
        self.log_function(f"Performing RV-Om deblur on: {input_source}", "info")
        from Omer.to_neo.main_deblur import main_deblur

        # Example function call (replace with actual function)
        if main_deblur(video_path=input_source, use_roi=False, save_videos=True,
                       blur_video_mp4=os.path.join(path_to_directory_output, "blur_video.mp4"),
                       deblur_video_mp4=os.path.join(path_to_directory_output, "deblur_video.mp4")
                       ) is False:
            self.log_function('Failed to perform deblur RV-Om')
            return False

    def perform_NAFNET(self, input_source, source_data):
        from ben_deblur.ImageDeBlur.deblur_functions import main_nafnet_deblur

        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DEBLUR", service_name="NafNet",
                                                              input_name=input_name)
        self.log_function(f"Performing NafNet deblur on: {input_source}", "info")

        if main_nafnet_deblur(video_path=input_source, output_folder=path_to_directory_output) is False:
            self.log_function("Failed to perform NafNet deblur")

    def perform_NUBKE(self, input_source, source_data):
        self.log_function(f"Performing Optical Flow denoise on: {input_source}", "info")
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DEBLUR", service_name=DeblurOptions.NUBKE,
                                                              input_name=input_name)
        average_blur_kernels_list, deblurred_crops_list = WienerDeconvolution.get_blur_kernels_and_deblurred_images_using_NUBKE(
            source_data[0:8])
        save_results(average_blur_kernels_list=average_blur_kernels_list, deblurred_crops_list=deblurred_crops_list,
                     output_directory=path_to_directory_output)

    def perform_NUMBKE2WIN(self, input_source, source_data):
        self.log_function(f"Performing Optical Flow denoise on: {input_source}", "info")
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service(base_directory="DEBLUR", service_name=DeblurOptions.NUBKE,
                                                              input_name=input_name)
        average_blur_kernels_list, deblurred_crops_list = WienerDeconvolution.get_blur_kernel_using_NUBKE_and_deblur_using_Wiener_all_options()
        save_results(average_blur_kernels_list=average_blur_kernels_list, deblurred_crops_list=deblurred_crops_list,
                     output_directory=path_to_directory_output)

    def perform_UNSUPERWIN(self, input_source, source_data):
        self.log_function(f"Performing UnsuperWin deblur on: {input_source}", "info")
        # Add implementation here


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
            path_to_directory_output = build_directory_to_service(base_directory="DENOISE", service_name="DENOISE_YOV",
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


def build_directory_to_service(base_directory, service_name, input_name):
    FULL_PATH_OUTPUT = Path("OUTPUT")
    if not FULL_PATH_OUTPUT.exists():
        FULL_PATH_OUTPUT.mkdir()

    path_to_output_directory_base = FULL_PATH_OUTPUT / base_directory
    if not path_to_output_directory_base.exists():
        path_to_output_directory_base.mkdir()

    path_to_output_directory_input = path_to_output_directory_base / input_name
    if not path_to_output_directory_input.exists():
        path_to_output_directory_input.mkdir()

    path_to_output_directory_service = path_to_output_directory_input / service_name
    if not path_to_output_directory_service.exists():
        path_to_output_directory_service.mkdir()

    return path_to_output_directory_service


import os
import cv2
import numpy as np
from pathlib import Path


def save_results(aligned_crops=None, average_crop=None, average_blur_kernels_list=None, deblurred_crops_list=None,
                 output_directory='output'):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save aligned crops as a video and individual frames if not None
    if aligned_crops is not None:
        video_path = os.path.join(output_directory, 'aligned_crops.mp4')
        height, width, layers = aligned_crops[0].shape
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        for frame in aligned_crops:
            video.write(frame)

        video.release()

        # Save aligned crops as individual frames
        frames_dir = os.path.join(output_directory, 'dir_aligned_crops_frames')
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        for i, frame in enumerate(aligned_crops):
            frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
            cv2.imwrite(frame_path, frame)

        # Save numpy data for aligned crops
        np.save(os.path.join(output_directory, 'aligned_crops.npy'), aligned_crops)

    # Save average crop as a PNG image and numpy file if not None
    if average_crop is not None:
        average_crop_path = os.path.join(output_directory, 'average_crop.png')
        cv2.imwrite(average_crop_path, average_crop)

        # Save numpy data for average crop
        np.save(os.path.join(output_directory, 'average_crop.npy'), average_crop)

    # Save average blur kernels list as individual frames if not None
    if average_blur_kernels_list is not None:
        blur_kernels_dir = os.path.join(output_directory, 'dir_average_blur_kernels')
        if not os.path.exists(blur_kernels_dir):
            os.makedirs(blur_kernels_dir)

        for i, kernel in enumerate(average_blur_kernels_list):
            kernel_path = os.path.join(blur_kernels_dir, f'kernel_{i:04d}.png')
            cv2.imwrite(kernel_path, kernel)

        # Save numpy data for average blur kernels list
        np.save(os.path.join(output_directory, 'average_blur_kernels_list.npy'), average_blur_kernels_list)

    # Save deblurred crops as a video and individual frames if not None
    if deblurred_crops_list is not None:
        deblurred_video_path = os.path.join(output_directory, 'deblurred_crops.mp4')
        height, width, layers = deblurred_crops_list[0].shape
        deblurred_video = cv2.VideoWriter(deblurred_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        for frame in deblurred_crops_list:
            deblurred_video.write(frame)

        deblurred_video.release()

        # Save deblurred crops as individual frames
        deblurred_frames_dir = os.path.join(output_directory, 'dir_deblurred_crops_frames')
        if not os.path.exists(deblurred_frames_dir):
            os.makedirs(deblurred_frames_dir)

        for i, frame in enumerate(deblurred_crops_list):
            frame_path = os.path.join(deblurred_frames_dir, f'frame_{i:04d}.png')
            cv2.imwrite(frame_path, frame)

        # Save numpy data for deblurred crops list
        np.save(os.path.join(output_directory, 'deblurred_crops_list.npy'), deblurred_crops_list)

# Example usage:
# aligned_crops = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
# average_crop = np.zeros((480, 640, 3), dtype=np.uint8)
# average_blur_kernels_list = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
# deblurred_crops_list = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
# save_results(aligned_crops, average_crop, average_blur_kernels_list, deblurred_crops_list, './output_directory')
