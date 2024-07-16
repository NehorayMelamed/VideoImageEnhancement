import tkinter as tk
import sys
import mimetypes
import os

import numpy as np

import AlignClass
import PARAMETER
from WienerDeconvolution import WienerDeconvolution
from pathlib import Path

from ben_deblur.ImageDeBlur.deblur_functions import main_nafnet_list_of_frames

sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer/to_neo")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer/RVRT_deblur_inference.py")

from AlignClass import *





class InputType:
    DIR_IMAGES = "dir of images"
    VIDEO = "video"
    IMAGE = "image"
    OTHER = "other"


class InputChecker:
    @staticmethod
    def check_input_type(input_source):
        if os.path.isdir(input_source):
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

    class OptionsServices:
        SCC = "SCC"
        ECC = "ECC"
        FEATURE_BASED = "FEATURE_BASED"
        OPTICAL_FLOW = "OPTICAL_FLOW"
        DENOISE_COT = "DENOISE_COT"
        RV_CLASSIC = "RV_CLASSIC"
        DENOISE_YOV = "DENOISE_YOV"

    def __init__(self, log_function):
        self.log_function = log_function


    def perform_SCC(self, input_source, source_data, input_method, user_input):
        self.log_function(f"Performing {DenoiseOptions.OptionsServices.SCC} denoise on: {input_source}", "info")
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service("DENOISE", DenoiseOptions.OptionsServices.SCC, input_name)
        aligned_crops, average_crop = AlignClass.align_and_average_frames_using_SCC(source_data,
                                                                                    input_method=input_method, user_input=user_input)
        # save_results(aligned_crops=aligned_crops, average_crop=average_crop, output_directory=path_to_directory_output)
        return aligned_crops

    def perform_ECC(self, input_source, source_data, input_method, user_input):
        self.log_function(f"Performing {DenoiseOptions.OptionsServices.ECC} denoise on: {input_source}", "info")
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service("DENOISE", DenoiseOptions.OptionsServices.ECC, input_name)
        aligned_crops, average_crop = AlignClass.align_and_average_frames_using_ECC(source_data,
                                                                                    input_method=input_method, user_input=user_input)
        aligned_crops = [np.clip(aligned_crops[i]*255, 0, 255).astype(np.uint8) for i in np.arange(len(aligned_crops))]
        imshow_video(list_to_numpy(aligned_crops), FPS=1)
        # save_results(aligned_crops=aligned_crops, average_crop=average_crop, output_directory=path_to_directory_output)
        return aligned_crops

    def perform_FEATURE_BASED(self, input_source, source_data, input_method, user_input):
        self.log_function(f"Performing {DenoiseOptions.OptionsServices.FEATURE_BASED} denoise on: {input_source}", "info")
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service("DENOISE", DenoiseOptions.OptionsServices.FEATURE_BASED, input_name)
        aligned_crops, average_crop = AlignClass.align_and_average_frames_using_FeatureBased(frames=source_data,
                                                                                             input_method=input_method, user_input=user_input)
        save_results(aligned_crops, average_crop, output_directory=path_to_directory_output)
        return aligned_crops

    def perform_OPTICAL_FLOW(self, input_source, source_data, input_method, user_input):
        self.log_function(f"Performing  {DenoiseOptions.OptionsServices.OPTICAL_FLOW} denoise on: {input_source}", "info")
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service("DENOISE", DenoiseOptions.OptionsServices.OPTICAL_FLOW, input_name)
        aligned_crops, average_crop = AlignClass.align_and_average_frames_using_FlowFormer_and_PWC(source_data,
                                                                                            input_method=input_method, user_input=user_input)
        aligned_crops = [scale_array_stretch_hist(np.clip(aligned_crops[i], 0, 255), min_max_values_to_scale_to=(0,255)).astype(np.uint8) for i in np.arange(len(aligned_crops))]
        # imshow_video(list_to_numpy(aligned_crops), FPS=1)
        # imshow_np(average_crop/255)
        # save_results(aligned_crops, average_crop, output_directory=path_to_directory_output)
        return aligned_crops

    def perform_DENOISE_COT(self, input_source, source_data, input_method, user_input):
        self.log_function(f"Performing {DenoiseOptions.OptionsServices.DENOISE_COT}on: {input_source}", "info")
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service("DENOISE", DenoiseOptions.OptionsServices.DENOISE_COT, input_name)
        aligned_crops, average_crop = AlignClass.align_and_average_frames_using_CoTracker(frames=source_data,
                                                                                          input_method=input_method, user_input=user_input)
        imshow_video(list_to_numpy(aligned_crops), FPS=1)
        imshow_video(list_to_numpy(source_data), FPS=1)
        save_results(aligned_crops, average_crop, output_directory=path_to_directory_output)
        return aligned_crops

    def perform_RV_CLASSIC(self, input_source, source_data, input_method, user_input):
        self.log_function(f"Performing {DenoiseOptions.OptionsServices.RV_CLASSIC} denoise on: {input_source}", "info")
        return None

    def perform_DENOISE_YOV(self, input_source, source_data, input_method, user_input):
        input_name = os.path.basename(input_source).split('.')[0]
        self.log_function(f"Performing {DenoiseOptions.OptionsServices.DENOISE_YOV} on: {input_source}", "info")
        # path_to_directory_output = build_directory_to_service("DENOISE", DenoiseOptions.OptionsServices.DENOISE_YOV, input_name)
        aligned_crops, average_crop = AlignClass.align_and_average_frames_using_FlowFormer_and_PWC(source_data,
                                                                                                   input_method=input_method, user_input=user_input)
        aligned_crops = [np.clip(aligned_crops[i], 0,255).astype(np.uint8) for i in np.arange(len(aligned_crops))]
        return aligned_crops


class DeblurOptions:
    class OptionsServices:
        RV_OM = "RV_OM"
        NAFNET = "NAFNET"
        NUBKE = "NUBKE"
        NUMBKE2WIN = "NUMBKE2WIN"
        UNSUPERWIN = "UNSUPERWIN"
        SHARPEN = "SHARPEN"
        STRETCH = "STRETCH"
        MEAN = "MEAN"

    def __init__(self, log_function):
        self.log_function = log_function



    def perform_STRETCH(self, input_source, source_data, input_method, user_input):
        self.log_function(f"Performing {DeblurOptions.OptionsServices.NAFNET} deblur on: {input_source}", "info")
        all_processed_frames = AlignClass.stretch_histogram(frames=source_data)
        return all_processed_frames

    def perform_SHARPEN(self, input_source, source_data, input_method, user_input):
        self.log_function(f"Performing {DeblurOptions.OptionsServices.SHARPEN} deblur on: {input_source}", "info")
        all_processed_frames = AlignClass.unsharp_mask_list(frames=source_data)
        return all_processed_frames

    def perform_MEAN(self, input_source, source_data, input_method, user_input):
        self.log_function(f"Performing {DeblurOptions.OptionsServices.NAFNET} deblur on: {input_source}", "info")
        all_processed_frames = AlignClass.mean_over_frames(frames=source_data)
        return all_processed_frames


    def perform_RV_OM(self, input_source, source_data, input_method, user_input):
        #TOdO SUPPORT NUMPY ARRAY AND NOT A VIDEO PATH
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service("DEBLUR", DeblurOptions.OptionsServices.RV_OM, input_name)
        self.log_function(f"Performing {DeblurOptions.OptionsServices.RV_OM} deblur on: {input_source}", "info")
        from Omer.to_neo.main_deblur import main_deblur_list_of_frames

        number_of_frames = len(source_data)
        max_number_of_frames = int(np.floor(number_of_frames//2)*2)
        # H_new,W_new = source_data[0].shape[0:2]
        # H_new = int((H_new//4)*4)
        # W_new = int((W_new//4)*4)
        # bla = source_data[0:max_number_of_frames]
        # bla = [bla[i][0:H_new, 0:W_new] for i in np.arange(len(bla))]
        # bla = [crop_tensor(bla[i],(300,400)) for i in np.arange(len(bla))]
        # bla = numpy_to_list(np.random.randn(8,255,255,3))

        numpy_video_result = main_deblur_list_of_frames(list_of_numpy_frames=source_data[0:max_number_of_frames], use_roi=False, save_videos=True,
                                         blur_video_mp4=os.path.join(path_to_directory_output, "blur_video.mp4"),
                                         deblur_video_mp4=os.path.join(path_to_directory_output, "deblur_video.mp4"))

        return numpy_video_result

    def perform_NAFNET(self, input_source, source_data, input_method, user_input):
        #TOdO SUPPORT NUMPY ARRAY AND NOT A VIDEO PATH

        from ben_deblur.ImageDeBlur.deblur_functions import main_nafnet_deblur

        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service("DEBLUR", DeblurOptions.OptionsServices.NAFNET, input_name)
        self.log_function(f"Performing {DeblurOptions.OptionsServices.NAFNET} deblur on: {input_source}", "info")

        all_processed_frames = main_nafnet_list_of_frames(frame_list=source_data)
        return all_processed_frames


    def perform_NUBKE(self, input_source, source_data, input_method, user_input):
        self.log_function(f"Performing {DeblurOptions.OptionsServices.NUBKE} deblur on: {input_source}", "info")
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service("DEBLUR", DeblurOptions.OptionsServices.NUBKE, input_name)
        average_blur_kernels_list, deblurred_crops_list = WienerDeconvolution.get_blur_kernels_and_deblurred_images_using_NUBKE(
            source_data,input_method=input_method, user_input=user_input)
        # imshow_np(DeblurOptions.unsharp_mask(deblurred_crops_list[-1], sigma=3, strength=5.5))
        deblurred_crops_list = [np.clip(BW2RGB(deblurred_crops_list[i])*255, 0, 255).astype(np.uint8) for i in
                            np.arange(len(deblurred_crops_list))]
        # imshow_np(deblurred_crops_list[-1])
        # imshow_video(list_to_numpy(deblurred_crops_list), FPS=1)
        # imshow_video(list_to_numpy(source_data), FPS=1)
        # imshow_np(source_data[0]-deblurred_crops_list[0]*255)
        # save_results(average_blur_kernels_list=average_blur_kernels_list, deblurred_crops_list=deblurred_crops_list,
        #              output_directory=path_to_directory_output)
        return deblurred_crops_list

    def perform_NUMBKE2WIN(self, input_source, source_data, input_method, user_input):
        self.log_function(f"Performing {DeblurOptions.OptionsServices.NUMBKE2WIN} deblur on: {input_source}", "info")
        input_name = os.path.basename(input_source).split('.')[0]
        path_to_directory_output = build_directory_to_service("DEBLUR", DeblurOptions.OptionsServices.NUMBKE2WIN, input_name)

        if isinstance(source_data, list) and isinstance(source_data[0], numpy.ndarray):
            source_data = source_data[0]  ### For single image
        else:
            raise ValueError(f"Not supported  source_data {source_data} ")
        deblurred_images, string_explanations = WienerDeconvolution.get_blur_kernel_using_NUBKE_and_deblur_using_Wiener_all_options(
            source_data, input_method=input_method, user_input=user_input)
        # save_results(average_blur_kernels_list=deblurred_images,
        #              output_directory=path_to_directory_output)
        deblurred_images = [np.clip(BW2RGB(deblurred_images[i]), 0,255).astype(np.uint8) for i in np.arange(len(deblurred_images))]
        return deblurred_images

    def perform_UNSUPERWIN(self, input_source, source_data, input_method, user_input):
        self.log_function(f"Performing {DeblurOptions.OptionsServices.UNSUPERWIN} deblur on: {input_source}", "info")
        # Add implementation here
        return None




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


def save_results(aligned_crops=None, average_crop=None, average_blur_kernels_list=None, deblurred_crops_list=None,
                 output_directory='output'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if aligned_crops is not None:
        video_path = os.path.join(output_directory, 'aligned_crops.mp4')
        height, width, layers = aligned_crops[0].shape
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        for frame in aligned_crops:
            video.write(frame)

        video.release()

        frames_dir = os.path.join(output_directory, 'dir_aligned_crops_frames')
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        for i, frame in enumerate(aligned_crops):
            frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
            cv2.imwrite(frame_path, frame)

        np.save(os.path.join(output_directory, 'aligned_crops.npy'), aligned_crops)

    if average_crop is not None:
        average_crop_path = os.path.join(output_directory, 'average_crop.png')
        cv2.imwrite(average_crop_path, average_crop)

        np.save(os.path.join(output_directory, 'average_crop.npy'), average_crop)

    if average_blur_kernels_list is not None:
        blur_kernels_dir = os.path.join(output_directory, 'dir_average_blur_kernels')
        if not os.path.exists(blur_kernels_dir):
            os.makedirs(blur_kernels_dir)

        for i, kernel in enumerate(average_blur_kernels_list):
            kernel_path = os.path.join(blur_kernels_dir, f'kernel_{i:04d}.png')
            cv2.imwrite(kernel_path, kernel)

        np.save(os.path.join(output_directory, 'average_blur_kernels_list.npy'), average_blur_kernels_list)

    if deblurred_crops_list is not None:
        deblurred_video_path = os.path.join(output_directory, 'deblurred_crops.mp4')
        height, width, layers = deblurred_crops_list[0].shape
        deblurred_video = cv2.VideoWriter(deblurred_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        for frame in deblurred_crops_list:
            deblurred_video.write(frame)

        deblurred_video.release()

        deblurred_frames_dir = os.path.join(output_directory, 'dir_deblurred_crops_frames')
        if not os.path.exists(deblurred_frames_dir):
            os.makedirs(deblurred_frames_dir)

        for i, frame in enumerate(deblurred_crops_list):
            frame_path = os.path.join(deblurred_frames_dir, f'frame_{i:04d}.png')
            cv2.imwrite(frame_path, frame)

        np.save(os.path.join(output_directory, 'deblurred_crops_list.npy'), deblurred_crops_list)
