import tkinter as tk
import sys
import mimetypes
import os

import numpy as np
import ast
import AlignClass
import PARAMETER
# sys.path.append(os.path.join(PARAMETER.ImageRestoration_base_dir, "FMA_Net"))
# sys.path.append(os.path.join(PARAMETER.ImageRestoration_base_dir))
#
# from ImageRestoration.FMA_Net.model_usage import main_fma_net_interface
from ImageRestoration.LaKDNet.main_inference import main_laknet_deblur_infer_numpy, main_laknet_defocus_infer_numpy
from ImageRestoration.UFPDeblur.model_usage import main_ufpd_interface

from util.video_to_numpy_array import get_video_frames
from vnlb_repo.main_interface import vnlb_denoise

sys.path.append("../")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer/to_neo")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer/RVRT_deblur_inference.py")

from WienerDeconvolution import WienerDeconvolution
from pathlib import Path


from RVRT.main_test_5 import main_denoise_rvrt_interface


# sys.path.append(PARAMETER.base_dir_ben_code)
from ben_deblur.ImageDeBlur.deblur_functions import main_nafnet_list_of_frames


# sys.path.append(PARAMETER.Base_dir_NAFNet)
# from NAFNet_width64 import nafnet_config as nafnet_config
#
# # Ensure the BASE_PROJECT is correctly added to the path for imports
# sys.path.append(f"{PARAMETER.BASE_PROJECT}/ben_deblur/ImageDeBlur")


from AlignClass import *
from Classic_ISP import *

class InputType:
    DIR_IMAGES = "dir of images"
    VIDEO = "video"
    IMAGE = "image"
    OTHER = "other"


def add_text_to_images(images, texts, font_size=2, font_thickness=2):
    """
    Adds a text to the center top of each image in the list.

    Parameters:
    -----------
    images : list of np.ndarray
        List of images.
    texts : list of str
        List of strings to print on each image.
    font_size : int
        The size of the font for the text.
    font_thickness : int
        The thickness of the font for the text.

    Returns:
    --------
    list of np.ndarray
        List of images with text printed on them.
    """
    if len(images) != len(texts):
        raise ValueError("The number of images must be equal to the number of texts.")

    font = cv2.FONT_HERSHEY_SIMPLEX

    images_with_text = []

    for img, text in zip(images, texts):
        # Calculate the width and height of the text to be added
        text_size = cv2.getTextSize(text, font, font_size, font_thickness)[0]

        # Calculate X position to center the text
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = text_size[1] + 10  # A little padding from the top

        # Add text to image
        img_with_text = cv2.putText(img.copy(), text, (text_x, text_y), font, font_size, (255, 255, 255),
                                    font_thickness, cv2.LINE_AA)

        # Append the modified image to the list
        images_with_text.append(img_with_text)

    return images_with_text

def frames_to_constant_format(frames, dtype_requested='uint8', range_requested=[0, 255], channels_requested=3, threshold=5):
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
    original_channels = first_frame.shape[2] if len(first_frame.shape) == 3 else 1  # Determine the original number of channels

    if original_dtype == np.uint8:  # Check if the original dtype is uint8
        original_range = [0, 255]  # Set original range to [0, 255]
    else:
        max_val = np.max(first_frame)  # Get the maximum value of the first frame
        original_range = [0, 255] if max_val > threshold else [0, 1]  # Determine the original range based on max value and threshold

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
                frame = np.clip(frame / 255.0, 0, 1)  # Convert range from [0, 255] to [0, 1]
            elif original_range == [0, 1] and range_requested == [0, 255]:
                frame = np.clip(frame * 255.0, 0, 255)  # Convert range from [0, 1] to [0, 255]
        else:
            if original_range == [0,255]:
                frame = np.clip(frame, 0, 255)
            elif original_range == [0,1]:
                frame = np.clip(frame, 0, 1)

        ### Convert Dtype if Needed: ###
        if original_dtype != dtype_requested:  # Check if dtype conversion is needed
            frame = frame.astype(dtype_requested)  # Convert dtype

        processed_frames.append(frame)  # Add the processed frame to the list

    return processed_frames  # Return the list of processed frames

def process_image_dictionary(input_dict):
    """
    Process a dictionary of images by keeping only valid images and using frames_to_constant_format on them.

    Args:
        input_dict (dict): Dictionary containing images and other data.

    Returns:
        dict: Dictionary containing only the processed images.
    """

    ### Initialize Output Dictionary: ###
    output_dict = {}  # Initialize an empty dictionary to store the output

    ### Loop Through Dictionary Items: ###
    for key, value in input_dict.items():  # Iterate through each key-value pair in the input dictionary
        if isinstance(value, np.ndarray) and (value.ndim == 2 or (value.ndim == 3 and value.shape[2] in [1, 3])):  # Check if the value is an image
            output_dict[key] = frames_to_constant_format([value])  # Process the image and add it to the output dictionary
        elif isinstance(value, list):  # Check if the value is a list
            valid_images = [item for item in value if isinstance(item, np.ndarray) and (item.ndim == 2 or (item.ndim == 3 and item.shape[2] in [1, 3]))]  # Filter valid images
            if len(valid_images) == len(value):  # Check if all items in the list are valid images
                output_dict[key] = frames_to_constant_format(valid_images)  # Process the list of images and add it to the output dictionary

    return output_dict  # Return the output dictionary containing only the processed images
def RGB2BW(input_image):
    if len(input_image.shape) == 2:
        return input_image

    if len(input_image.shape) == 3:
        if type(input_image) == torch.Tensor and input_image.shape[0] == 3:
            grayscale_image = 0.299 * input_image[0:1, :, :] + 0.587 * input_image[1:2, :, :] + 0.114 * input_image[2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1] + 0.587 * input_image[:, :, 1:2] + 0.114 * input_image[:, :, 2:3]
        else:
            grayscale_image = input_image

    elif len(input_image.shape) == 4:
        if type(input_image) == torch.Tensor and input_image.shape[1] == 3:
            grayscale_image = 0.299 * input_image[:, 0:1, :, :] + 0.587 * input_image[:, 1:2, :, :] + 0.114 * input_image[:, 2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, :, 0:1] + 0.587 * input_image[:, :, :, 1:2] + 0.114 * input_image[:, :, :, 2:3]
        else:
            grayscale_image = input_image

    elif len(input_image.shape) == 5:
        if type(input_image) == torch.Tensor and input_image.shape[2] == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1, :, :] + 0.587 * input_image[:, :, 1:2, :, :] + 0.114 * input_image[:, :, 2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, :, :, 0:1] + 0.587 * input_image[:, :, :, :, 1:2] + 0.114 * input_image[:, :, :, :, 2:3]
        else:
            grayscale_image = input_image

    return grayscale_image

def BW2RGB(input_image):
    ### For Both Torch Tensors and Numpy Arrays!: ###
    # Actually... we're not restricted to RGB....
    if len(input_image.shape) == 2:
        if type(input_image) == torch.Tensor:
            RGB_image = input_image.unsqueeze(0)
            RGB_image = torch.cat([RGB_image,RGB_image,RGB_image], 0)
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
        DICT = {"SCC": {},
                "ECC": {"flag_pre_align_using_SCC": 0},
                "FEATURE_BASED": {},
                "Correlation_Tracker": {},
                "OPTICAL_FLOW": {"flag_use_homography": 0},
                "DENOISE_COT": {"alignment_method": 0, "post_process_method": 0, "homography_method":0},
                "RV_CLASSIC": {},
                "DENOISE_YOV": {},
                "CLAHE": {"number_of_clahe_iterations": 1, "color_space": 0, 'clip_limit':2, 'tile_grid_size':(8,8)},
                "RCC": {},
                "SHARPEN": {},
                "STRETCH": {'q1':0.01, 'q2':0.99},
                "MEAN": {},
                "WINDOWED_MEAN": {"window_number_of_frames": 1000},
                "HOMOMORAHPIC_FILTER": {"d0":30, "rh":2.0, "rl":0.5, "c":1.0},
                "RETINEX_FILTER": {"gain":1, "offset":0, "alpha":125, "beta":46},
                "AUTOMATIC_SEGMENTATION_AND_HOMOGRAPHY": {},
                "DENOISE_WAVELET": {},
                "DENOISE_NLM": {},
                "GAUSSIAN_BLUR": {'sigma': 5},
                "vnlb": {"std_noise": 50},
                "denoise_rvrt": {"sigma": 50},
                }



    def __init__(self, log_function):
        self.log_function = log_function

    def perform_Correlation_Tracker(self, dict_input):
        # AlignClass.align_frames_crops_using_opencv_tracker(frames, initial_bbox_XYWH=)
        ### Get Region From User: ###
        user_input = dict_input['user_input']
        input_method = dict_input['input_method']
        frames = dict_input['frames']
        reference_frame = frames[0]
        H,W = reference_frame.shape[0:2]
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, initial_grid_points, flag_no_input, flag_list = user_input_to_all_input_types(
            user_input,
            input_method=input_method,
            input_shape=(H, W))
        segmentation_mask_tensor = torch.tensor(initial_segmentation_mask).unsqueeze(0)

        ### Track object using OpenCV: ###
        initial_BB_XYWH = BB_convert_notation_XYXY_to_XYWH(initial_BB_XYXY)
        output_dict = AlignClass.align_frames_crops_using_opencv_tracker(frames, initial_BB_XYWH, tracker_type='CSRT')
        aligned_crops_initial_tracking = output_dict['aligned_crops']
        averaged_crop = output_dict['averaged_crop']
        return process_image_dictionary(output_dict)

    def perform_DENOISE_NLM(self, dict_input):
        frames = AlignClass.non_local_means_denoising_list(dict_input['frames'])
        # AlignClass.denoise_images_tv_chambolle()
        # AlignClass.denoise_images_bilateral()
        output_dict = add_prefix_to_dict_keys(dict_input)
        if 'original_frames_to_denoise' not in list(output_dict.keys()):
            output_dict['original_frames_to_denoise'] = dict_input['frames']
        output_dict['frames'] = frames
        return process_image_dictionary(output_dict)

    def perform_DENOISE_WAVELET(self, dict_input):
        frames = AlignClass.denoise_images_wavelet(dict_input['frames'])
        # AlignClass.denoise_images_tv_chambolle()
        # AlignClass.denoise_images_bilateral()
        output_dict = add_prefix_to_dict_keys(dict_input)
        if 'original_frames_to_denoise' not in list(output_dict.keys()):
            output_dict['original_frames_to_denoise'] = dict_input['frames']
        output_dict['frames'] = frames
        return process_image_dictionary(output_dict)

    def perform_GAUSSIAN_BLUR(self, dict_input):
        gaussian_sigma = dict_input['params']['sigma']
        frames = AlignClass.apply_gaussian_blur_to_list(dict_input['frames'], sigmaX=gaussian_sigma)
        output_dict = add_prefix_to_dict_keys(dict_input)
        if 'original_frames_to_denoise' not in list(output_dict.keys()):
            output_dict['original_frames_to_denoise'] = dict_input['frames']
        output_dict['frames'] = frames
        return process_image_dictionary(output_dict)

    def perform_AUTOMATIC_SEGMENTATION_AND_HOMOGRAPHY(self, dict_input):
        output_dict = AlignClass.perform_automatic_SAM_registration(dict_input)
        return process_image_dictionary(output_dict)

    def perform_HOMOMORAHPIC_FILTER(self, dict_input):
        d0 = dict_input['params']['d0']
        rh = dict_input['params']['rh']
        rl = dict_input['params']['rl']
        c = dict_input['params']['c']
        frames = dict_input['frames']
        frames = [BW2RGB(AlignClass.homomorphic_filter(RGB2BW(frames[i])[:,:,0], d0=d0, rh=rh, rl=rl, c=c)) for i in np.arange(len(frames))]
        output_dict = add_prefix_to_dict_keys(dict_input)
        if 'original_frames_to_denoise' not in list(output_dict.keys()):
            output_dict['original_frames_to_denoise'] = dict_input['frames']
        output_dict['frames'] = frames
        return process_image_dictionary(output_dict)

    def perform_RETINEX_FILTER(self, dict_input):
        # sigma_list = dict_input['sigma_list']
        gain = dict_input['params']['gain']
        offset = dict_input['params']['offset']
        alpha = dict_input['params']['alpha']
        beta = dict_input['params']['beta']
        frames = dict_input['frames']
        frames = [AlignClass.Retinex(frames[i], gain=gain, offset=offset, alpha=alpha, beta=beta) for i in np.arange(len(frames))]
        output_dict = add_prefix_to_dict_keys(dict_input)
        if 'original_frames_to_denoise' not in list(output_dict.keys()):
            output_dict['original_frames_to_denoise'] = dict_input['frames']
        output_dict['frames'] = frames
        return process_image_dictionary(output_dict)


    def perform_SCC(self, dict_input):
        self.log_function(f"Performing SCC", "info")
        output_dict = AlignClass.align_and_average_frames_using_SCC(dict_input)
        output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)
        # output_dict = AlignClass.perform_automatic_pipeline(dict_input)
        return process_image_dictionary(output_dict)

    def perform_CLAHE(self, dict_input):
        self.log_function(f"Performing CLAHE", "info")
        # frames = dict_input['frames']
        # frames = [AlignClass.resize_image(frames[i], (608, 809)) for i in np.arange(len(frames))]
        # dict_input['frames'] = frames
        output_dict = add_prefix_to_dict_keys(dict_input)
        if 'original_frames_to_denoise' not in list(output_dict.keys()):
            output_dict['original_frames_to_denoise'] = dict_input['frames']
        dict_input['frames'] = frames_to_constant_format(dict_input['frames'])
        clip_limit = dict_input['params']['clip_limit']
        if type(dict_input['params']['tile_grid_size']) == tuple:
            tile_grid_size = dict_input['params']['tile_grid_size']
        else:
            tile_grid_size = ast.literal_eval(dict_input['params']['tile_grid_size'])
        output_dict = AlignClass.apply_clahe_equalization(dict_input, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        return process_image_dictionary(output_dict)

    def perform_RCC(self, dict_input):
        self.log_function(f"Performing RCC", "info")
        frames = dict_input['frames']
        frames = numpy_to_torch(list_to_numpy(frames)).cuda()
        frames, residual = perform_RCC_on_tensor(frames, lambda_weight=50, number_of_iterations=1)
        frames = numpy_to_list(torch_to_numpy(frames))
        output_dict = {}
        output_dict['frames'] = frames
        output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)
        return process_image_dictionary(output_dict)

    def perform_WINDOWED_MEAN(self, dict_input):
        self.log_function(f"Performing Windowed Mean", "info")
        frames = dict_input['frames']
        window_number_of_frames = dict_input['params']['window_number_of_frames']
        frames = numpy_to_torch(list_to_numpy(frames)).cuda()
        frames = convn_torch(frames, kernel=torch.ones([int(window_number_of_frames)]).cuda()/window_number_of_frames, dim=0)
        frames = numpy_to_list(torch_to_numpy(frames))
        frames = frames_to_constant_format(frames)
        output_dict = add_prefix_to_dict_keys(dict_input)
        if 'original_frames_to_denoise' not in list(output_dict.keys()):
            output_dict['original_frames_to_denoise'] = dict_input['frames']
        output_dict['frames'] = frames
        return process_image_dictionary(output_dict)

    def perform_ECC(self, dict_input):
        self.log_function(f"Performing ECC", "info")
        output_dict = AlignClass.align_and_average_frames_using_ECC(dict_input)
        output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)
        return process_image_dictionary(output_dict)

    def perform_FEATURE_BASED(self, dict_input):
        self.log_function(f"Performing FEATURE_BASED denoise ", "info")
        output_dict = AlignClass.align_and_average_frames_using_FeatureBased(dict_input)
        output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)
        return process_image_dictionary(output_dict)

    def perform_OPTICAL_FLOW(self, dict_input):
        self.log_function(f"Performing OPTICAL_FLOW denoise ", "info")
        output_dict = AlignClass.align_and_average_frames_using_FlowFormer_and_PWC(dict_input)
        output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)
        return process_image_dictionary(output_dict)

    def perform_DENOISE_COT(self, dict_input):
        self.log_function(f"Performing DENOISE_COT ", "info")
        output_dict = AlignClass.align_and_average_frames_using_CoTracker(dict_input)
        if 'original_frames_to_denoise' not in list(output_dict.keys()):
            output_dict['original_frames_to_denoise'] = dict_input['frames']
        output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)
        return process_image_dictionary(output_dict)

    def perform_RV_CLASSIC(self, dict_input):
        self.log_function(f"Performing RV_CLASSIC denoise ", "info")
        return None

    def perform_DENOISE_YOV(self, dict_input):
        self.log_function(f"Performing DENOISE_YOV ", "info")
        output_dict = AlignClass.align_and_average_frames_using_FlowFormer_and_PWC(dict_input)
        output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)
        return process_image_dictionary(output_dict)

    def perform_STRETCH(self, dict_input):
        self.log_function(f"Performing STRETCH ", "info")
        frames = dict_input.get(DictInput.frames)
        q1 = dict_input['params']['q1']
        q2 = dict_input['params']['q2']
        frames = AlignClass.stretch_histogram(frames, q1, q2)
        output_dict = add_prefix_to_dict_keys(dict_input)
        if 'original_frames_to_denoise' not in list(output_dict.keys()):
            output_dict['original_frames_to_denoise'] = dict_input['frames']
        output_dict['frames'] = frames
        return process_image_dictionary(output_dict)

    def perform_SHARPEN(self, dict_input):
        self.log_function(f"Performing SHARPEN ", "info")
        frames = dict_input.get(DictInput.frames)
        frames = AlignClass.apply_sharpening_filter_list(frames)
        output_dict = add_prefix_to_dict_keys(dict_input)
        if 'original_frames_to_denoise' not in list(output_dict.keys()):
            output_dict['original_frames_to_denoise'] = dict_input['frames']
        output_dict['frames'] = frames
        return process_image_dictionary(output_dict)

    def perform_MEAN(self, dict_input):
        self.log_function(f"Performing MEAN", "info")
        frames = dict_input.get(DictInput.frames)
        frames = AlignClass.mean_over_frames(frames)
        output_dict = add_prefix_to_dict_keys(dict_input,'')
        if 'original_frames_to_denoise' not in list(output_dict.keys()):
            output_dict['original_frames_to_denoise'] = dict_input['frames']
        output_dict['frames'] = frames
        return process_image_dictionary(output_dict)


    def perform_denoise_rvrt(self, dict_input):
        frames = dict_input.get(DictInput.frames)
        sigma = dict_input['params']["sigma"]
        frames_result = main_denoise_rvrt_interface(frames, sigma=sigma)
        dict_result = {DictInput.frames: frames_result}
        return process_image_dictionary(dict_result)


    def perform_vnlb(self, dict_input):
        frames = dict_input.get(DictInput.frames)
        std = dict_input['params']["std_noise"]
        frames_result = vnlb_denoise(frames, std)
        dict_result = {DictInput.frames: frames_result}
        return process_image_dictionary(dict_result)


class DeblurOptions:
    class OptionsServices:
        DICT = {"RV_OM": {},
                "Deep_End2End_Deblur_NAFNET": {},
                "Deep_Kernel_Estimation_And_Deblur": {"n_iters": 30},
                "Deep_Kernel_Estimation_To_Wiener": {},
                "Unsupervised_Blind_Deconvolution": {},
                "WIENER_DECONVOLUTION": {'wiener_parameter': [5], 'number_of_iterations': 5, 'number_of_psf_iterations': 5},
                "WIENER_DECONVOLUTION_ALL_OPTIONS": {"scale_factors":(0.9, 1.0, 1.1),
                                                     "rotation_degrees":(-5, 0, 5),
                                                     "default_SNRs_list":(0.1, 0.5, 1, 2, 5, 10, 20, 50),
                                                     "default_balance_list":(0.1, 0.5, 1, 2, 5, 10, 20, 50)},
                "WIENER_DECONVOLUTION_ALL_OPTIONS_COTRACKER": {"scale_factors": (0.25, 0.5),
                                                     "rotation_degrees": (0),
                                                     "default_SNRs_list": (0.1, 1, 5, 10),
                                                     "default_balance_list": (0.1, 1, 5, 10)},
                "DEEP_WIENER_DECONVOLUTION": {},
                "LaKDnet_defocus":{},
                "LaKDnet_deblur":{},
                "UFDP_defocus":{},
                "Align_To_Rectangle":{},
                }
    def __init__(self, log_function):
        self.log_function = log_function

    def perform_RV_OM(self, dict_input):
        from Omer.to_neo.main_deblur import main_deblur_list_of_frames
        self.log_function(f"Performing RV_OM", "info")
        list_of_numpy_frames = dict_input[PARAMETER.DictInput.frames]
        N = len(list_of_numpy_frames)
        N = (N//2)*2
        list_of_numpy_frames = list_of_numpy_frames[0:N]
        numpy_result = main_deblur_list_of_frames(list_of_numpy_frames=list_of_numpy_frames)
        list_of_numpy_result = frames_to_constant_format(numpy_result)
        return {
            DictOutput.frames: list_of_numpy_result
        }

    def perform_Align_To_Rectangle(self, dict_input):
        output_dict = AlignClass.straighten_polygon_to_rectange_dict(dict_input)
        return process_image_dictionary(output_dict)



    def perform_WIENER_DECONVOLUTION_ALL_OPTIONS_COTRACKER(self, dict_input):
        self.log_function(f"Performing Wiener Deconvolution All Options", "info")
        ### Check all the different average blur kernels i have: ###
        # {"scale_factors": (0.9, 1.0, 1.1),
        #  "rotation_degrees": (-5, 0, 5),
        #  "default_SNRs_list": (0.1, 0.5, 1, 2, 5, 10, 20, 50),
        #  "default_balance_list": (0.1, 0.5, 1, 2, 5, 10, 20, 50)}
        params_dict = dict_input['params']
        if type(params_dict['default_SNRs_list']) == str:
            snr_parameter_list = ast.literal_eval(params_dict['default_SNRs_list'])
        else:
            snr_parameter_list = params_dict['default_SNRs_list']
        if type(params_dict['default_balance_list']) == str:
            balance_parameter_list = ast.literal_eval(params_dict['default_balance_list'])
        else:
            balance_parameter_list = params_dict['default_balance_list']
        if type(params_dict['rotation_degrees']) == str:
            rotation_degrees_list = ast.literal_eval(params_dict['rotation_degrees'])
        else:
            rotation_degrees_list = params_dict['rotation_degrees']
        if type(params_dict['scale_factors']) == str:
            scale_factors_list = ast.literal_eval(params_dict['scale_factors'])
        else:
            scale_factors_list = params_dict['scale_factors']

        ### Get all affine transforms strings: ###
        affine_strings_list = []
        if (type(scale_factors_list) is not list) and (type(scale_factors_list) is not tuple):
            scale_factors_list = [scale_factors_list]
        if (type(rotation_degrees_list) is not list) and (type(rotation_degrees_list) is not tuple):
            rotation_degrees_list = [rotation_degrees_list]
        for scale_factor in scale_factors_list:
            for rotation in rotation_degrees_list:
                current_string = 'scale_factor ' + str(scale_factor) + ', rotation ' + str(rotation)
                affine_strings_list.append(current_string)

        ### Get all blur kernels from previous enhancement (regular + thersholded + straight line fit): ###
        previous_results_dict = dict_input['previous_results']
        all_blur_kernels_list = []
        all_blur_kernels_dict = {}
        explantion_prefix_list = ['co_tracker']
        for i, (key, value) in enumerate(previous_results_dict.items()):
            if 'blur_kernels_from_cotracker' == key:
                all_blur_kernels_list.append(value)
                all_blur_kernels_dict[key] = value

        ### Loop over the different blur kernel ESTIMATION TYPES (regular + thresholded + straight line fit): ### #TODO: for now i assume a single image, will immediately expand to a list of images
        total_output_dict = {}
        total_output_dict['frames'] = []
        total_output_dict['string_explanations'] = []
        for i, (key, value) in enumerate(all_blur_kernels_dict.items()):
            ### Get current iteration input_dict: ###
            blur_kernel_type_name = key
            blur_kernel = value[0]
            input_dict = {}
            # input_dict['frames'] = dict_input['frames']
            input_dict['frames'] = dict_input['previous_results']['original_frames_to_denoise'][0]
            input_dict['input_method'] = dict_input['input_method']
            input_dict['user_input'] = dict_input['user_input']
            input_dict['params'] = dict_input['params']
            input_dict['snr_list'] = snr_parameter_list
            input_dict['balance_list'] = balance_parameter_list
            input_dict['blur_kernel_basic_type'] = blur_kernel[:, :, 0]
            blur_kernel_type_string = explantion_prefix_list[i]

            ### Get current iteration all blur kernel affine augmentations: ###
            augmented_psfs, augmentations_strings = WienerDeconvolution.augment_psf(
                input_dict['blur_kernel_basic_type'].astype(float),
                scale_factors=scale_factors_list,
                rotation_degrees=rotation_degrees_list)
            # display_media(np.clip(cv2.resize((augmented_psfs[6].astype(float) * 250), (500, 500)), 0,255).astype(np.uint8))

            ### Loop over all augmentations and get the output of the "all_options", meaning all parameter/methods options: ###
            for i in np.arange(len(augmented_psfs)):
                ### Update input_dict to inculde current blur kernel type with specific affine transform: ###
                current_blur_kernel = augmented_psfs[i]
                input_dict['blur_kernel'] = current_blur_kernel
                current_output_dict = WienerDeconvolution.get_deblurred_image_from_blur_kernel_using_Wiener_all_parameter_options(input_dict)
                blur_kernel_type_plus_affine_string = blur_kernel_type_string + ', ' + affine_strings_list[i]

                ### Update total dict with specific key prefixes: ###
                current_output_dict = add_prefix_to_dict_keys(current_output_dict,
                                                              prefix=blur_kernel_type_plus_affine_string + '_')
                total_output_dict = merge_dicts_with_missing_keys(total_output_dict, current_output_dict)

                ### Update "frames" key in total output dict: ###
                values_ending_with_frames = \
                [value for key, value in current_output_dict.items() if key.endswith("frames")][0]
                values_ending_with_explanations = \
                [[blur_kernel_type_plus_affine_string + ' ' + item for item in value] for key, value in
                 current_output_dict.items() if key.endswith("explanations")][0]
                total_output_dict['string_explanations'].extend(values_ending_with_explanations)
                frames_with_expalanations = add_text_to_images(values_ending_with_frames,
                                                               values_ending_with_explanations, font_size=0.4,
                                                               font_thickness=2)
                total_output_dict['frames'].extend(frames_with_expalanations)

        # ### Single Image: ###
        # output_dict = WienerDeconvolution.get_deblurred_image_from_blur_kernel_using_Wiener_all_parameter_options(dict_input)
        # output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)

        return process_image_dictionary(total_output_dict)


    def perform_WIENER_DECONVOLUTION_ALL_OPTIONS(self, dict_input):
        self.log_function(f"Performing Wiener Deconvolution All Options", "info")
        ### Check all the different average blur kernels i have: ###
        # {"scale_factors": (0.9, 1.0, 1.1),
        #  "rotation_degrees": (-5, 0, 5),
        #  "default_SNRs_list": (0.1, 0.5, 1, 2, 5, 10, 20, 50),
        #  "default_balance_list": (0.1, 0.5, 1, 2, 5, 10, 20, 50)}
        params_dict = dict_input['params']
        if type(params_dict['default_SNRs_list']) == str:
            snr_parameter_list = ast.literal_eval(params_dict['default_SNRs_list'])
        else:
            snr_parameter_list = params_dict['default_SNRs_list']
        if type(params_dict['default_balance_list']) == str:
            balance_parameter_list = ast.literal_eval(params_dict['default_balance_list'])
        else:
            balance_parameter_list = params_dict['default_balance_list']
        if type(params_dict['rotation_degrees']) == str:
            rotation_degrees_list = ast.literal_eval(params_dict['rotation_degrees'])
        else:
            rotation_degrees_list = params_dict['rotation_degrees']
        if type(params_dict['scale_factors']) == str:
            scale_factors_list = ast.literal_eval(params_dict['scale_factors'])
        else:
            scale_factors_list = params_dict['scale_factors']

        ### Get all affine transforms strings: ###
        affine_strings_list = []
        for scale_factor in scale_factors_list:
            for rotation in rotation_degrees_list:
                current_string = 'scale_factor ' + str(scale_factor) + ', rotation ' + str(rotation)
                affine_strings_list.append(current_string)

        ### Get all blur kernels from previous enhancement (regular + thersholded + straight line fit): ###
        previous_results_dict = dict_input['previous_results']
        all_blur_kernels_list = []
        all_blur_kernels_dict = {}
        explantion_prefix_list = ['Regular', 'Thresholded', 'Straight Line']
        for i, (key, value) in enumerate(previous_results_dict.items()):
            if 'average_blur_kernel' in key:
                all_blur_kernels_list.append(value)
                all_blur_kernels_dict[key] = value

        ### Loop over the different blur kernel ESTIMATION TYPES (regular + thresholded + straight line fit): ### #TODO: for now i assume a single image, will immediately expand to a list of images
        total_output_dict = {}
        total_output_dict['frames'] = []
        total_output_dict['string_explanations'] = []
        for i, (key, value) in enumerate(all_blur_kernels_dict.items()):
            ### Get current iteration input_dict: ###
            blur_kernel_type_name = key
            blur_kernel = value[0]
            input_dict = {}
            input_dict['frames'] = dict_input['frames']
            input_dict['input_method'] = dict_input['input_method']
            input_dict['user_input'] = dict_input['user_input']
            input_dict['params'] = dict_input['params']
            input_dict['snr_list'] = snr_parameter_list
            input_dict['balance_list'] = balance_parameter_list
            input_dict['blur_kernel_basic_type'] = blur_kernel[:,:,0]
            blur_kernel_type_string = explantion_prefix_list[i]

            ### Get current iteration all blur kernel affine augmentations: ###
            augmented_psfs, augmentations_strings = WienerDeconvolution.augment_psf(input_dict['blur_kernel_basic_type'].astype(float),
                                                                            scale_factors=scale_factors_list,
                                                                            rotation_degrees=rotation_degrees_list)
            # display_media(np.clip(cv2.resize((augmented_psfs[6].astype(float) * 250), (500, 500)), 0,255).astype(np.uint8))

            ### Loop over all augmentations and get the output of the "all_options", meaning all parameter/methods options: ###
            for i in np.arange(len(augmented_psfs)):
                ### Update input_dict to inculde current blur kernel type with specific affine transform: ###
                current_blur_kernel = augmented_psfs[i]
                input_dict['blur_kernel'] = current_blur_kernel
                current_output_dict = WienerDeconvolution.get_deblurred_image_from_blur_kernel_using_Wiener_all_parameter_options(input_dict)
                blur_kernel_type_plus_affine_string = blur_kernel_type_string + ', ' + affine_strings_list[i]

                ### Update total dict with specific key prefixes: ###
                current_output_dict = add_prefix_to_dict_keys(current_output_dict, prefix=blur_kernel_type_plus_affine_string+'_')
                total_output_dict = merge_dicts_with_missing_keys(total_output_dict, current_output_dict)

                ### Update "frames" key in total output dict: ###
                values_ending_with_frames = [value for key, value in current_output_dict.items() if key.endswith("frames")][0]
                values_ending_with_explanations = [ [blur_kernel_type_plus_affine_string + ' ' + item for item in value] for key, value in current_output_dict.items() if key.endswith("explanations")][0]
                total_output_dict['string_explanations'].extend(values_ending_with_explanations)
                frames_with_expalanations = add_text_to_images(values_ending_with_frames, values_ending_with_explanations, font_size=0.4, font_thickness=2)
                total_output_dict['frames'].extend(frames_with_expalanations)

        # ### Single Image: ###
        # output_dict = WienerDeconvolution.get_deblurred_image_from_blur_kernel_using_Wiener_all_parameter_options(dict_input)
        # output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)

        return process_image_dictionary(total_output_dict)
    
    def perform_DEEP_WIENER_DECONVOLUTION(self, dict_input):
        self.log_function(f"Performing Deep Wiener Deconvolution", "info")
        return dict_input
    
    def perform_Deep_End2End_Deblur_NAFNET(self, dict_input):
        from ben_deblur.ImageDeBlur.deblur_functions import main_nafnet_deblur
        self.log_function(f"Performing Deep End2End Deblur", "info")
        frames = dict_input[PARAMETER.DictInput.frames]
        frames = main_nafnet_list_of_frames(frames)
        output_dict = {}
        output_dict['frames'] = frames
        output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)
        return process_image_dictionary(output_dict)

    def perform_Deep_Kernel_Estimation_And_Deblur(self, dict_input):
        self.log_function(f"Performing Deep Kernel Estimation And Deblur", "info")
        output_dict = WienerDeconvolution.get_blur_kernels_and_deblurred_images_using_NUBKE(dict_input)
        output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)
        return process_image_dictionary(output_dict)
    def perform_Deep_Kernel_Estimation_To_Wiener(self, dict_input):
        self.log_function(f"Performing Deep_Kernel_Estimation_To_Wiener", "info")
        output_dict = WienerDeconvolution.get_blur_kernel_using_NUBKE_and_deblur_using_Wiener_all_options(dict_input)
        output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)
        return process_image_dictionary(output_dict)

    def perform_Unsupervised_Blind_Deconvolution(self, dict_input):
        self.log_function(f"Performing Unsupervised_Blind_Deconvolution", "info")
        output_dict = WienerDeconvolution.get_blur_kernel_and_deblurred_image_unsupervised_wiener(dict_input)
        output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)
        return process_image_dictionary(output_dict)

    def perform_WIENER_DECONVOLUTION(self, dict_input):
        self.log_function(f"Performing Wiener Deconvolution", "info")
        output_dict = WienerDeconvolution.get_deblurred_image_from_blur_kernel_using_Wiener_specific_option(dict_input)
        output_dict = merge_dicts_with_missing_keys(output_dict, dict_input)
        return process_image_dictionary(output_dict)


    def perform_FMA_Net(self, dict_input):
        list_of_images_numpy = dict_input[PARAMETER.DictInput.frames]
        raise NotImplementedError
        result = main_fma_net_interface(list_of_images_numpy)
        return {PARAMETER.DictInput.frames: result}

    def perform_LaKDnet_deblur(self, dict_input):
        list_of_images_numpy = dict_input[PARAMETER.DictInput.frames]
        frames_resized, (H_new, W_new) = AlignClass.crop_tensor_to_multiple_preserving_aspect_ratio(list_of_images_numpy, size_multiple=8, min_size=None, max_size=None)
        result = main_laknet_deblur_infer_numpy(frames_resized)
        return {PARAMETER.DictInput.frames: result}

    def perform_LaKDnet_defocus(self, dict_input):
        list_of_images_numpy = dict_input[PARAMETER.DictInput.frames]
        frames_resized, (H_new, W_new) = AlignClass.crop_tensor_to_multiple_preserving_aspect_ratio(list_of_images_numpy, size_multiple=8, min_size=None, max_size=None)
        result = main_laknet_deblur_infer_numpy(frames_resized)
        return {PARAMETER.DictInput.frames: result}

    def perform_UFDP_defocus(self, dict_input):
        list_of_images_numpy = dict_input[PARAMETER.DictInput.frames]
        frames_resized, (H_new, W_new) = AlignClass.crop_tensor_to_multiple_preserving_aspect_ratio(list_of_images_numpy, size_multiple=8, min_size=None, max_size=None)
        result = main_laknet_deblur_infer_numpy(frames_resized)
        return {PARAMETER.DictInput.frames: result}




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


def demy_log(a,b):
    print(a)

if __name__ == '__main__':


    video_path = r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\videos\Car_Going_Down\scene_0_resized_short_compressed.mp4"
    frames = get_video_frames(video_path)[:3]


    # Example list of image paths
    image_paths = [
        r'C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\Shaback\litle_amount_of_images_resized\00057774.png',
        r'C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\Shaback\litle_amount_of_images_resized\00057775.png',

        # Add more image paths as needed
    ]
    image_list = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in
                  image_paths]  # Read and convert images
    frames = image_list



    dict_input = {DictInput.frames: frames,
                  "flag_use_homography": True}
    # de_blur_class_obj = DeblurOptions(demy_log)
    de_noise_class_obj = DenoiseOptions(demy_log)


    ##### test functions #####


    #### Deblur
    ## Omer deblur ###
    # dict_res = de_blur_class_obj.perform_RV_OM(dict_input=dict_input)

    ### Ben NAFNET Deblur
    # dict_res = de_blur_class_obj.perform_Deep_End2End_Deblur_NAFNET(dict_input=dict_input)

    ### RealbasicVSR TODO

    ### perform_MEAN
    # dict_res = de_blur_class_obj.perform_MEAN(dict_input)

    ### perform_MEAN
    # dict_res = de_blur_class_obj.perform_SHARPEN(dict_input)

    ### perform_MEAN
    # dict_res = de_blur_class_obj.perform_STRETCH(dict_input)


    ### Winner  ###
    #nubke
    # dict_res = de_blur_class_obj.perform_NUBKE(dict_input)

    # #nubke2win
    # FixMe
    # dict_res = de_blur_class_obj.perform_Deep_Kernel_Estimation_To_Wiener(dict_input)
    #


    ### Denoise  ####
    ### AlignClass ####


    result = de_noise_class_obj.perform_denoise_rvrt(dict_input)
    print(1)

    # result = de_noise_class_obj.perform_vnlb(dict_input)
    # print(1)

    ### perform_SCC
    # dict_res = AlignClass.align_and_average_frames_using_SCC(dict_input)

    ### perform ecc
    # dict_res = AlignClass.align_and_average_frames_using_ECC(dict_input)

    ## feature base
    # dict_res = AlignClass.align_and_average_frames_using_FeatureBased(dict_input)

    # ## optical flow flow formwer
    # dict_res = AlignClass.align_and_average_frames_using_FlowFormer_and_PWC(dict_input)

    ### co tracker
    # dict_res  = AlignClass.align_and_average_frames_using_CoTracker(dict_input)

    #
    # final_res = dict_res[PARAMETER.DictOutput.frames]





    # de_blur_class_obj.perform_FMA_Net(dict_input)
    # de_blur_class_obj.perform_LaKDnet_deblur(dict_input)  # Works but not on the video FRAMES
    # de_blur_class_obj.perform_LaKDnet_defocus(dict_input) # dont works beacuse of the model size
    # de_blur_class_obj.perform_UFDP_defocus(dict_input)  # Works

    