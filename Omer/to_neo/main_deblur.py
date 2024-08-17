import os
import sys
import time
import cv2
import numpy as np
import skvideo
from skvideo import io
import torch



# import os
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'



import PARAMETER
from RVRT_deblur_inference import Deblur
from RapidBase.Utils.IO.imshow_torch_local import torch_to_numpy
from Utils import list_to_numpy, numpy_to_list
# from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import numpy_unsqueeze, BW2RGB, scale_array_to_range
# from RapidBase.import_all import *
from util.crop_image_using_mouse import ImageCropper
from util.save_video_as_mp4 import save_video_as_mp4
DEVICE = 0

###### Rapid base

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


def scale_array_to_range(input_tensor, min_max_values_to_scale_to=(0,1)):
    input_tensor_normalized = (input_tensor - input_tensor.min()) / (input_tensor.max()-input_tensor.min() + 1e-16) * (min_max_values_to_scale_to[1]-min_max_values_to_scale_to[0]) + min_max_values_to_scale_to[0]
    return input_tensor_normalized

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


def get_video_frames(video_path):
    """
    Extracts frames from a video file.

    Parameters:
    video_path (str): The path to the video file.

    Returns:
    list: A list of frames extracted from the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames



def numpy_unsqueeze(input_tensor, dim=-1):
    return np.expand_dims(input_tensor, dim)

def numpy_array_to_video_ready(input_tensor):
    if len(input_tensor.shape) == 2:
        input_tensor = numpy_unsqueeze(input_tensor, -1)
    elif len(input_tensor.shape) == 3 and (input_tensor.shape[0] == 1 or input_tensor.shape[0] == 3):
        input_tensor = input_tensor.transpose([1, 2, 0])  # [C,H,W]/[1,H,W] -> [H,W,C]
    input_tensor = BW2RGB(input_tensor)

    threshold_at_which_we_say_input_needs_to_be_normalized = 2
    if input_tensor.max() < threshold_at_which_we_say_input_needs_to_be_normalized:
        scale = 255
    else:
        scale = 1

    input_tensor = (input_tensor * scale).clip(0, 255).astype(np.uint8)
    return input_tensor

def numpy_to_torch(input_image, device='cpu', flag_unsqueeze=False):
    #Assumes "Natural" RGB form to there's no BGR->RGB conversion, can also accept BW image!:
    if input_image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        input_image = np.expand_dims(input_image, axis=2) #[H,W]->[H,W,1]
    if input_image.ndim == 3:
        input_image = np.transpose(input_image, (2, 0, 1))  # [H,W,C]->[C,H,W]
    elif input_image.ndim == 4:
        input_image = np.transpose(input_image, (0, 3, 1, 2)) #[T,H,W,C] -> [T,C,H,W]
    input_image = torch.from_numpy(input_image.astype(np.float)).float().to(device) # to float32

    if flag_unsqueeze:
        input_image = input_image.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
    return input_image



def video_torch_array_to_video(input_tensor, video_name='my_movie.mp4', FPS=25.0, flag_stretch=False, output_shape=None):
    ### Initialize Writter: ###
    T, C, H, W = input_tensor.shape
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Be sure to use lower case
    # video_writer = cv2.VideoWriter(video_name, fourcc, FPS, (W, H))

    ### Resize If Wanted: ###
    if output_shape is not None:
        output_tensor = torch.nn.functional.interpolate(input_tensor, size=output_shape)
    else:
        output_tensor = input_tensor

    ### Use Scikit-Video: ###
    output_numpy_array = numpy_array_to_video_ready(output_tensor.permute([0,2,3,1]).clamp(0, 255).cpu().numpy())
    if flag_stretch:
        output_numpy_array = scale_array_to_range(output_numpy_array, (0,255))
    skvideo.io.vwrite(video_name,
                      output_numpy_array,
                      outputdict={'-vcodec': 'libx264', '-b': '300000000'}) #chose the bitrate to be very high so no loss of information

    # for frame_counter in np.arange(T):
    #     current_frame = input_tensor[frame_counter]
    #     current_frame = current_frame.permute([1,2,0]).cpu().numpy()
    #     current_frame = BW2RGB(current_frame)
    #     current_frame = (current_frame * 255).clip(0,255).astype(np.uint8)
    #     video_writer.write(current_frame)
    # video_writer.release()


########

def video_read_video_to_numpy_tensor(input_video_path: str, frame_index_to_start, frame_index_to_end):
    if os.path.isfile(input_video_path) is False:
        raise FileNotFoundError("Failed to convert video to numpy array")
    print("Downloading target sub video to torch... it may take a few moments")

    # return skvideo.io.vread(input_video_path)
    ### Get video stream object: ###
    video_stream = cv2.VideoCapture(input_video_path)
    # video_stream.open()
    all_frames = []
    frame_index = 0
    while video_stream.isOpened():
        flag_frame_available, current_frame = video_stream.read()
        if frame_index < frame_index_to_start:
            frame_index += 1
            continue
        elif frame_index == frame_index_to_end:
            break

        if flag_frame_available:
            all_frames.append(current_frame)
            frame_index += 1
        else:
            break
    video_stream.release()
    # print("\n\n\n\npre stack")
    full_arr = np.stack(all_frames)
    # print("post stack")
    return full_arr


def main_deblur(video_path, use_roi=False, start_read_frame=0, end_read_frame=50, save_videos=True, blur_video_mp4="blur_video.mp4",
                deblur_video_mp4="deblur_video.mp4"):

    video_path = video_path

    ### if need to Read few frames

    numpy_video = video_read_video_to_numpy_tensor(video_path, start_read_frame, end_read_frame)

    ### If should use the crop ROI
    if use_roi is True:
        ### if need to read as a ROI crop
        ###  ROI crop image | Get relevant for ceop video ###
        first_frame = numpy_video[0]
        image_cropper = ImageCropper(first_frame)

        ### Get from user crop image
        first_image_cropped_image = image_cropper.get_tensor_crop()

        ### Set into members
        x1, y1, x2, y2 = image_cropper.get_coordinates_by_ROI_area()

        ### the new video crop acording to the ROI
        cropped_video = numpy_video[:, y1:y2, x1:x2, :]

        numpy_video = cropped_video
    ### Move numpy to torch
    torch_video = numpy_to_torch(numpy_video)

    my_deblur_obj = Deblur(input_torch_video=torch_video)
    input_torch_vid, output_torch_vid = my_deblur_obj.get_video_torch_deblur_result()


    ### Saving videos
    if save_videos is True:
        print("Save video to ", blur_video_mp4,deblur_video_mp4 )
        input_torch_vid = input_torch_vid[0].cpu()
        video_torch_array_to_video(input_torch_vid, video_name=blur_video_mp4)
        video_torch_array_to_video(output_torch_vid, video_name=deblur_video_mp4)


    return torch_to_numpy(output_torch_vid)


def main_deblur_list_of_frames(list_of_numpy_frames, use_roi=False, start_read_frame=0, end_read_frame=50, save_videos=True, blur_video_mp4="blur_video.mp4",
                deblur_video_mp4="deblur_video.mp4"):
    """List of numpy frames"""

    list_of_numpy_frames = list_of_numpy_frames

    ### if need to Read few frames


    numpy_video = list_to_numpy(list_of_numpy_frames)

    ### If should use the crop ROI
    if use_roi is True:
        ### if need to read as a ROI crop
        ###  ROI crop image | Get relevant for ceop video ###
        first_frame = numpy_video[0]
        image_cropper = ImageCropper(first_frame)

        ### Get from user crop image
        first_image_cropped_image = image_cropper.get_tensor_crop()

        ### Set into members
        x1, y1, x2, y2 = image_cropper.get_coordinates_by_ROI_area()

        ### the new video crop acording to the ROI
        cropped_video = numpy_video[:, y1:y2, x1:x2, :]

        numpy_video = cropped_video
    ### Move numpy to torch
    torch_video = numpy_to_torch(numpy_video)
    print("Running om rvr")
    my_deblur_obj = Deblur(input_torch_video=torch_video, train_device='cuda', test_device='cuda')
    input_torch_vid, output_torch_vid = my_deblur_obj.get_video_torch_deblur_result()


    ### Saving videos
    if save_videos is True:
        print("Save video to ", blur_video_mp4,deblur_video_mp4 )
        input_torch_vid = input_torch_vid[0].cpu()
        video_torch_array_to_video(input_torch_vid, video_name=blur_video_mp4)
        video_torch_array_to_video(output_torch_vid, video_name=deblur_video_mp4)

    return torch_to_numpy(output_torch_vid)


def main_deblur_dict(input_dict):
    list_of_numpy_frames = input_dict[PARAMETER.AlignClassDictInput.frames]
    numpy_result = main_deblur_list_of_frames(list_of_numpy_frames=list_of_numpy_frames)
    list_of_numpy_result = numpy_to_list(numpy_result)
    return list_of_numpy_result

if __name__ == '__main__':
    print(1)
    # main_deblur(video_path=video_path, use_roi=False, save_videos=True)
    video_path = r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\videos\Car_Going_Down\scene_0_resized_short_compressed.mp4"
    frames = get_video_frames(video_path)
    main_deblur_list_of_frames(frames)



# ### Reading and convert video ###
# video_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/TSOMET__HERTSOG_ZALMAN_SHNEOR/ch01_00000000212000000.mp4"
# numpy_video = video_read_video_to_numpy_tensor(video_path, 2, 10)
# torch_video = numpy_to_torch(numpy_video)
#
#
# #### Deblur the video ###
# my_deblur_obj = Deblur(input_torch_video=torch_video)
# input_torch_vid, output_torch_vid = my_deblur_obj.get_video_torch_deblur_result()
#
#
# ### Display the video ###
# # plt.imshow(input_torch_vid[0][0].permute(1, 2, 0).cpu())
# # plt.imshow(output_torch_vid[0].permute(1, 2, 0).cpu())
#
# imshow_torch_video(input_torch_vid, video_title="input")
# time.sleep(3)
# imshow_torch_video(output_torch_vid, video_title="tweets_per_accounts")
#
#
# ### Dudy
# # final_tensor = torch.cat([input_torch_vid.cpu().squeeze(0), output_torch_vid], -1)
# # imshow_torch_video(final_tensor, FPS=1)
# # bla = RGB2BW(output_torch_vid.cpu().squeeze(0)-output_torch_vid)
# # imshow_torch_video(bla, FPS=1)
#
# # plt.show()



