import os
import gc
import cv2
import numpy as np
import torch
import random
from RapidBase.import_all import *

def Alpha_Composite_Images_In_Known_Place_And_Size(input_image_RGB, drone_image_RGBA, counter, H_size_range=(70, 70), placement=(320,320) ):
    ### Get Final Size: ###
    if len(input_image_RGB.shape) == 2:
        input_image_RGB = np.expand_dims(input_image_RGB, axis=-1)
    H_image, W_image, C_image = input_image_RGB.shape
    # flag_input_image_BW = C_image == 1
    H_drone_initial, W_drone_initial, C_drone_initial = drone_image_RGBA.shape
    min_scale_factor = H_size_range[0] / H_drone_initial
    max_scale_factor = H_size_range[1] / H_drone_initial
    scale_factor = get_random_number_in_range(min_scale_factor, max_scale_factor)
    # if 'Mavic2' in drone_filename:
    #     H_size_range = (18, 18)
    H_drone_new = H_drone_initial * scale_factor
    W_drone_new = W_drone_initial * scale_factor
    H_drone_new = int(H_drone_new)
    W_drone_new = int(W_drone_new)

    ### rescale drone image: ###
    # tic()
    drone_image_RGBA_torch = numpy_to_torch(drone_image_RGBA).unsqueeze(0)



    if scale_factor <= 1:
        scale_factor_inverse = (1 / scale_factor).item()
        scale_factor_inverse_int = int(round(scale_factor_inverse))
        # gaussian_blur_layer = Gaussian_Blur_Layer(channels=4, kernel_size=7, sigma=scale_factor_inverse, dim=2)
        # drone_image_RGBA_torch_new = gaussian_blur_layer.forward(drone_image_RGBA_torch)
        random_downsample = int(np.round(get_random_number_in_range(0, 1)))
        if random_downsample == 0:
            drone_image_RGBA_torch = fast_binning_2D_PixelBinning(drone_image_RGBA_torch.unsqueeze(0),
                                                                  binning_size=scale_factor_inverse_int,
                                                                  overlap_size=0).squeeze(
                0) / scale_factor_inverse_int ** 2
        else:
            drone_image_RGBA_torch = fast_binning_2D_AvgPool2d(drone_image_RGBA_torch,
                                                               binning_size_tuple=(
                                                                   scale_factor_inverse_int, scale_factor_inverse_int),
                                                               overlap_size_tuple=(0, 0))
    # toc('Binning')
    # (2). Perform Downsampling/Upsampling If Needed:
    # tic()
    downsample_layer = nn.Upsample(size=(H_drone_new, W_drone_new), mode='bilinear')
    drone_image_RGBA_torch_final = downsample_layer.forward(drone_image_RGBA_torch)
    # toc('Upsample')
    # tic()
    drone_image_RGBA_numpy_final = torch_to_numpy(drone_image_RGBA_torch_final[0])
    drone_image_RGB_numpy_final = torch_to_numpy(drone_image_RGBA_numpy_final[:, :, 0:3])
    drone_image_Alpha_numpy_final = torch_to_numpy(drone_image_RGBA_numpy_final[:, :, -1:])
    # imagesc_torch(drone_image_RGBA_torch_final[:, 0:3])
    # toc('torch to numpy again')

    ### Get New, Scaled, Drone Image Shape: ###
    B_drone, C_drone, H_drone, W_drone = drone_image_RGBA_torch_final.shape

    H_start, H_stop = int(placement[0]), int(placement[0] + H_drone)
    # W_start, W_stop = get_random_start_stop_indices_for_crop(W_drone, W_image)
    W_start = int(placement[1])
    W_stop = W_start + W_drone
    H_size = H_stop - H_start
    W_size = W_stop - W_start
    original_image_in_drone_ROI = input_image_RGB[H_start:H_stop, W_start:W_stop, :]
    contrast_where_drone_is = original_image_in_drone_ROI.std()
    drone_contrast = drone_image_RGBA_torch.std()
    # while drone_contrast - contrast_where_drone_is < drone_contrast - 5:
    #     ### Put drone in random spot in image: ###
    #     H_start, H_stop = get_random_start_stop_indices_for_crop(H_drone, H_image)
    #     # W_start, W_stop = get_random_start_stop_indices_for_crop(W_drone, W_image)
    #     W_start = int(get_random_number_in_range(0, W_image - H_size_range[1]))
    #     W_stop = W_start + W_drone
    #     H_size = H_stop - H_start
    #     W_size = W_stop - W_start
    #
    #     ### contrast computation ###
    #     original_image_in_drone_ROI = input_image_RGB[H_start:H_stop, W_start:W_stop, :]
    #     contrast_where_drone_is = original_image_in_drone_ROI.std()
    # toc('calculate contrast')

    # tic()
    BB_tuple = (W_start, H_start, W_stop, H_stop)  # initial, before correcting for COCO dataset
    large_empty_image_for_drone_RGB_numpy = np.zeros_like(BW2RGB(input_image_RGB))
    large_transperancy_mask_where_drone_is_numpy = np.zeros((H_image, W_image))
    # final_blended_image = np.zeros((H_image, W_image, 3))
    # toc('initialize empty matrices')

    ### Build Logical Mask: ###
    # tic()
    large_transperancy_mask_where_drone_is_numpy[H_start:H_stop, W_start:W_stop] = drone_image_RGBA_numpy_final[:, :,
                                                                                   -1] / 255
    large_empty_image_for_drone_RGB_numpy[H_start:H_stop, W_start:W_stop] = BW2RGB(
        RGB2BW(drone_image_RGBA_numpy_final[:, :, 0:3]))


    ### Actually Blend The Images Together: ###
    # (1). Randomize Drone Max Intensity Factor:
    max_intensity_range = (5, 5)
    current_max_intensity = get_random_number_in_range(max_intensity_range[0], max_intensity_range[1])
    large_empty_image_for_drone_RGB_numpy = large_empty_image_for_drone_RGB_numpy * current_max_intensity / large_empty_image_for_drone_RGB_numpy.max()
    final_large_empty_image_for_drone_RGB_numpy = (large_empty_image_for_drone_RGB_numpy).clip(0, 255)

    plus_minus = 1
    # plus_minus = np.random.randint(0, 2)
    ### Perform the blending: ###
    final_drone_image = (numpy_unsqueeze(large_transperancy_mask_where_drone_is_numpy)) * RGB2BW(final_large_empty_image_for_drone_RGB_numpy)
    final_image_blend = RGB2BW((1 - numpy_unsqueeze(large_transperancy_mask_where_drone_is_numpy, -1)) * input_image_RGB)
    if plus_minus == 1:
        # input_image_RGB = BW2RGB(RGB2BW(input_image_RGB))
        final_blended_image = (final_drone_image) + final_image_blend
    # else:
    #     final_blended_image = (-1 * final_drone_image) + final_image_blend
    print(counter)

    save_path = os.path.join("/media/simteam-j/Datasets1/Object_Deteciton_datasets/class3/bgs/test", string_rjust(counter, 5) + ".npy")
    np.save(save_path, final_blended_image)





# RGBA_path = "/media/simteam-j/Datasets1/Object_Deteciton_datasets/class3/Stationary/Var1_81d"
RGBA_path = "/media/simteam-j/Datasets1/Object_Deteciton_datasets/VISDrone_Palantir_ObjectDetection/hold/"
image_path = "/media/simteam-j/Datasets1/Object_Deteciton_datasets/VISDrone_Palantir_ObjectDetection/images_bgs"
RGBA_image_list = os.listdir(RGBA_path)
image_list = os.listdir(RGBA_path)
image_list = os.listdir(image_path)
image_list.sort(), RGBA_image_list.sort()
for i in range(len(image_list)):
    drone_filename = os.path.join(RGBA_path, RGBA_image_list[i])
    image_filname = os.path.join(image_path, image_list[i])
    # input_image = np.transpose(np.load(image_filname), (1, 2, 0))
    input_image = np.load(image_filname)
    current_add_on_image_RGBA = cv2.imread(drone_filename, flags=cv2.IMREAD_UNCHANGED)
    Alpha_Composite_Images_In_Known_Place_And_Size(input_image, current_add_on_image_RGBA, i, placement=(input_image.shape[0] / 2, input_image.shape[1] / 2))

