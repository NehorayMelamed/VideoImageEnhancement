import os
import gc
import cv2
import numpy as np
import torch
import random
from RapidBase.import_all import *

def Alpha_Composite_Images_In_Random_Place(input_image_RGB, add_on_image_RGBA):
    ### Get Input Shapes: ###
    H_image, W_image, C_image = input_image_RGB.shape

    ### rescale drone image: ###
    drone_image_RGBA_torch = numpy_to_torch(add_on_image_RGBA).unsqueeze(0)
    downsample_layer = nn.Upsample(size=(150, 150))
    drone_image_RGBA_torch_final = downsample_layer.forward(drone_image_RGBA_torch)
    drone_image_RGBA_numpy_final = torch_to_numpy(drone_image_RGBA_torch_final[0])
    B_add_on, C_add_on, H_add_on, W_add_on = drone_image_RGBA_torch_final.shape

    ### Put drone in random spot in image: ###
    H_start, H_stop = get_random_start_stop_indices_for_crop(H_add_on, H_image)
    # W_start, W_stop = get_random_start_stop_indices_for_crop(W_add_on, W_image)
    W_start = get_random_number_in_range(3300,4600)
    W_stop = W_start + W_add_on
    large_empty_image_for_drone_RGB_numpy = np.zeros_like(input_image_RGB)
    large_logical_mask_where_drone_is_numpy = np.zeros((H_image, W_image))

    large_logical_mask_where_drone_is_numpy[H_start:H_stop, W_start:W_stop] = drone_image_RGBA_numpy_final[:, :, -1] / 255
    large_empty_image_for_drone_RGB_numpy[H_start:H_stop, W_start:W_stop] = drone_image_RGBA_numpy_final[:, :, 0:3]
    large_logical_mask_where_drone_is_numpy = numpy_unsqueeze(large_logical_mask_where_drone_is_numpy)

    final_blended_image = (large_logical_mask_where_drone_is_numpy) * large_empty_image_for_drone_RGB_numpy + (1 - large_logical_mask_where_drone_is_numpy) * input_image_RGB
    # imshow(final_blended_image / 255)

    ### Add Bounding Box: ###
    final_blended_image_with_BB = draw_bounding_box_with_label_on_image_XYXY(final_blended_image, BB_tuple=(W_start, H_start, W_stop, H_stop), color=(0, 120, 120), flag_draw_on_same_image=False)
    # imshow(final_blended_image_with_BB / 255)

    BB_tuple = (W_start, H_start, W_stop, H_stop)

    return final_blended_image, final_blended_image_with_BB, BB_tuple


def Alpha_Composite_Multiple_AddOn_Images_In_Random_Places_In_Single_Image(input_image_RGB, add_on_image_RGBA_list):
    ### Blend All Images In: ###
    BB_tuple_list = []
    final_blended_image = input_image_RGB
    for i in np.arange(len(add_on_image_RGBA_list)):
        final_blended_image, final_blended_image_with_BB, BB_tuple = Alpha_Composite_Images_In_Random_Place(final_blended_image, add_on_image_RGBA_list[i])
        BB_tuple_list.append(BB_tuple)

    ### Draw All Bounding Boxes: ###
    final_blended_image_with_BB = draw_bounding_boxes_with_labels_on_image_XYXY(final_blended_image, BB_tuple_list, flag_draw_on_same_image=False, color=(255,0,0))

    return final_blended_image, final_blended_image_with_BB, BB_tuple_list


def Alpha_Composite_Images_In_Random_Place_And_Size(input_image_RGB, drone_image_RGBA, H_size_range=(8, 20), flag_to_BW=False,
                                                    flag_output_image_with_BB=False,
                                                    RGB_augmentations_layer=None, geometric_augmentations_layer=None, speckles_filenames_list=None, drone_filename=None):
    ### Get Final Size: ###
    if len(input_image_RGB.shape) == 2:
        input_image_RGB = np.expand_dims(input_image_RGB,axis= -1)
    H_image, W_image, C_image = input_image_RGB.shape
    # flag_input_image_BW = C_image == 1
    H_drone_initial, W_drone_initial, C_drone_initial = drone_image_RGBA.shape
    min_scale_factor = H_size_range[0]/H_drone_initial
    max_scale_factor = H_size_range[1]/H_drone_initial
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
    drone_image_torch_RGB = drone_image_RGBA_torch[:, 0:3]
    # tic()
    # drone_image_torch_RGB = kornia.filters.gaussian_blur2d(drone_image_torch_RGB, (101, 101), (60.5, 60.5))
    # drone_image_torch_Alpha = drone_image_RGBA_torch[:, -1:]
    # toc('numpy to torch')

    ########################################################################################################################
    ### Perform Augmentations: ###
    input_tensor_RGB = drone_image_torch_RGB
    input_tensor_RGBA = drone_image_RGBA_torch
    input_tensor_RGB = input_tensor_RGB.cuda()
    input_tensor_RGBA = input_tensor_RGBA.cuda()
    # gtic()
    # output_RGB = RGB_augmentations_layer(input_tensor_RGB/255)
    output_RGB = input_tensor_RGB/255
    input_tensor_RGBA[:, 0:3] = output_RGB
    ### RGBA (geometric) augmentations: ###
    # gtoc('Initialize Augmentations')
    # gtic()
    # output_tensor_RGBA = geometric_augmentations_layer(input_tensor_RGBA)
    output_tensor_RGBA = input_tensor_RGBA
    output_tensor_RGB = output_tensor_RGBA[:, 0:3]
    # gtoc('perform augmentations')
    # imshow_torch(output_tensor_RGBA)
    # imshow_torch(torch.cat([input_tensor_RGB, output_tensor_RGB], dim=-1))


    # ### Apply random contrast along the image: ###
    # number_of_speckle_images_to_generate = 500
    # speckles_folder_path = r'/media/simteam-j/Datasets1/Object_Deteciton_datasets/VISDrone_Palantir_ObjectDetection/Speckles'
    # for frame_index in np.arange(number_of_speckle_images_to_generate):
    #     print(frame_index)
    #     speckles = create_speckles_of_certain_size_in_pixels(input_tensor_RGB.shape[-2:], speckle_size_in_pixels=500, polarization=0.5)
    #     speckles_torch = torch.tensor(speckles).unsqueeze(0).unsqueeze(0)
    #     current_speckles_full_filepath = os.path.join(speckles_folder_path, string_rjust(frame_index, 4) + '.pt')
    #     torch.save(speckles_torch, current_speckles_full_filepath)

    # tic()image_list
    if np.random.randint(0, 9) > 5:
        random_number = int(get_random_number_in_range(0, len(speckles_filenames_list)))
        current_speckles_full_filepath = speckles_filenames_list[random_number]
        speckles_torch = torch.load(current_speckles_full_filepath)
        speckles_torch = speckles_torch.cuda()
        speckles_torch = speckles_torch.clamp(1, 1.5)
        speckles_torch = speckles_torch - 1
        output_tensor_RGB_with_speckles = output_tensor_RGB * speckles_torch
        drone_image_RGBA_torch[:, 0:3] = output_tensor_RGB_with_speckles
        # imshow_torch(drone_image_RGBA_torch)
    else:
        drone_image_RGBA_torch = output_tensor_RGBA
        # imshow_torch(drone_image_RGBA_torch)
    # toc('speckles')

    # imshow_torch(torch.cat([output_tensor_RGB, output_tensor_RGB_with_speckles], dim=-1))
    #(1). Perform Blurring If Image Is Smaller:
    # tic()
    if scale_factor <= 1:
        scale_factor_inverse = (1/scale_factor).item()
        scale_factor_inverse_int = int(round(scale_factor_inverse))
        # gaussian_blur_layer = Gaussian_Blur_Layer(channels=4, kernel_size=7, sigma=scale_factor_inverse, dim=2)
        # drone_image_RGBA_torch_new = gaussian_blur_layer.forward(drone_image_RGBA_torch)
        random_downsample = int(np.round(get_random_number_in_range(0,1)))
        if random_downsample == 0:
            drone_image_RGBA_torch = fast_binning_2D_PixelBinning(drone_image_RGBA_torch.unsqueeze(0),
                                                                  binning_size=scale_factor_inverse_int,
                                                                  overlap_size=0).squeeze(0) / scale_factor_inverse_int**2
        else:
            drone_image_RGBA_torch = fast_binning_2D_AvgPool2d(drone_image_RGBA_torch,
                                                                  binning_size_tuple=(scale_factor_inverse_int, scale_factor_inverse_int),
                                                                  overlap_size_tuple=(0,0))
    # toc('Binning')
    #(2). Perform Downsampling/Upsampling If Needed:
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

    # ### Correct For Unsqueezing: ###
    # tic()
    # drone_image_RGBA_torch = drone_image_RGBA_torch[0]
    # drone_image_RGBA_torch_final = drone_image_RGBA_torch_final[0]
    H_start, H_stop = get_random_start_stop_indices_for_crop(H_drone, H_image)
    # W_start, W_stop = get_random_start_stop_indices_for_crop(W_drone, W_image)
    W_start = int(get_random_number_in_range(0,W_image - H_size_range[1]))
    W_stop = W_start + W_drone
    H_size = H_stop - H_start
    W_size = W_stop - W_start
    original_image_in_drone_ROI = input_image_RGB[H_start:H_stop, W_start:W_stop, :]
    contrast_where_drone_is = original_image_in_drone_ROI.std() 
    drone_contrast = drone_image_RGBA_torch.std()
    while drone_contrast - contrast_where_drone_is < drone_contrast - 5:
    ### Put drone in random spot in image: ###
        H_start, H_stop = get_random_start_stop_indices_for_crop(H_drone, H_image)
        # W_start, W_stop = get_random_start_stop_indices_for_crop(W_drone, W_image)
        W_start = int(get_random_number_in_range(0,W_image - H_size_range[1]))
        W_stop = W_start + W_drone
        H_size = H_stop - H_start
        W_size = W_stop - W_start

        ### contrast computation ###
        original_image_in_drone_ROI = input_image_RGB[H_start:H_stop, W_start:W_stop, :]
        contrast_where_drone_is = original_image_in_drone_ROI.std()
    # toc('calculate contrast')

    # tic()
    BB_tuple = (W_start, H_start, W_stop, H_stop)  #initial, before correcting for COCO dataset
    large_empty_image_for_drone_RGB_numpy = np.zeros_like(BW2RGB(input_image_RGB))
    large_transperancy_mask_where_drone_is_numpy = np.zeros((H_image, W_image))
    # final_blended_image = np.zeros((H_image, W_image, 3))
    # toc('initialize empty matrices')

    ### Build Logical Mask: ###
    # tic()
    large_transperancy_mask_where_drone_is_numpy[H_start:H_stop, W_start:W_stop] = drone_image_RGBA_numpy_final[:, :, -1] / 255
    if flag_to_BW:
        large_empty_image_for_drone_RGB_numpy[H_start:H_stop, W_start:W_stop] = BW2RGB(RGB2BW(drone_image_RGBA_numpy_final[:, :, 0:3]))
    else:
        large_empty_image_for_drone_RGB_numpy[H_start:H_stop, W_start:W_stop] = drone_image_RGBA_numpy_final[:, :, 0:3]
    large_transperancy_mask_where_drone_is_numpy = numpy_unsqueeze(large_transperancy_mask_where_drone_is_numpy)
    # toc('build transperancy mask')

    ### Actually Blend The Images Together: ###
    #(1). Randomize Drone Max Intensity Factor:
    max_intensity_range = (15, 40)
    current_max_intensity = get_random_number_in_range(max_intensity_range[0], max_intensity_range[1])
    large_empty_image_for_drone_RGB_numpy = large_empty_image_for_drone_RGB_numpy * current_max_intensity/large_empty_image_for_drone_RGB_numpy.max()
    final_large_empty_image_for_drone_RGB_numpy = (large_empty_image_for_drone_RGB_numpy).clip(0, 255)

    # #(2). Decide Whether To Use Hard Blending Instead Of Soft Blending:
    # # random_number = get_random_number_in_range(0, 1)
    # random_number = 1
    # if random_number > 0.5:
    #     large_logical_mask_where_drone_is_numpy = (large_logical_mask_where_drone_is_numpy > 0).astype(float)

    # #(3). Increase Alpha Channel To Increase Contrast At The Expanse Of Realism:  #TODO: this is relevant for real images, not BGS
    # ### linear equation p1(10, 1) p2(15, 0.2), f(x) = 0.1333x + 2.7
    # ### make sure if you input H size in place of the x the result is positive
    # start_factor = -0.13 * H_size + 2.5
    # random_factor_for_transperancy = get_random_number_in_range(start_factor, 1)
    # large_transperancy_mask_where_drone_is_numpy = (large_transperancy_mask_where_drone_is_numpy * random_factor_for_transperancy).clip(0, 1)


    #(4). Perform The Blending:
    # tic()
    ### Randomize whether we're plus or minus in the BGS: ###
    plus_minus = np.random.randint(0, 6)
    ### Perform the blending: ###
    if flag_to_BW:
    # if 0:
        final_drone_image = (large_transperancy_mask_where_drone_is_numpy) * RGB2BW(final_large_empty_image_for_drone_RGB_numpy)
        final_image_blend = RGB2BW((1 - large_transperancy_mask_where_drone_is_numpy) * input_image_RGB)
        if plus_minus == 1:
            input_image_RGB = BW2RGB(RGB2BW(input_image_RGB))
            final_blended_image = (final_drone_image) + final_image_blend
            final_blended_image = (final_drone_image) + final_image_blend
        else:
            final_blended_image = (-1 * final_drone_image) + final_image_blend
    else:
        final_blended_image = (large_transperancy_mask_where_drone_is_numpy) * final_large_empty_image_for_drone_RGB_numpy + \
                              (1 - large_transperancy_mask_where_drone_is_numpy) * (input_image_RGB)
    # imshow(final_blended_image/255)
    # toc('perform blending')

    ### Calculate Drone Contrast (if it's too small get rid of it): ###
    # ###### Way I: ########
    # #(1). Get Contrast Image:
    # original_image_in_drone_ROI = input_image_RGB[H_start:H_stop, W_start:W_stop, :]
    # contrast_image = (drone_image_RGB_numpy_final - original_image_in_drone_ROI) / (original_image_in_drone_ROI)
    # contrast_image = contrast_image.mean(-1)
    # #(2). Get Gradient Of The Alpha Image:
    # vx, vy = np.gradient((drone_image_Alpha_numpy_final>0).astype(float).squeeze())
    # v_total = np.sqrt(vx**2+vy**2)
    # v_total = (v_total>0).astype(float)
    # #(3). Get Image Contrast:
    # image_contrast = contrast_image[v_total.astype(bool)].mean()
    # image_contrast = np.abs(image_contrast)
    # ###### Way I: ########
    # original_image_in_drone_ROI = input_image_RGB[H_start:H_stop, W_start:W_stop, :]
    # mean_drone_value = (drone_image_RGB_numpy_final * drone_image_Alpha_numpy_final/255)[(drone_image_Alpha_numpy_final.squeeze()>0).astype(bool)].mean()
    # mean_BG_value = (original_image_in_drone_ROI * (1-drone_image_Alpha_numpy_final/255))[((1-drone_image_Alpha_numpy_final).squeeze()>0).astype(bool)].mean()
    # image_contrast = np.abs((mean_drone_value - mean_BG_value) / (mean_BG_value + 1e-3))


    ### Add Bounding Box: ###
    # tic()
    if flag_output_image_with_BB:
        final_blended_image_with_BB = draw_bounding_box_with_label_on_image_XYXY(final_blended_image,
                                                                                 BB_tuple=BB_tuple,
                                                                                 color=(255, 255, 255),
                                                                                 flag_draw_on_same_image=False)
    else:
        final_blended_image_with_BB = None
    # imshow(final_blended_image_with_BB/255)
    # toc('draw BB on frame')

    # print('***************************************************')


    ### Write Bounding Box In COCO Format: ###
    BB_tuple_COCO = np.array(BB_tuple).astype(float)
    X_center = (BB_tuple_COCO[2] + BB_tuple_COCO[0])/2
    Y_center = (BB_tuple_COCO[3] + BB_tuple_COCO[1])/2
    X_width = (BB_tuple_COCO[2] - BB_tuple_COCO[0])
    Y_width = (BB_tuple_COCO[3] - BB_tuple_COCO[1])
    X_center_normalized = X_center / W_image
    Y_center_normalized = Y_center / H_image
    X_width_normalized = X_width / W_image
    Y_width_normalized = Y_width / H_image
    BB_tuple_COCO[0] = X_center_normalized
    BB_tuple_COCO[1] = Y_center_normalized
    BB_tuple_COCO[2] = X_width_normalized
    BB_tuple_COCO[3] = Y_width_normalized
    BB_tuple_COCO = tuple(BB_tuple_COCO)

    # ### If Contrast Is Too Low Don't Return It: ###
    # if image_contrast < 0.3:
    #     final_blended_image = input_image_RGB
    #     final_blended_image_with_BB = input_image_RGB
    #     BB_tuple = []

    return final_blended_image, final_blended_image_with_BB, BB_tuple, BB_tuple_COCO


def Alpha_Composite_Multiple_AddOn_Images_In_Random_Places_And_Size_In_Single_Image(input_image_RGB, add_on_image_RGBA_list, H_size_range=(8, 20)):
    ### Blend All Images In: ###
    BB_tuple_list = []
    final_blended_image = input_image_RGB
    for i in np.arange(len(add_on_image_RGBA_list)):
        final_blended_image, final_blended_image_with_BB, BB_tuple = Alpha_Composite_Images_In_Random_Place_And_Size(final_blended_image, add_on_image_RGBA_list[i], H_size_range)
        BB_tuple_list.append(BB_tuple)

    ### Draw All Bounding Boxes: ###
    final_blended_image_with_BB = draw_bounding_boxes_with_labels_on_image_XYXY(final_blended_image, BB_tuple_list, flag_draw_on_same_image=False, color=(255,0,0))

    return final_blended_image, final_blended_image_with_BB, BB_tuple_list


def Alpha_Composite_Multiple_AddOn_Images_From_Folder_In_Random_Places_And_Size_In_Single_Image(input_image_RGB,
                                                                                                alpha_channel_folder_path,
                                                                                                number_of_add_ons_per_image=(0, 5),
                                                                                                H_size_range=(8, 20),
                                                                                                flag_to_BW=False,
                                                                                                add_on_image_RGBA_list=None,
                                                                                                flag_output_image_with_BB=False,
                                                                                                RGB_augmentations_layer=None,
                                                                                                geometric_augmentations_layer=None):

    ### Get Images From Folder: ###
    drone_filenames = path_get_all_filenames_from_folder(alpha_channel_folder_path, True)

    ### Get All Speckles Filenames List From Folder: ###
    speckles_folder_path = r'/media/simteam-j/Datasets1/Object_Deteciton_datasets/VISDrone_Palantir_ObjectDetection/Speckles'
    speckles_filenames_list = path_get_all_filenames_from_folder(speckles_folder_path)

    ### Blend All Images In: ###
    BB_tuple_list = []
    BB_tuple_list_COCO = []
    final_blended_image = input_image_RGB
    if type(number_of_add_ons_per_image) != list and type(number_of_add_ons_per_image) != tuple:
        number_of_add_ons_per_image = (number_of_add_ons_per_image, number_of_add_ons_per_image)
    final_number_of_add_ons_per_images = int(get_random_number_in_range(number_of_add_ons_per_image[0], number_of_add_ons_per_image[1]))
    for i in np.arange(0, final_number_of_add_ons_per_images):  #TODO:
        ### Randomize Image Index To Load: ###
        add_on_image_index = int(get_random_number_in_range(0, len(drone_filenames)))

        ### Load image from filenames list: ###
        drone_filename = drone_filenames[add_on_image_index]
        current_add_on_image_RGBA = cv2.imread(drone_filename, flags=cv2.IMREAD_UNCHANGED)

        final_blended_image, final_blended_image_with_BB, BB_tuple, BB_tuple_COCO = Alpha_Composite_Images_In_Random_Place_And_Size(final_blended_image,
                                                                                                                                     current_add_on_image_RGBA,
                                                                                                                                     H_size_range,
                                                                                                                                     flag_to_BW,
                                                                                                                                    flag_output_image_with_BB=False,
                                                                                                                                    RGB_augmentations_layer=RGB_augmentations_layer,
                                                                                                                                    geometric_augmentations_layer=geometric_augmentations_layer,
                                                                                                                                    speckles_filenames_list=speckles_filenames_list,
                                                                                                                                    drone_filename=drone_filename)
        BB_tuple_list.append(BB_tuple)
        BB_tuple_list_COCO.append(BB_tuple_COCO)

    ### Draw All Bounding Boxes: ###
    if flag_output_image_with_BB:
        final_blended_image_with_BB = draw_bounding_boxes_with_labels_on_image_XYXY(final_blended_image, BB_tuple_list, flag_draw_on_same_image=False, color=(255,0,0), line_thickness=1)
    else:
        final_blended_image_with_BB = None

    return final_blended_image, final_blended_image_with_BB, BB_tuple_list, BB_tuple_list_COCO


def Alpha_Composite_Multiple_AddOn_Images_From_Folders_In_Random_Places_And_Size_In_Single_Image(input_image,
                                                                                                alpha_channel_folders_path,
                                                                                                number_of_add_ons_per_image=(0, 5),
                                                                                                H_size_range=(8, 20),
                                                                                                flag_to_BW=False,
                                                                                                BB_colors=None,
                                                                                                add_on_image_RGBA_list_of_lists=None):
    ##############################################################################################################
    ### RGB augmentations layer: ###
    transform_1 = kornia.augmentation.RandomEqualize(same_on_batch=False, p=0.2, keepdim=True, return_transform=None)
    transform_2 = kornia.augmentation.RandomGamma(gamma=(0.5, 2), same_on_batch=False, p=0.2, keepdim=True,
                                                  return_transform=None)
    transform_3 = kornia.augmentation.RandomSharpness(sharpness=15, same_on_batch=False, p=1.0, keepdim=True,
                                                      return_transform=None)
    transform_4 = kornia.augmentation.RandomHue(hue=(-0.2, 0.2), same_on_batch=False, p=1.0, keepdim=True,
                                                return_transform=None)
    transform_5 = kornia.augmentation.RandomContrast(contrast=(0.5, 1), clip_output=True, same_on_batch=False, p=1.0,
                                                     keepdim=True, return_transform=None)




    RGB_aug = [
        transform_1,
        transform_2,
        transform_3,
        transform_4,
        transform_5,
    ]
    RGB_augmentations_layer = nn.Sequential(*RGB_aug)

    ### Geometric Augmentations Layer: ###
    transform_6 = kornia.augmentation.RandomRotation([-180, 180], same_on_batch=False, align_corners=True, p=1,
                                                     keepdim=True, return_transform=None)
    transform_7 = kornia.augmentation.RandomPerspective(distortion_scale=0.35, same_on_batch=False, align_corners=False,
                                                        p=0.35, keepdim=True, sampling_method='basic',
                                                        return_transform=None)
    transform_8 = kornia.augmentation.RandomThinPlateSpline(scale=0.2, align_corners=False, same_on_batch=False, p=0.65,
                                                            keepdim=False, return_transform=None)
    geometric_augmentations = [
        transform_6,
        transform_7,
        transform_8
    ]
    geometric_augmentations_layer = nn.Sequential(*geometric_augmentations)
    ##############################################################################################################


    ### Get List Of Folders Containing RGBA Images: ###
    alpha_channel_folders_list = list(path_get_folder_names(alpha_channel_folders_path))

    ### Check If Image Is RGB or BW: ###
    flag_BW = input_image.shape[-1] == 1
    # input_image_RGB = BW2RGB(input_image)

    ### Blend All Images In: ###
    BB_tuple_list = []
    final_blended_image = input_image
    BB_tuple_tensor = torch.empty((0,5))
    for class_index in np.arange(len(alpha_channel_folders_list)):
        current_alpha_channel_folder_path = alpha_channel_folders_list[class_index]
        final_blended_image, final_blended_image_with_BB, BB_tuple, BB_tuple_COCO = \
            Alpha_Composite_Multiple_AddOn_Images_From_Folder_In_Random_Places_And_Size_In_Single_Image(final_blended_image,
                                                                                                        current_alpha_channel_folder_path,
                                                                                                        number_of_add_ons_per_image,
                                                                                                        H_size_range,
                                                                                                        flag_to_BW,
                                                                                                        None,
                                                                                                        flag_output_image_with_BB=False,
                                                                                                        RGB_augmentations_layer=RGB_augmentations_layer,
                                                                                                        geometric_augmentations_layer=geometric_augmentations_layer)
        current_BB_tuple_tensor = torch.tensor(BB_tuple_COCO)
        N_BB = current_BB_tuple_tensor.shape[0]
        class_vec_torch = torch.ones(N_BB, 1) * 0 #TODO change this line to multiply by zero casue all drones are 1 class at the end
        current_BB_tuple_tensor = torch.cat([class_vec_torch, current_BB_tuple_tensor], -1)
        try:
            BB_tuple_tensor = torch.cat([BB_tuple_tensor, current_BB_tuple_tensor], 0)
        except:
            1  #TODO: maybe make sure this still works with zero add ons
        BB_tuple_list.append(BB_tuple)

    ### Draw All Bounding Boxes: ###
    if BB_colors is None:
        colors_list = get_n_colors(len(alpha_channel_folders_list))
    else:
        colors_list = BB_colors
    final_blended_image_with_BB = copy.deepcopy(final_blended_image)
    for class_index in np.arange(len(alpha_channel_folders_list)):
        current_BB_color = colors_list[class_index]
        current_class_BB_tuple_list = BB_tuple_list[class_index]
        final_blended_image_with_BB = draw_bounding_boxes_with_labels_on_image_XYXY(final_blended_image_with_BB,
                                                                                    current_class_BB_tuple_list,
                                                                                    flag_draw_on_same_image=True,
                                                                                    color=current_BB_color)

    ### Unify All Bounding Boxes From All Classes To List: ###
    final_BB_tuple_list = BB_tuple_tensor.tolist()
    return final_blended_image, final_blended_image_with_BB, final_BB_tuple_list


def Alpha_Composite_Multiple_AddOn_Images_From_Folders_In_Random_Places_And_Size_On_Multiple_Images(input_images_folder,
                                                                                                    alpha_channel_folders_path,
                                                                                                    output_data_folder,
                                                                                                    number_of_add_ons_per_image=(0, 5),
                                                                                                    H_size_range=(8, 20),
                                                                                                    flag_to_BW=False,
                                                                                                    number_of_images=10):
    ### Get Input Images Filenames: ###
    input_images_filenames_list = get_filenames_from_folder(input_images_folder)
    alpha_channel_folders_list = path_get_folder_names(alpha_channel_folders_path)
    colors_list = get_n_colors(len(alpha_channel_folders_path))

    # ### Pre-Fetch RGBA Drone Images: ### #TODO: add possibility of on-the-fly loading
    # add_on_image_RGBA_list_of_lists = create_empty_list_of_lists(len(alpha_channel_folders_list))
    # for alpha_channel_folder_index in np.arange(len(alpha_channel_folders_list)):
    #     ### Read Current Folder Filenames: ###
    #     current_alpha_channel_folder_path = alpha_channel_folders_list[alpha_channel_folder_index]
    #     drone_filenames = path_get_all_filenames_from_folder(current_alpha_channel_folder_path, True)
    #     ### Read Current Folder Images: ###


    #     add_on_image_RGBA_list = []
    #     for i in np.arange(len(drone_filenames)):
    #         add_on_image_RGBA_list_of_lists[alpha_channel_folder_index].append(cv2.imread(drone_filenames[i], flags=cv2.IMREAD_UNCHANGED))
    ### Changed it to None as it is not used to save on RAM: ###   #TODO: delete after everything works!!!!
    add_on_image_RGBA_list_of_lists = None

    ### Loop Over Input Images: ###
    for frame_index in np.arange(3283, number_of_images):
        print(frame_index)
        frame_index_iterator = random.randint(0, len(input_images_filenames_list)- 1)
        ### Get Current Image: ###
        current_image_filename = input_images_filenames_list[frame_index_iterator]
        current_image_name = os.path.split(current_image_filename)[-1].split(".")[0]
        if current_image_filename.endswith('.npy'):
            current_image = np.load(current_image_filename)
        else:
            current_image = read_image_cv2(current_image_filename)
        

        ### Alpha Blend Drones Into Current Image: ###
        final_blended_image_without_BB, final_blended_image_with_BB, final_BB_tuple_list = \
            Alpha_Composite_Multiple_AddOn_Images_From_Folders_In_Random_Places_And_Size_In_Single_Image(current_image,
                                                                                                         alpha_channel_folders_path,
                                                                                                         number_of_add_ons_per_image,
                                                                                                         H_size_range,
                                                                                                         flag_to_BW,
                                                                                                         colors_list,
                                                                                                         add_on_image_RGBA_list_of_lists)

        ### Save Outputs: ###
        #(1). Get Filenames:
        # images_with_BB_folder = os.path.join(output_data_folder, 'images_with_BB')
        # images_without_BB_folder = os.path.join(output_data_folder, 'images_without_BB')
        BB_coordinates_folder = os.path.join(output_data_folder, 'BB_coordinates_numpy')
        BB_coordinates_txt_folder = os.path.join(output_data_folder, 'BB_coordinates_txt')
        image_as_numpy = os.path.join(output_data_folder, 'numpy_images')
        image_BB_numpy = os.path.join(output_data_folder, 'numpy_images_with_BB')
        # path_make_path_if_none_exists(images_with_BB_folder)
        # path_make_path_if_none_exists(images_without_BB_folder)
        path_make_path_if_none_exists(BB_coordinates_folder)
        path_make_path_if_none_exists(BB_coordinates_txt_folder)
        path_make_path_if_none_exists(image_as_numpy)
        path_make_path_if_none_exists(image_BB_numpy)
        # image_with_BB_filename = string_rjust(frame_index, 6) + '.png'
        # image_without_BB_filename = string_rjust(frame_index, 6) + '.png'
        BB_coordinates_filename = string_rjust(frame_index, 6) + '.npy'
        BB_coordinates_txt_filename = string_rjust(frame_index, 6) + '.txt'
        numpy_file_name = string_rjust(frame_index, 6) + '.npy'
        # final_image_with_BB_path = os.path.join(images_with_BB_folder, image_with_BB_filename)
        # final_image_without_BB_path = os.path.join(images_without_BB_folder, image_without_BB_filename)
        final_BB_coordinates_path = os.path.join(BB_coordinates_folder, BB_coordinates_filename)
        final_BB_coordinates_txt_path = os.path.join(BB_coordinates_txt_folder, BB_coordinates_txt_filename)
        final_numpy_image_path = os.path.join(image_as_numpy, numpy_file_name)
        final_numpy_image_BB_path = os.path.join(image_BB_numpy, numpy_file_name)
        #(2). Actually Save Images:
        # cv2.imwrite(final_image_with_BB_path, cv2.cvtColor(final_blended_image_with_BB.astype(np.uint8), cv2.COLOR_BGR2RGB))
        # cv2.imwrite(final_image_without_BB_path, cv2.cvtColor(final_blended_image_without_BB.astype(np.uint8), cv2.COLOR_BGR2RGB))
        np.save(final_BB_coordinates_path, final_BB_tuple_list, allow_pickle=True)
        np.save(final_numpy_image_path, final_blended_image_without_BB)
        np.save(final_numpy_image_BB_path, final_blended_image_with_BB)


        ### Save BB Information To .txt File As In COCO Format: ###
        log = open(final_BB_coordinates_txt_path, 'a')
        for BB_index, current_BB_tuple in enumerate(final_BB_tuple_list):
            current_BB_tuple[0] = int(current_BB_tuple[0])
            current_string = ''
            for element_index in np.arange(len(current_BB_tuple)):
                current_string += str(current_BB_tuple[element_index]) + ' '
            current_string = current_string + '\n'
            log.write(current_string)
        log.close()

    # del final_blended_image_without_BB, final_blended_image_with_BB, 
    # gc.collect()
        #TODO: detail projects for leon and yodan
        #TODO: send movie to yaron "tracking" for initial POC and tell him a 1 week work-together is agreed for 10K, if all goes well we will continue to a month


### Get Images: ###
# input_images_folder = r'/media/simteam-j/Datasets1/Object_Deteciton_datasets/VISDrone_Palantir_ObjectDetection/images'
# input_images_folder = r'/media/simteam-j/Datasets1/Object_Deteciton_datasets/VISDrone_Palantir_ObjectDetection/images_bgs'
input_images_folder = r'/media/simteam-j/Datasets1/Object_Deteciton_datasets/VISDrone_Palantir_ObjectDetection/images_bgs'
# output_data_folder = r'/media/simteam-j/Datasets1/Object_Deteciton_datasets/holder'
output_data_folder = r'/media/simteam-j/Datasets1/Object_Deteciton_datasets/shoval_dataset'
# alpha_channel_folders_path = r'/media/simteam-j/Datasets1/Object_Deteciton_datasets/VISDrone_Palantir_ObjectDetection/Drones_alpha_channel/'
alpha_channel_folders_path = r'/media/simteam-j/Datasets1/Object_Deteciton_datasets/VISDrone_Palantir_ObjectDetection/Drones'

### Create DataSet: ###
gtic()
Alpha_Composite_Multiple_AddOn_Images_From_Folders_In_Random_Places_And_Size_On_Multiple_Images(input_images_folder,
                                                                                                alpha_channel_folders_path,
                                                                                                output_data_folder,
                                                                                                number_of_add_ons_per_image=(5, 10),
                                                                                                H_size_range=(22, 27),
                                                                                                flag_to_BW=True,
                                                                                                number_of_images=7500)
print('bla')
gtoc()

# ### Get input and drone image: ###
# #(1). input_image
# input_image_RGB = read_image_default()*255
# input_image_RGB = read_image_cv2(r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/sky_image.jpg')
# # input_image_RGB = RGB2BW(input_image_RGB)
# H_image, W_image, C_image = input_image_RGB.shape
#
# ### Add Drones In Random Places: ###
# final_blended_image, final_blended_image_with_BB, BB_tuple_list = \
#     Alpha_Composite_Multiple_AddOn_Images_From_Folders_In_Random_Places_And_Size_In_Single_Image(input_image_RGB,
#                                                                                                 alpha_channel_folders_path=drones_path,
#                                                                                                 number_of_add_ons_per_image=(10,10),
#                                                                                                 H_size_range=(10,30),
#                                                                                                  flag_to_BW=False)
# imshow(final_blended_image_with_BB/255)
# imshow(final_blended_image/255)



