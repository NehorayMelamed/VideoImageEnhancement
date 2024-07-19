from typing import List, Tuple

import torch, cv2, random
from torch import Tensor
import numpy as np
from Ostrich.OstrichRun.Algorithm.utils.Smoothing import simple_linear_expanded_smoothing
from Ostrich.OstrichRun.Algorithm.TrajectoryCentering.CSRT_initialization_methods import scale_array_to_range
from Ostrich.OstrichRun.Algorithm.utils.Smoothing import csaps_smoothn_torch
from Ostrich.OstrichRun.Algorithm.utils.tensor_operations import torch_to_numpy


def init_trackers(frame, bboxes):
    trackers = []
    colors = []
    for bbox in bboxes:
        if bbox[3] == 1 or bbox[2] == 1:
            print("ammended")
            ammended_bbox = tuple([max(a, 2) for a in bbox])
        else:
            ammended_bbox = bbox
        tracker = cv2.TrackerCSRT_create()
        ok = tracker.init(frame, ammended_bbox)
        trackers.append(tracker)
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    return trackers, colors


def update_trackers(trackers, frame):
    # loop over trackers and update
    bboxs=[]
    oks=[]
    for tracker in trackers:
        ok, bbox = tracker.update(frame)
        bboxs.append(bbox)
        oks.append(ok)
    return oks, bboxs

# def tracking_use_mouse_to_select_BBs_of_frame(frame):
#     boxes = []
#     colors = []
#     while True:
#         ### Draw rectangle, 'q' to finish: ###
#         cv2.namedWindow('MultiTracker', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
#         cv2.resizeWindow('MultiTracker', 1500, 700)
#         box = cv2.selectROI("MultiTracker", frame)
#         cv2.resizeWindow('MultiTracker', 1500, 700)
#
#         ### Append box: ###
#         boxes.append(box)
#         colors.append((np.random.randint(64, 255), np.random.randint(64, 255), np.random.randint(64, 255)))
#
#         # print("Press 'n' to select next object, or 'q' to start tracking.")
#         key = cv2.waitKey(0) & 0xFF
#         if key == ord('q'):  # Finish selection
#             break
#     cv2.destroyWindow("MultiTracker")
#     return boxes, colors


def tracking_use_mouse_to_select_BBs_of_frame(image, original_size_HW=None, window_size_HW=(700,1500)):
    """
    Allows drawing bounding boxes on an image with the mouse. Boxes are added with a right-click and the function exits with 'q'.

    Args:
    image (np.array): The image on which to draw.
    original_size_HW (tuple): The original dimensions of the image (height, width).
    window_size_HW (tuple): The size of the window in which the image is displayed (height, width).

    Returns:
    list: List of bounding boxes in the format [(x1, y1, x2, y2), ...] in original image coordinates.
    """
    if original_size_HW is None:
        original_size_HW = image.shape[0:2]

    boxes = []  # List to store bounding box coordinates
    current_box = []  # Temporary list to store the current bounding box
    drawing = False  # Flag to indicate that drawing is in progress
    colors_list = []

    # Resize image for display
    window_size_XY = (window_size_HW[1], window_size_HW[0])
    display_image = cv2.resize(image, window_size_XY)
    scale_x = original_size_HW[1] / window_size_HW[1]
    scale_y = original_size_HW[0] / window_size_HW[0]

    def mouse_event(event, x, y, flags, param):
        nonlocal current_box, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            if not drawing:
                # Start drawing the rectangle
                drawing = True
                current_box = [(int(x * scale_x), int(y * scale_y))]  # Convert to original coordinates
                colors_list.append((np.random.randint(64, 255), np.random.randint(64, 255), np.random.randint(64, 255)))
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # Update the current rectangle
                img_copy = display_image.copy()
                cv2.rectangle(img_copy, (int(current_box[0][0] / scale_x), int(current_box[0][1] / scale_y)), (x, y),
                              colors_list[-1], 2)
                cv2.imshow("MultiTracker", img_copy)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if drawing:
                # Finish the current rectangle
                drawing = False
                # current_box.append((int(x * scale_x), int(y * scale_y)))  # Convert to original coordinates
                current_box = [current_box[0][0], current_box[0][1], int(x*scale_x), int(y*scale_y)]
                boxes.append(current_box)  # Add the box to the list
                cv2.rectangle(display_image, (int(current_box[0] / scale_x), int(current_box[1] / scale_y)),
                              (x, y), colors_list[-1], 2)
                cv2.imshow("MultiTracker", display_image)

    # Set up the window and bind the mouse event function
    cv2.namedWindow("MultiTracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MultiTracker", window_size_HW[1], window_size_HW[0])  # Resize window to the specified size
    cv2.imshow("MultiTracker", display_image)
    cv2.setMouseCallback("MultiTracker", mouse_event)

    # Loop to keep the window open until the user presses 'q'
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    ### Destroy all windows: ###
    cv2.destroyAllWindows()

    return boxes, colors_list, scale_x, scale_y


def tracking_create_opencv_tracker_by_name(tracker_type_string='KCF'):
    if tracker_type_string == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting.create()
    elif tracker_type_string == 'CSRT':
        tracker = cv2.TrackerCSRT.create()
    elif tracker_type_string == 'MIL':
        tracker = cv2.TrackerMIL.create()
    elif tracker_type_string == 'KCF':
        tracker = cv2.TrackerKCF.create()
    elif tracker_type_string == 'MIL':
        tracker = cv2.TrackerMIL.create()
    elif tracker_type_string == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow.create()

    return tracker

from RapidBase.Utils.IO.Imshow_and_Plots import BB_convert_notation_XYXY_to_XYWH
def tracking_test_opencv_trackers_on_input_movie_file(movie_full_filename, tracker_type_name='KCF', opencv_window_size=(700,1500)):
    ### Open the video: ###
    cap = cv2.VideoCapture(movie_full_filename)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    ### Let user draw bounding boxes: ###
    boxes_XYXY, colors, scale_x, scale_y = tracking_use_mouse_to_select_BBs_of_frame(frame.copy(), window_size_HW=opencv_window_size)

    ### Convert from XYXY to XYWH as trackers accept: ###
    boxes_XYWH = BB_convert_notation_XYXY_to_XYWH(boxes_XYXY)

    ### Create MultiTracker object: ###
    multi_tracker_list = []
    for box in boxes_XYWH:
        current_tracker = tracking_create_opencv_tracker_by_name(tracker_type_name)
        current_tracker.init(frame, box)
        multi_tracker_list.append(current_tracker)

    ### Process video and track objects: ###
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ### Update tracker: ###
        boxes_XYWH = []
        for tracker in multi_tracker_list:
            success, box = tracker.update(frame)
            if success:
                boxes_XYWH.append(box)
            else:
                boxes_XYWH.append(None)

        # ### Display frame: ###
        # cv2.imshow('MultiTracker', frame)
        # cv2.resizeWindow('MultiTracker', opencv_window_size[1], opencv_window_size[0])

        ### Draw tracked objects: ###
        for i, newbox in enumerate(boxes_XYWH):
            if newbox is not None:
                # new_box_scaled = [newbox[0]/scale_x, newbox[1]/scale_y, newbox[2]/scale_x, newbox[3]/scale_y]
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

        ### Display frame: ###
        cv2.namedWindow("MultiTracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MultiTracker", opencv_window_size[1], opencv_window_size[0])  # Resize window to the specified size
        cv2.imshow("MultiTracker", frame)

        ### Quit on ESC button: ###
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break

    ### Release windows: ###
    cap.release()
    cv2.destroyAllWindows()

# tracking_test_opencv_trackers_on_input_movie_file(r'/home/eylon/Downloads/video_for_debug.mp4')




def track_using_CSRT_BGS(BGS_TrjMov: Tensor, BB_initial_position, frame_stride=1) -> Tuple[np.ndarray, np.ndarray, int, int, List[int], Tensor]:
    ### Initialize: ###
    BGS_TrjMov = BGS_TrjMov.unsqueeze(1)
    W_start = BB_initial_position[0]
    H_start = BB_initial_position[1]
    BB_W = BB_initial_position[2]
    BB_H = BB_initial_position[3]
    BB_list = [(W_start, H_start, BB_W, BB_H)]

    ### Scale input_tensor to uint8 range to initialize tracker: ###
    input_tensor_scaled = scale_array_to_range(BGS_TrjMov, (0, 255))
    input_tensor_scaled = input_tensor_scaled.type(torch.uint8)
    input_tensor_scaled = torch_to_numpy(input_tensor_scaled)

    ### Initialize Trackers: ###
    first_frame = input_tensor_scaled[0]
    trackers, colors = init_trackers(first_frame, BB_list)

    ### Loop over frames from the video stream: ###
    BB_center_x_list = []
    BB_center_y_list = []
    BB_W_list = []
    BB_H_list = []
    BB_list = []
    input_tensor_after_logical_mask = BGS_TrjMov * 1
    for frame_index in np.arange(BGS_TrjMov.shape[0]):
        # print(frame_index)
        ### Get current frame: ###
        cur_frame = input_tensor_scaled[frame_index].squeeze(-1)

        ### Update Trackers only every frame_stride frames: ###
        if frame_index % frame_stride == 0:
            ### Update trackers with current frame: ###
            try:
                tracker_results = update_trackers(trackers, cur_frame)
            except cv2.error:
                print("CV2 threw an error. Using last trackers results")
                tracker_results = last_tracker_results
            last_tracker_results = tracker_results
            BB_W_start, BB_H_start, BB_W, BB_H = tracker_results[-1][0]
            BB_center_H = BB_H_start + BB_H / 2
            BB_center_W = BB_W_start + BB_W / 2
            BB_W_stop = BB_W_start + BB_W
            BB_H_stop = BB_H_start + BB_H

        ### Update BB center lists: ###
        BB_center_x_list.append(BB_center_W)
        BB_center_y_list.append(BB_center_H)
        BB_W_list.append(BB_W)
        BB_H_list.append(BB_H)
        BB_list.append((tracker_results[-1][0]))

        ### Zero out elements in TrjMov: ###
        current_frame_logical_mask = torch.zeros_like(BGS_TrjMov[0])
        current_frame_logical_mask[:, BB_H_start:BB_H_stop, BB_W_start:BB_W_stop] = 1
        input_tensor_after_logical_mask[frame_index] = input_tensor_after_logical_mask[frame_index] * current_frame_logical_mask

    ### Concat the list to an array: ###
    BB_center_x_list = np.array(BB_center_x_list)
    BB_center_y_list = np.array(BB_center_y_list)
    return BB_center_x_list, BB_center_y_list, BB_W, BB_H, BB_list, input_tensor_after_logical_mask


def zero_out_tensor(BGS_TrjMov: Tensor, smoothed_start_x: Tensor, smoothed_start_y: Tensor, bb_W: int, bb_H: int) -> Tensor:
    smoothed_start_x = torch.round(smoothed_start_x).squeeze().type(torch.int32)
    smoothed_start_y = torch.round(smoothed_start_y).squeeze().type(torch.int32)
    smoothed_end_x = (smoothed_start_x + bb_W).clip(0, BGS_TrjMov.shape[-1]).squeeze().type(torch.int32)
    smoothed_end_y = (smoothed_start_y + bb_H).clip(0, BGS_TrjMov.shape[-2]).squeeze().type(torch.int32)
    input_tensor_after_logical_mask = torch.zeros_like(BGS_TrjMov)
    for i in range(BGS_TrjMov.shape[0]):
        input_tensor_after_logical_mask[i, smoothed_start_y[i]:smoothed_end_y[i], smoothed_start_x[i]:smoothed_end_x[i]] = BGS_TrjMov[i, smoothed_start_y[i]:smoothed_end_y[i], smoothed_start_x[i]:smoothed_end_x[i]]
    return input_tensor_after_logical_mask


def track_using_CSRT_BGS_TS_smoothed(BGS_TrjMov: Tensor, BB_initial_position, frame_stride, csaps_smoothing_param: float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Args:
        BGS_TrjMov: TrjMov
        BB_initial_position: W start, H start, W size, H size. All must be ints
        frame_stride: what nth frame to take
        csaps_smoothing_param: smoothing param for smoothing the CSRT output

    Returns: start_x, start_y, x size (it changes each frame), y size (also changes each frame), the input tensor after logical mask

    """
    ### Initialize: ###
    BGS_TrjMov = BGS_TrjMov.unsqueeze(1)
    W_start = BB_initial_position[0]
    H_start = BB_initial_position[1]
    BB_W = BB_initial_position[2]
    BB_H = BB_initial_position[3]
    BB_list = [(W_start, H_start, BB_W, BB_H)]

    ### Scale input_tensor to uint8 range to initialize tracker: ###
    input_tensor_scaled = scale_array_to_range(BGS_TrjMov, (0, 255))
    input_tensor_scaled = input_tensor_scaled.type(torch.uint8)
    input_tensor_scaled = torch_to_numpy(input_tensor_scaled)

    ### Initialize Trackers: ###
    first_frame = input_tensor_scaled[0]
    trackers, colors = init_trackers(first_frame, BB_list)

    BB_start_x = []
    BB_start_y = []
    BB_W_list = []
    BB_H_list = []
    for frame_index in range(0, BGS_TrjMov.shape[0], frame_stride):
        cur_frame = input_tensor_scaled[frame_index].squeeze(-1)

        try:
            tracker_results = update_trackers(trackers, cur_frame)
            BB_W_start, BB_H_start, BB_W, BB_H = tracker_results[-1][0]
        except cv2.error:
            print("CV2 failed to track. Using last values")
            try:
                _ = BB_W_start # equivalent to BB_W_start, BB_H_start, BB_W, BB_H = (BB_W_start, BB_H_start, BB_W, BB_H)
            except NameError: # first batch
                BB_W_start, BB_H_start, BB_W, BB_H = BB_initial_position

        BB_start_x.append(max(BB_W_start, 0))
        BB_start_y.append(max(BB_H_start, 0))
        BB_W_list.append(BB_W)
        BB_H_list.append(BB_H)

    ### Concat the list to an array: ###
    BB_start_x = Tensor(BB_start_x)
    BB_start_y = Tensor(BB_start_y)
    smoothed_start_x = simple_linear_expanded_smoothing(BB_start_x, BGS_TrjMov.shape[0])
    smoothed_start_y = simple_linear_expanded_smoothing(BB_start_y, BGS_TrjMov.shape[0])


    BB_W = Tensor(BB_W_list)
    BB_H = Tensor(BB_H_list)
    median_bounding_box_width = BB_W.median()
    median_bounding_box_height = BB_H.median()
    BGS_TrjMov = BGS_TrjMov.squeeze(1)
    input_tensor_after_logical_mask = zero_out_tensor(BGS_TrjMov, smoothed_start_x, smoothed_start_y, median_bounding_box_width, median_bounding_box_height)
    return smoothed_start_x, smoothed_start_y, BB_W, BB_H, input_tensor_after_logical_mask




def Cut_And_Align_ROIs_Around_Trajectory_Torch(Movie, Movie_BGS, t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, params):
    ### Get basic parameters: ###
    ROI_allocated_around_suspect = params['ROI_allocated_around_suspect']
    H = Movie_BGS.shape[-2]
    W = Movie_BGS.shape[-1]
    number_of_time_steps = len(t_vec)

    # # TODO: delete and replace with pure pytorch
    # t_vec_numpy = t_vec.cpu().numpy()
    # trajectory_smoothed_polynom_Y_numpy = trajectory_smoothed_polynom_Y.cpu().numpy()
    # trajectory_smoothed_polynom_X_numpy = trajectory_smoothed_polynom_X.cpu().numpy()
    # Movie_BGS_numpy = Movie_BGS.cpu().numpy()
    #
    # ### Allocated large frame grid: ###
    # xM = np.arange(H)
    # yM = np.arange(W)
    # original_grid = (t_vec_numpy, xM, yM)
    #
    # ### Allocate suspect ROI grid: ###
    # xl = np.arange(ROI_allocated_around_suspect) - np.int(ROI_allocated_around_suspect/2)
    # xg = np.array(np.meshgrid(t_vec_numpy, xl, xl)).T  #just like meshgrid(x,y) creates two matrices X,Y --> this create 3 matrices of three dimensions which are then concatated
    # number_of_pixels_per_ROI = ROI_allocated_around_suspect**2
    #
    # ### loop over suspect ROI grid and add the trajectory to each point on the grid: ###
    # for ii in range(ROI_allocated_around_suspect):
    #     for jj in range(ROI_allocated_around_suspect):
    #         xg[ii, :, jj, 1] = xg[ii, :, jj, 1] + trajectory_smoothed_polynom_Y_numpy
    #         xg[ii, :, jj, 2] = xg[ii, :, jj, 2] + trajectory_smoothed_polynom_X_numpy
    # xg = xg.reshape((number_of_pixels_per_ROI * number_of_time_steps, 3))
    #
    # ### "align" ROIs which contain suspect on top of each other by multi-dimensional interpolation): ###
    # #TODO: change to pytorch grid sample interpolation for 3D volumetric data. i'll need to turn xg into a [-1,1] tensor. i think i did it before!!!!
    # TrjMovie_numpy = scipy.interpolate.interpn(original_grid, Movie_BGS_numpy[t_vec_numpy.astype(np.int),0,:,:], xg, method='linear')  #TODO: maybe nearest neighbor???? to avoid problems
    # TrjMovie_numpy = TrjMovie_numpy.reshape([ROI_allocated_around_suspect, number_of_time_steps, ROI_allocated_around_suspect]).transpose([1, 0, 2])
    # TrjMov = torch.Tensor(TrjMovie_numpy).to(trajectory_smoothed_polynom_X.device)

    ### since the interpolation is only spatial (not interpolating between t indices) i can do a 2D interpolation: ###
    # trajectory_smoothed_polynom_X = torch.linspace(251, 251, len(trajectory_smoothed_polynom_X)).to(trajectory_smoothed_polynom_X.device)
    # trajectory_smoothed_polynom_Y = torch.linspace(279, 279, len(trajectory_smoothed_polynom_Y)).to(trajectory_smoothed_polynom_Y.device)
    # ROI_allocated_around_suspect = 51

    ROI_grid = torch.arange(ROI_allocated_around_suspect) - np.int(ROI_allocated_around_suspect/2) # [grid_shape] = [B,H,W,2] = [499, H, W, 2]    /    [input_shape] = [B,C,H,W] = [499, 1, H, W]
    ROI_grid = ROI_grid.to(Movie_BGS.device).type(torch.float)
    ROI_grid_Y, ROI_grid_X = torch.meshgrid(ROI_grid, ROI_grid)
    ROI_grid_X = torch.cat([ROI_grid_X.unsqueeze(0)]*len(trajectory_smoothed_polynom_X), 0).unsqueeze(-1)
    ROI_grid_Y = torch.cat([ROI_grid_Y.unsqueeze(0)]*len(trajectory_smoothed_polynom_Y), 0).unsqueeze(-1)
    ROI_grid_X += trajectory_smoothed_polynom_X.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    ROI_grid_Y += trajectory_smoothed_polynom_Y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    ROI_grid_X_normalized = 2 * ROI_grid_X / max(params.W - 1, 1) - 1
    ROI_grid_Y_normalized = 2 * ROI_grid_Y / max(params.H - 1, 1) - 1
    output_grid = torch.cat([ROI_grid_X_normalized, ROI_grid_Y_normalized], -1)
    # output_grid = torch.cat([ROI_grid_Y, ROI_grid_X], -1)

    ### Get ROI Around Trajectory: ###
    TrjMov = torch.nn.functional.grid_sample(Movie_BGS[t_vec.type(torch.LongTensor)].float(), output_grid, 'bilinear')
    # TrjMov = torch.nn.functional.grid_sample(Movie[t_vec.type(torch.LongTensor)], output_grid, 'bilinear')
    TrjMov = TrjMov.squeeze(1)  #[T,1,H,W] -> [T,H,W]. to be changed later to [T,1,H,W] throughout the entire script
    # imshow_torch_video(TrjMov.unsqueeze(1), FPS=50, frame_stride=5)

    ### Get Trajectory Statistics And Graphs: ###
    cx, cy, cx2_modified, cy2_modified, cx_smooth, cy_smooth, \
    input_tensor_spatial_max_over_time, input_tensor_spatial_mean_over_time, input_tensor_spatial_contrast_over_time, \
    input_Tensor_spatial_max_over_time_fft, input_tensor_spatial_mean_over_time_fft, input_tensor_spatial_contrast_over_time_fft = \
        get_trajectory_statistics_and_graphs(TrjMov.abs(), params, t_vec)

    # ### Correct For Trajectory Drift & Get More Precise TrjMov: ###
    # ROI_grid_X += cx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # ROI_grid_Y += cy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # ROI_grid_X_normalized = 2 * ROI_grid_X / max(params.W - 1, 1) - 1
    # ROI_grid_Y_normalized = 2 * ROI_grid_Y / max(params.H - 1, 1) - 1
    # output_grid = torch.cat([ROI_grid_X_normalized, ROI_grid_Y_normalized], -1)
    # ### Get ROI Around Trajectory: ###
    # TrjMov = torch.nn.functional.grid_sample(Movie_BGS[t_vec.type(torch.LongTensor)].float(), output_grid, 'bilinear')
    # # TrjMov = torch.nn.functional.grid_sample(Movie[t_vec.type(torch.LongTensor)], output_grid, 'bilinear')
    # TrjMov = TrjMov.squeeze(1)  # [T,1,H,W] -> [T,H,W]. to be changed later to [T,1,H,W] throughout the entire script
    # # imshow_torch_video(TrjMov.unsqueeze(1), FPS=50, frame_stride=5)


    # #######################################################################################
    # ### Get Stats Over trajectory_smoothed_polynom_X, center_of_mass_trajectory etc': ###
    # max_velocity = 1
    # max_acceleration = 1 # can be derived numerically or from polynom coefficients (for instance, if there's a parabola peak, meaning sharp turn, it doesn't make sense...on the other hand maybe the RANSAC and polyfit got it wrong)
    # #######################################################################################


    # ### TODO: test FFT of Movie_BGS, Movie, and TrjMov: ###
    # #######################################################################################################
    # # input_tensor = Movie_BGS[t_vec.type(torch.LongTensor)]
    # # input_tensor = Movie_BGS
    # # input_tensor = Movie[t_vec.type(torch.LongTensor)]
    # input_tensor = TrjMov.unsqueeze(1)
    # H_center = None
    # W_center = None
    # area_around_center = 3
    # specific_case_string = 'TrjMovie_BGS_1_pixel_right_constant'
    # # plot_fft_graphs(input_tensor, H_center=None, W_center=None, area_around_center=None, specific_case_string='', params=params)
    #######################################################


    return TrjMov, \
           cx, cy, cx2_modified, cy2_modified, cx_smooth, cy_smooth, \
           input_tensor_spatial_max_over_time, input_tensor_spatial_mean_over_time, input_tensor_spatial_contrast_over_time


def outliers_TXY_to_THW_Full_Tensor(outliers_N3, reference_tensor=None):
    N_3_outliers_THW = torch.cat([outliers_N3[:, 0:1], outliers_N3[:, 2:3], outliers_N3[:, 1:2]], -1)
    Movie_outliers_QuantileFiltered_LogicalMap_Reduced = indices_to_logical_mask_torch(input_logical_mask=None,
                                                                                       input_tensor_reference=reference_tensor.squeeze( 1),
                                                                                       indices=N_3_outliers_THW.cuda().long())
    Movie_outliers_QuantileFiltered_LogicalMap_Reduced = Movie_outliers_QuantileFiltered_LogicalMap_Reduced.unsqueeze(1).float()
    return Movie_outliers_QuantileFiltered_LogicalMap_Reduced


def bgs_running_mean_outliers(bgs, bg_std, params):
    temporal_mean_kernel = torch.ones(params.BGS_running_mean_size).to(bgs.device) / params.BGS_running_mean_size
    pre_conv = LogPoint("pre first conv")
    Movie_BGS_running_mean = convn_torch(bgs, temporal_mean_kernel, 0)
    pre_conv.end()
    log_point(params.log_file, pre_conv)
    ratio_outlier_threshold = params.difference_to_STD_ratio_threshold / np.sqrt(
        params.BGS_running_mean_size)  # since we're averaging frames i can lower the threshold
    Movie_BGS_to_STD_ratio = Movie_BGS_running_mean / (bg_std + 1e-3)
    Movie_outliers = (Movie_BGS_to_STD_ratio.abs() > ratio_outlier_threshold).type(torch.float)
    return Movie_outliers

def bgs_no_running_mean(bgs, bg_std, params):
    ratio_outlier_threshold = params.difference_to_STD_ratio_threshold
    Movie_BGS_to_STD_ratio = bgs / (bg_std + 1e-3)
    Movie_outliers = (Movie_BGS_to_STD_ratio.abs() > ratio_outlier_threshold).type(torch.float)
    return Movie_outliers




def canny_edge_detection(Movie_outliers_QuantileFiltered_LogicalMap, background):
    thresholded_binary_BG = kornia.filters.canny(BW2RGB(background))[1]  # Get edge map of BG itself
    # Substract BG edges From Outliers
    Movie_outliers_QuantileFiltered_LogicalMap = Movie_outliers_QuantileFiltered_LogicalMap * (
                1 - thresholded_binary_BG)
    return Movie_outliers_QuantileFiltered_LogicalMap


def find_significant_outliers(bgs: Tensor, background: Tensor, bg_std: Tensor,
                              max_difference: Tuple[int, int], logical_mask, params) -> \
                                Tuple[Tensor, Tensor, Tensor]:
    entire_outl = LogPoint("Entire 120 batch")


    bgs_running_mean_point = LogPoint("BGS RUNNING MEAN")
    if params.perform_BGS_running_mean:
        Movie_outliers = bgs_running_mean_outliers(bgs, bg_std, params)
    else:
        Movie_outliers = bgs_no_running_mean(bgs, bg_std, params)
    bgs_running_mean_point.end()
    log_point(params.log_file, bgs_running_mean_point)

    # Record current movie Outliers sums to be used for next batch
    #    params.current_initial_outlier_sums = Movie_outliers.sum(0, True)
    sum_point = LogPoint("sum point")
    current_initial_outlier_sums = Movie_outliers.sum(0, True).squeeze()
    sum_point.end()
    log_point(params.log_file, sum_point)

    conv_point = LogPoint("convn")
    # Decide whether to perform outlier running mean
    if params.perform_outlier_running_mean:
        outlier_temporal_mean_kernel = torch.ones(params.outlier_detection_running_mean_size).to(bgs.device)
        Movie_outliers_QuantileFiltered_LogicalMap = convn_torch(Movie_outliers, outlier_temporal_mean_kernel, 0) >= params.outlier_after_temporal_mean_threshold
    else:
        Movie_outliers_QuantileFiltered_LogicalMap = Movie_outliers
    conv_point.end()
    log_point(params.log_file, conv_point)

    canny_point = LogPoint("Canny")
    # Perform Canny Edge Detection On Median Image
    if params.use_canny_edge_detection:
        Movie_outliers_QuantileFiltered_LogicalMap = canny_edge_detection(Movie_outliers_QuantileFiltered_LogicalMap, background)
    canny_point.end()
    log_point(params.log_file, canny_point)

    del Movie_outliers

    # Filter Out Points From Wherever It The User Says So
    Movie_outliers_QuantileFiltered_LogicalMap = Movie_outliers_QuantileFiltered_LogicalMap * logical_mask

    # Edges tend to be non-continuous
    max_H, max_W = max_difference
    Movie_outliers_QuantileFiltered_LogicalMap[..., :max_H, :] = 0
    Movie_outliers_QuantileFiltered_LogicalMap[..., Movie_outliers_QuantileFiltered_LogicalMap.shape[-2]-max_H:, :] = 0
    Movie_outliers_QuantileFiltered_LogicalMap[..., :, :max_W] = 0
    Movie_outliers_QuantileFiltered_LogicalMap[..., :, Movie_outliers_QuantileFiltered_LogicalMap.shape[-1]-max_W:] = 0


    # ### Perform Dilation Before CCL Using Iterations: ###
    # #TODO: probably do this using binning and thresholding and it will be quicker


    gtic()
    Movie_outliers_QuantileFiltered_LogicalMap = Movie_outliers_QuantileFiltered_LogicalMap.float()
    p_dilation = LogPoint("entire dilation")
    Movie_outliers_QuantileFiltered_LogicalMap = dilate_using_torch(Movie_outliers_QuantileFiltered_LogicalMap)
    p_dilation.end()
    log_point(params.log_file, p_dilation)
    gtoc('outlier dilation using torch')
    # ## Perform Dilation Before CCL Using Kernel: ###

    ### CCL: ###
    p_outliers = LogPoint("Blob Detection")
    outliers_N3 = blob_detection.blob_detection(Movie_outliers_QuantileFiltered_LogicalMap.squeeze(1))
    p_outliers.end()
    log_point(params.log_file, p_outliers)
    entire_outl.end()
    log_point(params.log_file, entire_outl)
    # Movie_outliers_QuantileFiltered_LogicalMap_Reduced = outliers_TXY_to_THW_Full_Tensor(outliers_N3, reference_tensor=Movie_outliers_QuantileFiltered_LogicalMap)
    return outliers_N3, current_initial_outlier_sums, Movie_outliers_QuantileFiltered_LogicalMap


def outlier_extraction_multiple_bg(frames: Tensor, max_difference: Tuple[int, int], params) -> Tuple[List[Tensor], Tensor, Tensor, Tensor]:
    # returns T,X,Y indices of events
    # Simpler Version: take the median of each sub-sequence as the BG for the sub-sequence and perform outlier detection
    # outlier_sequence_counter_map = torch.zeros(*frames.shape[-2:]).to(frames.device)
    within=LogPoint("Inside Outlier extraction func")
    number_of_sub_sequences = params.NumberOfSubsequences
    T,C,H,W = frames.shape
    number_of_frames_per_sub_sequence = T//number_of_sub_sequences
    movie_current_outliers_TXY_indices = []
    movie_bgs = []
    movie_outliers = []
    ### Calculate BG Stats: ###
    # background, bg_std = calculate_bg_stats(frames[:number_of_frames_per_sub_sequence], params)
    background, bg_std = calculate_bg_stats(frames, params)

    outliers_mask = find_valid_pixels(frames, params.MinTimePresent).to(torch.bool)

    for sub_sequence_index in range(0, number_of_sub_sequences): #WI: make it run once since we no longer update BG or need lists of whatever
        # estimate Outliers for current subsequence
        start_frame_index = sub_sequence_index * number_of_frames_per_sub_sequence
        stop_frame_index = start_frame_index + number_of_frames_per_sub_sequence
        current_frames = frames[start_frame_index:stop_frame_index]   #dudiodo: maybe even add strides here or within the function itself

        current_bgs = current_frames - background

        # get Outliers
        # Outliers is T,H,W, before blob filtering
        movie_current_outliers_TXY_indices_current, initial_outliers_sums, outliers_thw = \
            find_significant_outliers(current_bgs, background, bg_std, max_difference, outliers_mask, params)

        # num_outliers_per_pixel = outliers_thw.sum(dim=0)
        # torch.save(num_outliers_per_pixel, f"outliers_per_pixel_{sub_sequence_index}.pt")
        movie_current_outliers_TXY_indices_current[:, 0] += number_of_frames_per_sub_sequence * sub_sequence_index

        # append to final Outliers
        movie_current_outliers_TXY_indices.append(movie_current_outliers_TXY_indices_current)
        movie_bgs.append(current_bgs)
        movie_outliers.append(outliers_thw)

    catting = LogPoint("pre cat")
    movie_bgs = torch.cat(movie_bgs)
    movie_outliers = torch.cat(movie_outliers)
    catting.end()
    log_point(params.log_file, catting)
    within.end()
    log_point(params.log_file, within)
    return movie_current_outliers_TXY_indices, movie_bgs, movie_outliers, background



def estimate_noise_in_image_EMVA(clean_frame, QE=None, G_DL_to_e_EMVA=None, G_e_to_DL_EMVA=None, readout_noise_electrons=None, full_well_electrons=None, N_bits_EMVA=None, N_bits_final=None):
    ### Get Correct Gain Factors: ###
    G_DL_to_e_final = G_DL_to_e_EMVA * (2 ** N_bits_final / 2 ** N_bits_EMVA)

    ### Get Proper Readout Noise: ###
    readout_noise_DL_final = readout_noise_electrons * G_DL_to_e_final

    ### Get Shot Noise Per Pixel: ###
    photons_per_DL_final = 1 / QE * G_e_to_DL_EMVA * (2 ** N_bits_EMVA / 2 ** N_bits_final)
    photons_per_pixel_final = clean_frame * photons_per_DL_final
    photon_shot_noise_per_pixel_final = torch.sqrt(photons_per_pixel_final)
    electron_shot_noise_per_pixel_final = photon_shot_noise_per_pixel_final * QE
    DL_shot_noise_per_pixel_final = electron_shot_noise_per_pixel_final * G_DL_to_e_final

    ### Get Final Noise Levels Per DL: ###
    total_noise_per_pixel_final = DL_shot_noise_per_pixel_final + readout_noise_DL_final

    return total_noise_per_pixel_final


def set_patch(bg, params):
    replacement_coords = np.load(os.path.join(params.static_drone_coordinate_dir, "rep.npy"))
    replacement = bg[...,replacement_coords[0]:replacement_coords[1], replacement_coords[2]:replacement_coords[3]]
    for i in range(len(os.listdir(params.static_drone_coordinate_dir))-1):
        current_patch = np.load(os.path.join(params.static_drone_coordinate_dir, f"{i}.npy"))
        H = current_patch[1] - current_patch[0]
        W = current_patch[3] - current_patch[2]
        bg[...,current_patch[0]:current_patch[1], current_patch[2]:current_patch[3]] = Tensor(cv2.resize(replacement.squeeze().cpu().numpy(), (W, H))).cuda().unsqueeze(0).unsqueeze(0)
    return bg


def calculate_bg_stats(frames: Tensor, params):

    # initialize First BG As The Median Of The First Batch
    Movie_BG = frames[0::10].median(0)[0].unsqueeze(0).float().to(frames.device)

    if params.correct_for_static_drone:
        Movie_BG = set_patch(Movie_BG, params)
    ### Estimate Noise Using Clean Image and EMVA Report: ###
    Movie_BG_std_torch = estimate_noise_in_image_EMVA(Movie_BG,
                                                      QE=params.QE,
                                                      G_DL_to_e_EMVA=params.G_DL_to_e_EMVA,
                                                      G_e_to_DL_EMVA=params.G_e_to_DL_EMVA,
                                                      readout_noise_electrons=params.readout_noise_electrons,
                                                      full_well_electrons=params.full_well_electrons,
                                                      N_bits_EMVA=params.N_bits_EMVA,
                                                      N_bits_final=params.N_bits_final)

    return Movie_BG, Movie_BG_std_torch


def find_valid_pixels(stabilized_frames: Tensor, minimum_time_present: float) -> Tensor:
    # returns H,W tensor of valid pixels - ie. pixels that are not usually shifted out of existence
    threshold_tensor = (stabilized_frames.squeeze(1) > 0).type(torch.uint8)
    threshold_tensor = cp.asarray(threshold_tensor)
    accumulated_non_zero_vals = threshold_tensor.sum(axis=0)
    mask = accumulated_non_zero_vals > minimum_time_present*stabilized_frames.shape[0]
    mask = torch.as_tensor(mask).cuda()
    return mask

















