



# ### Imports: ###
# from scipy.interpolate import griddata
# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from skimage.transform import resize, warp
# from skimage.measure import ransac
# from skimage.transform import AffineTransform, ProjectiveTransform
# from skimage.feature import match_descriptors, ORB
# from skimage.color import rgb2gray
#
#
# ### Rapid Base Imports: ###
# import matplotlib
# from Rapid_Base.import_all import *
# from Rapid_Base.Anvil._transforms.shift_matrix_subpixel import _shift_matrix_subpixel_fft_batch_with_channels
# import torchvision.transforms as transforms
# from torchvision.models.optical_flow import raft_large
#
# matplotlib.use('TkAgg')
# from SHABAK.VideoImageEnhancement.Tracking.co_tracker.co_tracker import *
# from scipy.optimize import minimize
# import matplotlib.path as mpath

### Imports: ###
from RapidBase.import_all import *
from Utils import *
from RapidBase.Anvil._transforms.shift_matrix_subpixel import _shift_matrix_subpixel_fft_batch_with_channels
from Tracking.co_tracker.co_tracker import *

class AlignClass:


    ### Get Average Reference Given Shifts: ###
    @staticmethod
    def get_avg_reference_given_shifts(video, shifts):
        """
        Get average reference frame given shifts.
        Args:
            video (torch.Tensor): Video tensor.
            shifts (tuple): Tuple of shifts (sx, sy).
        Returns:
            torch.Tensor: Averaged reference frame.
        """

        sx, sy = shifts
        warped = _shift_matrix_subpixel_fft_batch_with_channels(video, -sy.to(video.device), -sx.to(video.device))  # Warp video frames
        return warped.mean(0)  # Return averaged reference frame

    ### Clean One Frame: ###
    @staticmethod
    def align_frames_in_window_towards_index_CCC(estimator_video, video_to_average, index, max_window=None):
        """
        Warp all video frames towards one frame and average the frame.
        Args:
            estimator_video (torch.Tensor): Estimator video tensor.  #for instance -> previous averaged video
            video_to_average (torch.Tensor): Averaged video tensor. #for instnace -> original video
            index (int): Frame index.
            max_window (int, optional): Maximum window size for averaging.
        Returns:
            tuple: Averaged reference frame, shift in x direction, shift in y direction.
        """

        if max_window is None:
            _, sy, sx, _ = align_to_reference_frame_circular_cc(estimator_video.unsqueeze(0), estimator_video[index:index + 1].unsqueeze(0).repeat(1, estimator_video.shape[0], 1, 1, 1))  # Align frames to reference
            avg_ref = AlignClass.get_avg_reference_given_shifts(video_to_average, [sx, sy])  # Get averaged reference frame
            return avg_ref, sx, sy
        if index + max_window < estimator_video.shape[0]:
            _, sy, sx, _ = align_to_reference_frame_circular_cc(estimator_video[index:index + max_window].unsqueeze(0), estimator_video[index:index + 1].unsqueeze(0).repeat(1, max_window, 1, 1, 1))  # Align frames to reference within window
            avg_ref = AlignClass.get_avg_reference_given_shifts(video_to_average[index:index + max_window], [sx, sy])  # Get averaged reference frame within window
            return avg_ref, sx, sy
        else:
            _, sy, sx, _ = align_to_reference_frame_circular_cc(estimator_video[index - max_window:index].unsqueeze(0), estimator_video[index:index + 1].unsqueeze(0).repeat(1, max_window, 1, 1, 1))  # Align frames to reference within window
            avg_ref = AlignClass.get_avg_reference_given_shifts(video_to_average[index - max_window:index], [sx, sy])  # Get averaged reference frame within window
            return avg_ref, sx, sy

    ### Shift to Center: ###
    @staticmethod
    def align_frames_to_center_CCC(video):
        """
        Warp all video frames towards the center frame.
        Args:
            video (torch.Tensor): Video tensor.
        Returns:
            torch.Tensor: Video tensor with frames shifted to center.
        """

        middle_frame = (video.shape[0] // 2 - 1)
        _, sy, sx, _ = align_to_reference_frame_circular_cc(video.unsqueeze(0),
                                                            video[middle_frame:middle_frame + 1].unsqueeze(0).repeat(1, video.shape[0], 1, 1, 1))  # Align frames to center frame
        video = _shift_matrix_subpixel_fft_batch_with_channels(video, -sy, -sx)  # Shift frames

        return video  # Return shifted video

    @staticmethod
    def align_frames_to_reference_CCC_batch(video: torch.Tensor, reference_frame: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Align frames using circular cross-correlation in batches.

        Args:
            video (torch.Tensor): Input video tensor with shape [T, C, H, W].
            reference_frame (torch.Tensor): Reference frame tensor with shape [C, H, W].
            batch_size (int): The size of each batch.

        Returns:
            torch.Tensor: Aligned video tensor with shape [T, C, H, W].
        """

        T, C, H, W = video.shape  # Extract dimensions

        ### Initializing Output Video Tensor ###
        aligned_video = torch.empty_like(video)  # Initialize tensor to store aligned frames

        ### Looping Over Batches: ###
        for start_idx in range(0, T, batch_size):  # Loop through batches
            end_idx = min(start_idx + batch_size, T)  # Calculate end index for current batch
            batch = video[start_idx:end_idx]  # Extract batch from video

            ### Align Frames Using Circular Cross-Correlation: ###
            _, sy, sx, _ = align_to_reference_frame_circular_cc(batch, reference_frame)  # Compute shifts
            aligned_batch = _shift_matrix_subpixel_fft_batch_with_channels(batch, -sy, -sx)  # Apply shifts

            aligned_video[start_idx:end_idx] = aligned_batch  # Store aligned batch in output tensor

        return aligned_video  # Return aligned video tensor

    @staticmethod
    def get_shifts_CCC_batches(video: torch.Tensor, reference_frame: torch.Tensor, batch_size: int) -> (torch.Tensor, torch.Tensor):
        """
        Compute shifts using circular cross-correlation in batches.

        Args:
            video (torch.Tensor): Input video tensor with shape [T, C, H, W].
            reference_frame (torch.Tensor): Reference frame tensor with shape [C, H, W].
            batch_size (int): The size of each batch.

        Returns:
            torch.Tensor: Shifts in y-direction with shape [T, H, W].
            torch.Tensor: Shifts in x-direction with shape [T, H, W].
        """

        T, C, H, W = video.shape  # Extract dimensions

        ### Initializing Shift Tensors ###
        sy = torch.empty(T, device=video.device)  # Initialize tensor to store y-direction shifts
        sx = torch.empty(T, device=video.device)  # Initialize tensor to store x-direction shifts

        ### Looping Over Batches: ###
        for start_idx in range(0, T, batch_size):  # Loop through batches
            end_idx = min(start_idx + batch_size, T)  # Calculate end index for current batch
            batch = video[start_idx:end_idx]  # Extract batch from video

            ### Compute Shifts Using Circular Cross-Correlation: ###
            _, batch_sy, batch_sx, _ = align_to_reference_frame_circular_cc(batch, reference_frame)  # Compute shifts

            sy[start_idx:end_idx] = batch_sy  # Store y-direction shifts
            sx[start_idx:end_idx] = batch_sx  # Store x-direction shifts

        return sy, sx  # Return shifts in y and x directions

    @staticmethod
    def apply_shifts_to_video_batches(video: torch.Tensor, sy: torch.Tensor, sx: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Apply shifts to the video in batches using subpixel FFT-based method.

        Args:
            video (torch.Tensor): Input video tensor with shape [T, C, H, W].
            sy (torch.Tensor): Shifts in y-direction with shape [T, H, W].
            sx (torch.Tensor): Shifts in x-direction with shape [T, H, W].
            batch_size (int): The size of each batch.

        Returns:
            torch.Tensor: Video tensor with frames shifted to align with the reference frame.
        """

        T, C, H, W = video.shape  # Extract dimensions

        ### Initializing Output Video Tensor ###
        aligned_video = torch.empty_like(video)  # Initialize tensor to store aligned frames

        ### Looping Over Batches: ###
        for start_idx in range(0, T, batch_size):  # Loop through batches
            end_idx = min(start_idx + batch_size, T)  # Calculate end index for current batch
            batch = video[start_idx:end_idx]  # Extract batch from video
            batch_sy = sy[start_idx:end_idx]  # Extract corresponding y-shifts
            batch_sx = sx[start_idx:end_idx]  # Extract corresponding x-shifts

            ### Apply Shifts Using Subpixel FFT-Based Method: ###
            aligned_batch = _shift_matrix_subpixel_fft_batch_with_channels(batch, -batch_sy, -batch_sx)  # Apply shifts

            aligned_video[start_idx:end_idx] = aligned_batch  # Store aligned batch in output tensor

        return aligned_video  # Return aligned video tensor

    @staticmethod
    def apply_shifts_to_video_batches_separate_channels(video: torch.Tensor, sy: torch.Tensor, sx: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Apply shifts to the video in batches using subpixel FFT-based method, with separate shifts for each channel.

        Args:
            video (torch.Tensor): Input video tensor with shape [T, C, H, W].
            sy (torch.Tensor): Shifts in y-direction with shape [T, C, H, W].
            sx (torch.Tensor): Shifts in x-direction with shape [T, C, H, W].
            batch_size (int): The size of each batch.

        Returns:
            torch.Tensor: Video tensor with frames shifted to align with the reference frame.
        """

        T, C, H, W = video.shape  # Extract dimensions

        ### Initializing Output Video Tensor ###
        aligned_video = torch.empty_like(video)  # Initialize tensor to store aligned frames

        ### Looping Over Batches: ###
        for start_idx in range(0, T, batch_size):  # Loop through batches
            end_idx = min(start_idx + batch_size, T)  # Calculate end index for current batch
            batch = video[start_idx:end_idx]  # Extract batch from video
            batch_sy = sy[start_idx:end_idx]  # Extract corresponding y-shifts
            batch_sx = sx[start_idx:end_idx]  # Extract corresponding x-shifts

            ### Applying Shifts Channel-Wise: ###
            aligned_batch_channels = []
            for c in range(C):  # Loop through each channel
                channel_batch = batch[:, c:c+1, :, :]  # Extract channel batch with shape [T, 1, H, W]
                channel_sy = batch_sy  # Extract corresponding y-shifts for channel
                channel_sx = batch_sx  # Extract corresponding x-shifts for channel

                ### Apply Shifts Using Subpixel FFT-Based Method: ###
                aligned_channel_batch = _shift_matrix_subpixel_fft_batch_with_channels(channel_batch, -channel_sy, -channel_sx)  # Apply shifts
                aligned_batch_channels.append(aligned_channel_batch)  # Append aligned channel batch to list

            ### Concatenate Aligned Channels: ###
            aligned_batch = torch.cat(aligned_batch_channels, dim=1)  # Concatenate aligned channels along channel dimension
            aligned_video[start_idx:end_idx] = aligned_batch  # Store aligned batch in output tensor

        return aligned_video  # Return aligned video tensor


    @staticmethod
    def align_frames_to_reference_CCC(input_frames, reference_frame=None, align_to='center', device='cuda'):
        """
        Warp all frames towards the reference frame or center/first frame.

        Args:
            input_frames (list or torch.Tensor): List of frames (numpy arrays) or a tensor of frames.
            reference_frame (np.ndarray or torch.Tensor): The reference frame for alignment.
            align_to (str): 'center' to align to the center frame, 'first' to align to the first frame.

        Returns:
            list or torch.Tensor: List of aligned frames or tensor of aligned frames.
        """

        ### Determine Input Type: ###
        is_list = isinstance(input_frames, list)  # Check if input frames are a list of numpy arrays
        is_numpy = isinstance(input_frames, np.ndarray)  # Check if input frames are a numpy array
        if is_list:
            video = numpy_to_torch(list_to_numpy(input_frames)).to(device) # Convert list of numpy arrays to torch tensor
        elif is_numpy:  # Convert to torch tensor if input is numpy
            video = numpy_to_torch(input_frames).to(device)
        else:  # Use the input tensor directly
            video = input_frames.to(device)
        video = video.contiguous()

        ### Determine Reference Frame: ###
        if reference_frame is None:
            if align_to == 'center':  # Align to center frame
                middle_frame_idx = (video.shape[0] // 2 - 1)  # Calculate middle frame index
                reference_frame = video[middle_frame_idx:middle_frame_idx + 1]  # Extract center frame
            elif align_to == 'first':  # Align to first frame
                reference_frame = video[0:1]  # Extract first frame
        else:
            if isinstance(reference_frame, np.ndarray):  # Convert numpy reference frame to torch tensor
                reference_frame = torch.tensor(reference_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Permute dimensions
            else:
                reference_frame = reference_frame.to(device)  # Use provided reference frame

        ### Align Frames Using Circular Cross Correlation: ###
        # _, sy, sx, _ = align_to_reference_frame_circular_cc(video,
        #                                                     reference_frame)  # Compute shifts
        # video = _shift_matrix_subpixel_fft_batch_with_channels(video, -sy, -sx)  # Apply shifts
        # video = align_frames_to_reference_CCC_batch(video, reference_frame, batch_size=15)
        sy, sx = AlignClass.get_shifts_CCC_batches(RGB2BW(video), RGB2BW(reference_frame), batch_size=15)
        torch.cuda.empty_cache()
        video = AlignClass.apply_shifts_to_video_batches_separate_channels(video, sy, sx, batch_size=15)
        torch.cuda.empty_cache()

        ### Return Aligned Video: ###
        if is_list:  # Convert back to numpy if input was numpy
            return numpy_to_list(torch_to_numpy(video)), sy, sx  # Return list of numpy frames
        elif is_numpy:  # Return torch tensor if input was numpy
            return torch_to_numpy(video), sy, sx  # Return torch tensor if input was numpy
        else:
            return video, sy, sx  # Return tensor of frames



    ### Stabilize Video with Averaging: ###
    @staticmethod
    def align_frames_to_center_CCC_iterative_approach(frames: list, max_window=5, max_frames=None, loop=1, device='cuda') -> torch.Tensor:
        """
        Stabilize video frames with averaging from a list of images.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
            max_window (int): Window size for averaging.
            max_frames (int, optional): Number of frames to stabilize.
            loop (int): Number of averaging iterations.

        Returns:
            torch.Tensor: Stabilized video tensor.
        """
        ### Converting frames to tensor ###
        if max_frames is None:
            max_frames = len(frames)
        video = torch.cat([torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).to(device) for frame in frames], dim=0).float()[:max_frames]  # Convert frames to tensor

        ### Perform averaging over temporal windows iteratively ###
        averaged_video = video.clone()  # Clone video tensor for averaging
        for i in range(loop):
            previous_avg_video = averaged_video.clone()  # Clone averaged video tensor
            for j in range(averaged_video.shape[0]):
                averaged_video[j], _, _ = AlignClass.align_frames_in_window_towards_index_CCC(previous_avg_video,
                                                                                   video,
                                                                                   j,
                                                                                   max_window)  # Clean one frame

        ### Take averaged video from above and perform final alignment to center ###
        averaged_video = AlignClass.align_frames_to_center_CCC(averaged_video)  # Shift frames to center

        return averaged_video  # Return stabilized video


    @staticmethod
    def get_bounding_box_from_points(points):
        """
        Calculate the bounding box from a set of points.

        Args:
            points (np.ndarray): Array of points with shape [N, 2].

        Returns:
            tuple: The bounding box coordinates (x0, y0, x1, y1).
        """

        x_coords = points[:, 0]  # Extract x coordinates
        y_coords = points[:, 1]  # Extract y coordinates

        x0 = np.min(x_coords)  # Minimum x coordinate
        y0 = np.min(y_coords)  # Minimum y coordinate
        x1 = np.max(x_coords)  # Maximum x coordinate
        y1 = np.max(y_coords)  # Maximum y coordinate

        return x0, y0, x1, y1  # Return bounding box coordinates

    @staticmethod
    def align_frames_using_predicted_points_per_frame_optical_flow(frames: list,
                                                      reference_frame: np.ndarray,
                                                      predicted_points_locations: np.ndarray,
                                                      method: str = 'optical_flow') -> (list, np.ndarray):
        #TODO: build on this and create an align_frames_to_reference_using_optical_flow_per_frame by having the predicted_points_locations simply be all the original grid
        #      and the predicted_points_locations being the original grid points prdicted locations using optical flow
        """
        Align bounding boxes using different methods and compute the averaged bounding box.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
            reference_frame (np.ndarray): Reference frame with shape [H, W, C].
            predicted_points_locations (np.ndarray): Predicted points locations for each frame. Shape is [T, N, 2].
            method (str): Method to use for alignment. Options are 'interpolation', 'tracker', 'predicted'.

        Returns:
            list: List of aligned frames.
            np.ndarray: Averaged aligned frame.
        """

        ### This Is The Code Block: ###
        aligned_frames = []  # List to store aligned frames
        aligned_frames_crops = []
        H, W = reference_frame.shape[:2]  # Get height and width from reference frame shape

        if method == 'optical_flow':
            ### Optical Flow Interpolation Method: ###
            ref_points = predicted_points_locations[0]  # Reference points are the predicted points of the first frame
            ref_bounding_box = AlignClass.get_bounding_box_from_points(ref_points)
            X0,Y0,X1,Y1 = int(ref_bounding_box[0]), int(ref_bounding_box[1]), int(ref_bounding_box[2]), int(ref_bounding_box[3]),  # Get bounding box coordinates

            ### Looping Over Frames: ###
            for t in range(len(predicted_points_locations)):  # Loop through each set of predicted points
                curr_points = predicted_points_locations[t]  # Get points for current frame

                ### Calculate Optical Flow for Predicted Points ###
                flow_vectors = curr_points - ref_points  # Calculate optical flow vectors from the first frame

                ### Interpolate Optical Flow for the Entire Bounding Box ###
                grid_y, grid_x = np.mgrid[0:H, 0:W]  # Create a meshgrid for interpolation
                interpolated_flow_y_ref = griddata(ref_points, flow_vectors[:, 1], (grid_x, grid_y), method='linear', fill_value=0)  # Interpolate flow in y-direction
                interpolated_flow_x_ref = griddata(ref_points, flow_vectors[:, 0], (grid_x, grid_y), method='linear', fill_value=0)  # Interpolate flow in x-direction
                interpolated_flow_y_curr = griddata(curr_points, flow_vectors[:, 1], (grid_x, grid_y), method='linear', fill_value=0)  # Interpolate flow in y-direction
                interpolated_flow_x_curr = griddata(curr_points, flow_vectors[:, 0], (grid_x, grid_y), method='linear', fill_value=0)  # Interpolate flow in x-direction

                ### Create Maps for Interpolation ###
                map_y_ref = (grid_y + interpolated_flow_y_ref).astype(np.float32)  # Map for y-direction
                map_x_ref = (grid_x + interpolated_flow_x_ref).astype(np.float32)  # Map for x-direction
                map_y_curr = (grid_y + interpolated_flow_y_curr).astype(np.float32)  # Map for y-direction
                map_x_curr = (grid_x + interpolated_flow_x_curr).astype(np.float32)  # Map for x-direction

                ### Apply Optical Flow to Align Frames ###
                aligned_frame_ref = cv2.remap(frames[t], map_x_ref, map_y_ref, interpolation=cv2.INTER_LINEAR)  # Remap current frame to align with the reference frame
                aligned_frame_curr = cv2.remap(frames[t], map_x_curr, map_y_curr, interpolation=cv2.INTER_LINEAR)  # Remap current frame to align with the reference frame

                ### Get Crop: ###
                aligned_frame_ref_crop = aligned_frame_ref[Y0:Y1, X0:X1]  # Crop aligned frame to the initial bounding box

                ### Append Aligned Frame ###
                aligned_frames.append(aligned_frame_ref)
                aligned_frames_crops.append(aligned_frame_ref_crop)

                # ### Debug: ###
                # imshow_np(frames[0], 'original', maximize_window=False)
                # imshow_np(interpolated_flow_x_ref, 'flow on reference', maximize_window=False)
                # imshow_np(frames[t], 'current', maximize_window=False)
                # imshow_np(interpolated_flow_x_curr, 'flow on current', maximize_window=False)
                # imshow_np(aligned_frame_ref, 'aligned frame ref', maximize_window=False)
                # imshow_np(aligned_frame_curr, 'aligned frame curr', maximize_window=False)

                # ### Debug: ###
                # aligned_frames_numpy = list_to_numpy(aligned_frames)
                # aligned_frames_crops_numpy = list_to_numpy(aligned_frames_crops)
                # imshow_video(aligned_frames_numpy, FPS=25, frame_stride=2)
                # imshow_video(aligned_frames_crops_numpy, FPS=25, frame_stride=2)

        elif method == 'tracker':
            ### Optical Flow Bounding Box Tracker Method: ###
            initial_bbox = cv2.boundingRect(predicted_points_locations[0].astype(np.float32))  # Initial bounding box

            for t in range(predicted_points_locations.shape[0]):  # Loop through each frame
                prev_points = predicted_points_locations[t - 1] if t > 0 else predicted_points_locations[0]
                curr_points = predicted_points_locations[t]

                ### Calculate Optical Flow for Bounding Box ###
                flow_vectors = curr_points - prev_points
                new_coords = predicted_points_locations[0] + flow_vectors  # Transform points using optical flow

                ### Calculate New Bounding Box ###
                new_coords_H_min, new_coords_W_min = np.min(new_coords, axis=0)
                new_coords_H_max, new_coords_W_max = np.max(new_coords, axis=0)
                new_X0 = int(new_coords_W_min)
                new_Y0 = int(new_coords_H_min)
                new_W = int(new_coords_W_max - new_coords_W_min)
                new_H = int(new_coords_H_max - new_coords_H_min)
                new_bbox = (new_X0, new_Y0, new_W, new_H)

                ### Crop and Align Frame ###
                aligned_frame = frames[t][new_Y0:new_Y0 + new_H, new_X0:new_X0 + new_W]
                aligned_frames.append(aligned_frame)  # Add aligned frame to list

        elif method == 'predicted':
            ### Predicted Points Transformation Method: ###
            for t in range(predicted_points_locations.shape[0]):  # Loop through each set of predicted points
                points = predicted_points_locations[t]
                flow_vectors = points - predicted_points_locations[0]
                new_coords = predicted_points_locations[0] + flow_vectors  # Transform points using optical flow

                ### Calculate New Bounding Box Using Percentiles ###
                new_coords_H_min, new_coords_W_min = np.percentile(new_coords, 5, axis=0)
                new_coords_H_max, new_coords_W_max = np.percentile(new_coords, 95, axis=0)
                new_X0 = int(new_coords_W_min)
                new_Y0 = int(new_coords_H_min)
                new_W = int(new_coords_W_max - new_coords_W_min)
                new_H = int(new_coords_H_max - new_coords_H_min)
                new_bbox = (new_X0, new_Y0, new_W, new_H)

                ### Crop and Align Frame ###
                aligned_frame = frames[t][new_Y0:new_Y0 + new_H, new_X0:new_X0 + new_W]
                aligned_frames.append(aligned_frame)  # Add aligned frame to list


        ### This Is The Code Block: ###
        aligned_frames_np = np.array(aligned_frames)  # Convert list of aligned frames to numpy array
        aligned_frames_crops_np = np.array(aligned_frames_crops)  # Convert list of aligned frames to numpy array
        averaged_aligned_frame = np.mean(aligned_frames_np, axis=0).astype(np.uint8)  # Compute the averaged aligned frame
        averaged_aligned_frame_crops = np.mean(aligned_frames_crops_np, axis=0).astype(np.uint8)  # Compute the averaged aligned frame
        return aligned_frames, aligned_frames_crops, averaged_aligned_frame, averaged_aligned_frame_crops  # Return aligned frames and averaged frame


    ### Function to align frames using optical flow ###
    @staticmethod
    def align_frames_to_reference_using_given_optical_flow(frames, reference_frame, optical_flow):
        #TODO: make robust to accept pytorch tensors instead of numpy arrays
        """
        Align frames using given optical flow and compute the averaged aligned frame.

        Args:
            frames (list): List of frames (numpy arrays) to align. Each frame shape is [H, W, C].
            reference_frame (np.ndarray): Reference frame with shape [H, W, C].
            optical_flow (np.ndarray): Optical flow for each frame relative to reference frame. Shape is [T, H, W, 2].

        Returns:
            list: List of aligned frames.
            np.ndarray: Averaged aligned frame.
        """

        ### This Is The Code Block: ###
        h, w = reference_frame.shape[:2]  # Get height and width from reference frame shape
        aligned_frames = []  # List to store aligned frames

        ### Looping Over Indices: ###
        for t, flow in enumerate(optical_flow):  # Loop through each frame and its corresponding optical flow
            flow_map = np.zeros((h, w, 2), dtype=np.float32)  # Initialize flow map
            flow_map[:, :, 0] = flow[:, :, 0]  # Set flow map x-coordinates
            flow_map[:, :, 1] = flow[:, :, 1]  # Set flow map y-coordinates
            flow_map_intensity = AlignClass.get_optical_flow_intensity(flow_map)

            ### Calculate New Coordinates for Each Pixel: ###
            coords = np.dstack(np.meshgrid(np.arange(w), np.arange(h)))  # Create a grid of coordinates
            coords = coords.astype(np.float32)  # Convert to float for accurate calculations
            new_coords = coords + flow  # Add flow to coordinates to get new coordinates

            ### Interpolate: ###
            remap_frame = cv2.remap(frames[t], new_coords[:,:,0], new_coords[:,:,1], cv2.INTER_LINEAR)  # Remap frame using flow map

            # ### Debug: ###
            # imshow_np(frames[0], 'reference')
            # imshow_np(frames[t], 'new')
            # imshow_np(remap_frame, 'aligned')

            ### Append to frames: ###
            aligned_frames.append(remap_frame)  # Add aligned frame to list

        # ### Debug: ###
        # imshow_video(list_to_numpy(aligned_frames), FPS=5, frame_stride=2)

        ### This Is The Code Block: ###
        averaged_frame = np.mean(aligned_frames, axis=0).astype(np.uint8)  # Compute the averaged frame
        return aligned_frames, averaged_frame  # Return aligned frames and averaged frame

    @staticmethod
    def align_frames_crops_to_reference_using_given_optical_flow(frames, reference_frame, optical_flow, initial_BB_XYXY):
        # TODO: make robust to accept pytorch tensors instead of numpy arrays
        """
        Align frames using given optical flow and compute the averaged aligned frame.

        Args:
            frames (list): List of frames (numpy arrays) to align. Each frame shape is [H, W, C].
            reference_frame (np.ndarray): Reference frame with shape [H, W, C].
            optical_flow (np.ndarray): Optical flow for each frame relative to reference frame. Shape is [T, H, W, 2].

        Returns:
            list: List of aligned frames.
            np.ndarray: Averaged aligned frame.
        """

        ### This Is The Code Block: ###
        h, w = reference_frame.shape[:2]  # Get height and width from reference frame shape
        aligned_frames = []  # List to store aligned frames

        ### Looping Over Indices: ###
        for t, flow in enumerate(optical_flow):  # Loop through each frame and its corresponding optical flow
            flow_map = np.zeros((h, w, 2), dtype=np.float32)  # Initialize flow map
            flow_map[:, :, 0] = flow[:, :, 0]  # Set flow map x-coordinates
            flow_map[:, :, 1] = flow[:, :, 1]  # Set flow map y-coordinates
            flow_map_intensity = AlignClass.get_optical_flow_intensity(flow_map)

            ### Calculate New Coordinates for Each Pixel: ###
            coords = np.dstack(np.meshgrid(np.arange(w), np.arange(h)))  # Create a grid of coordinates
            coords = coords.astype(np.float32)  # Convert to float for accurate calculations
            new_coords = coords + flow  # Add flow to coordinates to get new coordinates

            ### Interpolate: ###
            remap_frame = cv2.remap(frames[t], new_coords[:, :, 0], new_coords[:, :, 1],
                                    cv2.INTER_LINEAR)  # Remap frame using flow map

            # ### Debug: ###
            # imshow_np(frames[0], 'reference')
            # imshow_np(frames[t], 'new')
            # imshow_np(remap_frame, 'aligned')

            ### Append to frames: ###
            aligned_frames.append(remap_frame)  # Add aligned frame to list

        # ### Debug: ###
        # imshow_video(list_to_numpy(aligned_frames), FPS=5, frame_stride=2)

        ### This Is The Code Block: ###
        averaged_frame = np.mean(aligned_frames, axis=0).astype(np.uint8)  # Compute the averaged frame
        X0, Y0, X1, Y1 = initial_BB_XYXY  # Extract initial bounding box coordinates
        aligned_crops = [aligned_frames[i][Y0:Y1, X0:X1] for i in range(len(aligned_frames))]  # Extract aligned crops
        average_crop = np.mean(aligned_crops, axis=0)
        return aligned_crops, average_crop  # Return aligned frames and averaged frame

    @staticmethod
    def align_frames_crops_using_optical_flow_to_homography_matrix(frames, reference_frame, optical_flow, initial_BB_XYXY):
        """
        Align frames using given optical flow, compute the averaged aligned frame,
        and extract crops based on the initial bounding box.

        Args:
            frames (list): List of frames (numpy arrays) to align. Each frame shape is [H, W, C].
            reference_frame (np.ndarray): Reference frame with shape [H, W, C].
            optical_flow (np.ndarray): Optical flow for each frame relative to reference frame. Shape is [T, H, W, 2].
            initial_BB_XYXY (tuple): Initial bounding box coordinates in the format (X0, Y0, X1, Y1).

        Returns:
            list: List of aligned crops.
            np.ndarray: Averaged aligned crop.
            list: List of crops extracted from each aligned frame.
        """

        ### Unpack Bounding Box Coordinates: ###
        X0, Y0, X1, Y1 = initial_BB_XYXY  # Extract bounding box coordinates

        ### Get Bounding Box Dimensions: ###
        h, w = reference_frame.shape[:2]  # Get height and width from reference frame shape

        aligned_crops = []  # List to store crops of aligned frames
        crops = []  # List to store crops from each aligned frame

        ### Looping Over Indices: ###
        for t, flow in enumerate(optical_flow):  # Loop through each frame and its corresponding optical flow
            flow_map = np.zeros((h, w, 2), dtype=np.float32)  # Initialize flow map
            flow_map[:, :, 0] = flow[:, :, 0]  # Set flow map x-coordinates
            flow_map[:, :, 1] = flow[:, :, 1]  # Set flow map y-coordinates

            ### Calculate New Coordinates for Each Pixel: ###
            coords = np.dstack(np.meshgrid(np.arange(w), np.arange(h)))  # Create a grid of coordinates
            coords = coords.astype(np.float32)  # Convert to float for accurate calculations
            new_coords = coords + flow  # Add flow to coordinates to get new coordinates

            ### Interpolate: ###
            remap_frame = cv2.remap(frames[t], new_coords[:, :, 0], new_coords[:, :, 1],
                                    cv2.INTER_LINEAR)  # Remap frame using flow map

            ### Extract Crop: ###
            crop = remap_frame[Y0:Y1 + 1, X0:X1 + 1]  # Extract the crop from the remapped frame
            aligned_crops.append(crop)  # Add crop to list
            crops.append(remap_frame[Y0:Y1 + 1, X0:X1 + 1])  # Add crop to list

        ### Compute Averaged Crop: ###
        averaged_crop = np.mean(aligned_crops, axis=0).astype(np.uint8)  # Compute the average of the crops

        return aligned_crops, averaged_crop  # Return the aligned crops, average crop, and crops extracted from each aligned frame

    @staticmethod
    def align_crops_in_frames_using_given_bounding_boxes(frames: list, bounding_boxes: np.ndarray) -> (list, np.ndarray):
        """
        Align crops using bounding boxes and compute the averaged crop using affine transformation.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
            reference_frame (np.ndarray): Reference frame with shape [H, W, C].
            bounding_boxes (np.ndarray): Bounding boxes for each frame. Shape is [T, 4] with each row [X0, Y0, X1, Y1].

        Returns:
            list: List of aligned crops.
            np.ndarray: Averaged crop.
        """

        ### Calculate Reference Crop Size and Coordinates: ###
        ref_x0, ref_y0, ref_x1, ref_y1 = bounding_boxes[0]  # Extract reference bounding box coordinates

        ### Get Coords: ###
        # ref_bbox_coords = np.array([[ref_x0, ref_y0], [ref_x1, ref_y0], [ref_x0, ref_y1]], dtype=np.float32)  # Top-left, top-right, bottom-left
        ref_bbox_coords = np.array([[ref_x0, ref_y0], [ref_x1, ref_y0], [ref_x0, ref_y1], [ref_x1, ref_y1]], dtype=np.float32)  # Top-left, top-right, bottom-left
        ref_width = ref_x1 - ref_x0  # Calculate reference bounding box width
        ref_height = ref_y1 - ref_y0  # Calculate reference bounding box height

        ### Looping Over Indices: ###
        aligned_crops = []
        for i, current_bbox in enumerate(bounding_boxes):  # Loop through each bounding box
            ### Get Bounding Box: ###
            x0, y0, x1, y1 = current_bbox  # Extract bounding box coordinates
            x0 = max(x0, 0)  # Make sure x0 is non-negative
            y0 = max(y0, 0)  # Make sure y0 is non-negative
            x1 = min(x1, frames[i].shape[1])  # Make sure x1 is within frame width
            y1 = min(y1, frames[i].shape[0])  # Make sure y1 is within frame height

            ### Get Coords: ###
            curr_bbox_coords = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float32)  # Top-left, top-right, bottom-left

            ### Calculate Affine Transformation Matrix: ###
            H = cv2.getPerspectiveTransform(curr_bbox_coords, ref_bbox_coords)  # Compute perspective transformation matrix

            ### Apply Resize: ###
            aligned_crop = cv2.resize(frames[i][y0:y1, x0:x1], (ref_width, ref_height), interpolation=cv2.INTER_LINEAR)  # W
            aligned_crops.append(aligned_crop)  # Add aligned crop to list

        ### This Is The Code Block: ###
        aligned_crops_np = np.array(aligned_crops)  # Convert list of aligned crops to numpy array
        averaged_crop = np.mean(aligned_crops_np, axis=0).astype(np.uint8)  # Compute the averaged crop
        return aligned_crops, averaged_crop  # Return aligned crops and averaged crop


    ### Function to align frames using homography matrices ###
    @staticmethod
    def align_frames_using_given_homographies(frames: list, reference_frame: np.ndarray, homographies: list) -> (list, np.ndarray):
        #TODO: make robust to accept pytorch tensors instead of numpy arrays
        """
        Align frames using given homography matrices and compute the averaged aligned frame.

        Args:
            frames (list): List of frames (numpy arrays) to align. Each frame shape is [H, W, C].
            reference_frame (np.ndarray): Reference frame with shape [H, W, C].
            homographies (list): List of homography matrices for each frame. Each homography is a 3x3 numpy array.

        Returns:
            list: List of aligned frames.
            np.ndarray: Averaged aligned frame.
        """

        ### This Is The Code Block: ###
        h, w = reference_frame.shape[:2]  # Get height and width from reference frame shape
        aligned_frames = []  # List to store aligned frames

        ### Looping Over Indices: ###
        for i, H in enumerate(homographies):  # Loop through each frame and its corresponding homography matrix
            aligned_frame = cv2.warpPerspective(frames[i], H, (w, h))  # Apply homography to align frame
            aligned_frames.append(aligned_frame)  # Add aligned frame to list

        ### This Is The Code Block: ###
        averaged_frame = np.mean(aligned_frames, axis=0).astype(np.uint8)  # Compute the averaged frame
        return aligned_frames, averaged_frame  # Return aligned frames and averaged frame

    @staticmethod
    def align_frames_crops_using_opencv_tracker(frames: list, initial_bbox_XYWH):
        ### Track Object Using OpenCV Tracker: ###
        bounding_boxes_array = AlignClass.track_object_using_opencv_tracker(initial_bbox_XYWH,
                                                                 frames)  # Generate bounding boxes for each frame

        ### Align Crops and Calculate Averaged Crop: ###
        aligned_crops, avg_crop = AlignClass.align_crops_in_frames_using_given_bounding_boxes(frames,
                                                                                   bounding_boxes_array)  # Align crops

        return aligned_crops, avg_crop

    ### Function to align crops using a bounding box and homography matrices ###
    @staticmethod
    def align_frame_crops_using_given_homographies(frames, reference_frame, bounding_box, homographies,
                                                   size_factor=1.0):
        """
        Align crops using a bounding box and homography matrices, then compute the averaged crop.

        Args:
            frames (list or tensor): List of frames (numpy arrays or PyTorch tensors). Each frame shape is [H, W, C].
            reference_frame (np.ndarray or tensor): Reference frame with shape [H, W, C].
            bounding_box (list): Bounding box coordinates on the reference frame [X0, Y0, X1, Y1].
            homographies (list): List of homography matrices for each frame. Each homography is a 3x3 numpy array.
            size_factor (float): Factor by which to expand the bounding box size.

        Returns:
            list: List of aligned crops.
            np.ndarray: Averaged crop.
        """
        ### Convert frames and reference_frame to numpy arrays if they are tensors ###
        if isinstance(frames[0], torch.Tensor):
            frames = [frame.cpu().numpy() for frame in frames]  # Convert each frame to numpy array
        if isinstance(reference_frame, torch.Tensor):
            reference_frame = reference_frame.cpu().numpy()  # Convert reference frame to numpy array

        ### Extract bounding box coordinates ###
        x0, y0, x1, y1 = bounding_box

        ### Calculate the expanded bounding box coordinates ###
        box_width = x1 - x0  # Calculate box width
        box_height = y1 - y0  # Calculate box height
        center_x = x0 + box_width // 2  # Calculate center x-coordinate
        center_y = y0 + box_height // 2  # Calculate center y-coordinate
        new_width = int(box_width * size_factor)  # Calculate new width
        new_height = int(box_height * size_factor)  # Calculate new height
        new_x0 = max(center_x - new_width // 2, 0)  # Calculate new top-left x-coordinate
        new_y0 = max(center_y - new_height // 2, 0)  # Calculate new top-left y-coordinate
        new_x1 = min(center_x + new_width // 2, reference_frame.shape[1])  # Calculate new bottom-right x-coordinate
        new_y1 = min(center_y + new_height // 2, reference_frame.shape[0])  # Calculate new bottom-right y-coordinate

        ### Initialize list to store aligned crops ###
        aligned_crops = []

        ### Loop through each frame and its corresponding homography matrix ###
        for i, H in enumerate(homographies):
            h, w = reference_frame.shape[:2]  # Get height and width from reference frame shape
            aligned_frame = cv2.warpPerspective(frames[i], H, (w, h))  # Apply homography to align frame
            crop = aligned_frame[new_y0:new_y1,
                   new_x0:new_x1]  # Crop the aligned frame using expanded bounding box coordinates
            aligned_crops.append(crop)  # Add crop to list

        ### Compute the averaged crop ###
        averaged_crop = np.mean(aligned_crops, axis=0).astype(np.uint8)

        return aligned_crops, averaged_crop  # Return aligned crops and averaged crop

    @staticmethod
    def adjust_shape_to_multiple(shape: tuple, multiple: int, min_shape_tuple: tuple = None, method: str = 'pad') -> tuple:
        """
        Adjust the dimensions of an image to the closest multiples of a given number and ensure minimum size.

        Args:
            shape (tuple): Original dimensions of the image (H, W).
            multiple (int): The number to which the dimensions should be multiples of.
            min_shape_tuple (tuple, optional): Minimum dimensions (H_min, W_min). Defaults to None.
            method (str): Method to adjust the dimensions ('pad' or 'crop'). Defaults to 'pad'.

        Returns:
            tuple: Adjusted dimensions (H_new, W_new).
        """
        H, W = shape  # Extract original dimensions

        ### Calculating Adjusted Dimensions: ###
        H_new = (H // multiple + (1 if H % multiple != 0 and method == 'pad' else 0)) * multiple  # Calculate new height
        W_new = (W // multiple + (1 if W % multiple != 0 and method == 'pad' else 0)) * multiple  # Calculate new width

        if method == 'crop':  # Check if method is 'crop'
            H_new = (H // multiple) * multiple  # Adjust height by cropping
            W_new = (W // multiple) * multiple  # Adjust width by cropping

        ### Ensure Minimum Dimensions: ###
        if min_shape_tuple is not None:
            H_min, W_min = min_shape_tuple  # Extract minimum dimensions
            H_new = max(H_new, H_min)  # Ensure new height meets minimum
            W_new = max(W_new, W_min)  # Ensure new width meets minimum

        return H_new, W_new  # Return adjusted dimensions

    @staticmethod
    def calculate_optical_flow_raft_pairwise(frames, model_path=None, return_numpy=False):
        """
        Calculate the optical flow between each pair of consecutive frames using the RAFT model from torchvision.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
            model_path (str): Path to the pretrained RAFT model weights (optional, for custom weights).
            return_numpy (bool): If True, return a list of numpy arrays instead of tensors.

        Returns:
            list: List of optical flow tensors or numpy arrays.
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ### Initializing RAFT Model: ###
        model = raft_large(pretrained=True).to(device)  # Initialize RAFT model
        model.eval()  # Set model to evaluation mode

        if model_path:
            model.load_state_dict(torch.load(model_path))  # Load custom weights if provided

        flows = []  # List to store optical flow tensors

        transform = transforms.Compose([
            transforms.ToTensor()  # Transform to tensor
        ])

        ### Looping Over Frame Pairs: ###
        for i in range(len(frames) - 1):  # Loop through frame pairs
            ### Transform arrays to tensors: ###
            frame1 = transform(frames[i]).unsqueeze(0).to(device)  # Transform and add batch dimension to first frame
            frame2 = transform(frames[i + 1]).unsqueeze(0).to(device)  # Transform and add batch dimension to second frame

            ### Adjust shape to multiple of 8: ###
            H_new, W_new = AlignClass.adjust_shape_to_multiple(frame1.shape[-2:], multiple=8, method='crop')  # Adjust shape to multiple of 32
            frame1 = crop_tensor(frame1, (H_new, W_new))  # Crop frame to adjusted shape
            frame2 = crop_tensor(frame2, (H_new, W_new))

            ### Forward through RAFT Model: ###
            with torch.no_grad():
                flow = model(frame1, frame2)[-1]  # Calculate optical flow
                torch.cuda.empty_cache()

            if return_numpy:
                flows.append(flow.squeeze(0).cpu().numpy())  # Append optical flow as numpy array to list
            else:
                flows.append(flow.squeeze(0).cpu())  # Append optical flow tensor to list

        return flows  # Return list of optical flow tensors or numpy arrays

    @staticmethod
    def calculate_optical_flow_raft_reference(frames, reference_frame, model_path=None, return_numpy=False):
        """
        Calculate the optical flow between each frame and a reference frame using the RAFT model from torchvision.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
            reference_frame (np.ndarray): The reference frame.
            model_path (str): Path to the pretrained RAFT model weights (optional, for custom weights).
            return_numpy (bool): If True, return a list of numpy arrays instead of tensors.

        Returns:
            list: List of optical flow tensors or numpy arrays.
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ### Initializing RAFT Model: ###
        model = raft_large(pretrained=True).to(device)  # Initialize RAFT model
        model.eval()  # Set model to evaluation mode

        if model_path:
            model.load_state_dict(torch.load(model_path))  # Load custom weights if provided

        flows = []  # List to store optical flow tensors

        transform = transforms.Compose([
            transforms.ToTensor()  # Transform to tensor
        ])

        ### Looping Over Frame Pairs: ###
        mew_frames_list = []
        for i in range(len(frames)):  # Loop through frame pairs
            ### Transform arrays to tensors: ###
            frame1 = transform(reference_frame).unsqueeze(0).to(device)  # Transform and add batch dimension to first frame
            frame2 = transform(frames[i]).unsqueeze(0).to(device)  # Transform and add batch dimension to second frame

            ### Adjust shape to multiple of 8: ###
            H_new, W_new = AlignClass.adjust_shape_to_multiple(frame1.shape[-2:], multiple=8, method='crop', min_shape_tuple=(224,224))  # Adjust shape to multiple of 32
            frame1 = crop_tensor(frame1, (H_new, W_new))  # Crop frame to adjusted shape
            frame2 = crop_tensor(frame2, (H_new, W_new))
            mew_frames_list.append(frame2)
            new_reference_frame = frame1

            ### Forward through RAFT Model: ###
            with torch.no_grad():
                flow = model(frame1, frame2)[-1]  # Calculate optical flow
                torch.cuda.empty_cache()

            if return_numpy:
                flows.append(torch_to_numpy(flow).squeeze(0))  # Append optical flow as numpy array to list
            else:
                flows.append(flow.cpu())  # Append optical flow tensor to list

        return flows, mew_frames_list, new_reference_frame  # Return list of optical flow tensors or numpy arrays


    ### Function: calculate_optical_flow_pairwise ###
    @staticmethod
    def calculate_optical_flow_pairwise_opencv(frames):
        """
        Calculate the optical flow between each pair of consecutive frames.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].

        Returns:
            list: List of optical flow arrays.
        """

        flows = []  # List to store optical flow arrays

        ### Looping Over Frame Pairs: ###
        for i in range(len(frames) - 1):  # Loop through frame pairs
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)  # Convert previous frame to grayscale
            curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # Calculate optical flow
            flows.append(flow)  # Append optical flow to list
        return flows  # Return list of optical flow arrays

    @staticmethod
    def calculate_optical_flow_to_reference_frame_opencv(frames, reference_frame):
        """
        Calculate the optical flow between each frame and a reference frame.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
            reference_frame (numpy array): frame shape is [H, W, C].

        Returns:
            list: List of optical flow arrays relative to the reference frame.
        """

        ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)  # Convert reference frame to grayscale
        h,w = ref_gray.shape[0:2]
        flows = []  # List to store optical flow arrays

        ### Looping Over Frames: ###
        for i, frame in enumerate(frames):  # Loop through each frame
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale
            if (curr_gray - ref_gray).sum() < 1e-3:
                flows.append(np.zeros((h,w,2), dtype=np.float32))  # No flow for reference frame itself
                continue
            flow = cv2.calcOpticalFlowFarneback(ref_gray, curr_gray, None, 0.3, 5, 20, 7, 7, 1.5, 0)  # Calculate optical flow
            flows.append(flow)  # Append optical flow to list

        return flows  # Return list of optical flow arrays relative to reference frame

    @staticmethod
    def convert_pairwise_optical_flow_to_relative_to_first(pairwise_flows):
        """
        Calculate the optical flow relative to the first frame from pairwise optical flows.

        Args:
            pairwise_flows (list): List of pairwise optical flow arrays.

        Returns:
            list: List of optical flow arrays relative to the first frame.
        """

        num_frames = len(pairwise_flows) + 1  # Number of frames
        h, w, _ = pairwise_flows[0].shape  # Shape of the optical flow arrays

        flow_relative_to_first = [np.zeros((h, w, 2), dtype=np.float32)]  # Initialize with zero flow for the first frame
        cumulative_flow = np.zeros((h, w, 2), dtype=np.float32)  # Initialize cumulative flow

        ### Looping Over Pairwise Flows: ###
        for i, flow in enumerate(pairwise_flows):  # Loop through each pairwise flow
            ### Calculate New Coordinates for Each Pixel: ###
            coords = np.dstack(np.meshgrid(np.arange(w), np.arange(h)))  # Create a grid of coordinates
            coords = coords.astype(np.float32)  # Convert to float for accurate calculations
            new_coords = coords + cumulative_flow  # Apply cumulative flow to get new coordinates

            ### Interpolate Flow at New Coordinates: ###
            flow_x = cv2.remap(flow[..., 0], new_coords[..., 0], new_coords[..., 1], interpolation=cv2.INTER_LINEAR)
            flow_y = cv2.remap(flow[..., 1], new_coords[..., 0], new_coords[..., 1], interpolation=cv2.INTER_LINEAR)
            flow_at_new_coords = np.stack((flow_x, flow_y), axis=-1)  # Combine x and y components

            cumulative_flow += flow_at_new_coords  # Update cumulative flow
            flow_relative_to_first.append(cumulative_flow.copy())  # Append cumulative flow to list

        return flow_relative_to_first  # Return list of optical flow arrays relative to the first frame


    @staticmethod
    def get_optical_flow_intensity(optical_flows):
        """
        Calculate the optical flow intensity for each optical flow array or tensor.

        Args:
            optical_flows (list, np.ndarray, or torch.Tensor): List of optical flow arrays, a single optical flow array, or a tensor.

        Returns:
            list or torch.Tensor: List of optical flow intensity arrays or a tensor of intensities.
        """

        def calculate_intensity(flow):
            """
            Calculate the optical flow intensity for a single flow array or tensor.

            Args:
                flow (np.ndarray or torch.Tensor): Optical flow array or tensor.

            Returns:
                np.ndarray or torch.Tensor: Optical flow intensity.
            """
            if isinstance(flow, np.ndarray):  # Check if the flow is a numpy array
                if flow.shape[-1] == 2:  # Check if the last dimension is 2
                    flow_x, flow_y = flow[..., 0], flow[..., 1]  # Separate x and y components
                elif flow.shape[0] == 2:  # Check if the first dimension is 2
                    flow_x, flow_y = flow[0, ...], flow[1, ...]  # Separate x and y components
                else:
                    raise ValueError("Unsupported shape for optical flow. Expected [H, W, 2] or [2, H, W].")  # Raise error for unsupported shape
                intensity = np.sqrt(flow_x**2 + flow_y**2)  # Calculate flow intensity

            elif isinstance(flow, torch.Tensor):  # Check if the flow is a PyTorch tensor
                if flow.shape[-1] == 2:  # Check if the last dimension is 2
                    flow_x, flow_y = flow[..., 0], flow[..., 1]  # Separate x and y components
                elif flow.shape[1] == 2:  # Check if the second dimension is 2
                    flow_x, flow_y = flow[:, 0, ...], flow[:, 1, ...]  # Separate x and y components
                else:
                    raise ValueError("Unsupported shape for optical flow. Expected [H, W, 2] or [2, H, W].")  # Raise error for unsupported shape
                intensity = torch.sqrt(flow_x**2 + flow_y**2)  # Calculate flow intensity

            else:
                raise TypeError("Unsupported type for optical flow. Expected numpy array or PyTorch tensor.")  # Raise error for unsupported type

            return intensity  # Return optical flow intensity

        if isinstance(optical_flows, list):  # Check if the input is a list
            intensities = [calculate_intensity(flow) for flow in optical_flows]  # Calculate intensity for each flow
            return intensities  # Return list of intensities

        elif isinstance(optical_flows, torch.Tensor):  # Check if the input is a PyTorch tensor
            if optical_flows.dim() == 4:  # Check if the tensor has 4 dimensions
                if optical_flows.shape[1] == 2:  # Check if the second dimension is 2
                    intensities = torch.sqrt(optical_flows[:, 0, ...]**2 + optical_flows[:, 1, ...]**2)  # Calculate intensity
                elif optical_flows.shape[-1] == 2:  # Check if the last dimension is 2
                    intensities = torch.sqrt(optical_flows[..., 0]**2 + optical_flows[..., 1]**2)  # Calculate intensity
                else:
                    raise ValueError("Unsupported shape for optical flow. Expected [B, 2, H, W] or [B, H, W, 2].")  # Raise error for unsupported shape
                return intensities.unsqueeze(1)  # Return tensor with shape [B, 1, H, W]
            else:
                return calculate_intensity(optical_flows).unsqueeze(0).unsqueeze(0)  # Calculate intensity for single tensor and add batch and channel dimensions

        elif isinstance(optical_flows, np.ndarray):  # Check if the input is a numpy array
            if optical_flows.ndim == 4 and optical_flows.shape[-1] == 2:  # Check if the array has 4 dimensions and last dimension is 2
                intensities = np.sqrt(optical_flows[..., 0]**2 + optical_flows[..., 1]**2)  # Calculate intensity
                return intensities  # Return intensity array
            elif optical_flows.ndim == 3 and optical_flows.shape[0] == 2:  # Check if the array has 3 dimensions and first dimension is 2
                intensities = np.sqrt(optical_flows[0, ...]**2 + optical_flows[1, ...]**2)  # Calculate intensity
                return intensities  # Return intensity array
            else:
                return calculate_intensity(optical_flows)  # Calculate intensity for single array

        else:
            raise TypeError("Unsupported type for optical flow. Expected list, numpy array, or PyTorch tensor.")  # Raise error for unsupported type

    ### Function: track_points_using_optical_flow_opencv ###
    @staticmethod
    def track_points_using_optical_flow_opencv(frames, points):
        #TODO: this is specifically using opencv classical optical flow, i should make it robust to other types of optical flow
        """
        Track points across frames using optical flow.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
            points (np.ndarray): Initial points to track with shape [N, 2].

        Returns:
            np.ndarray: Tracked points with shape [T, N, 2].
        """

        tracked_points = []  # List to store tracked points
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  # Parameters for Lucas-Kanade optical flow
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)  # Convert first frame to grayscale
        p0 = np.float32(points).reshape(-1, 1, 2)  # Reshape initial points
        tracked_points.append(p0)  # Append initial points to tracked points

        ### Looping Over Frames: ###
        for i in range(1, len(frames)):  # Loop through remaining frames
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)  # Calculate optical flow
            tracked_points.append(p1)  # Append tracked points to list
            prev_gray = curr_gray  # Update previous frame
            p0 = p1  # Update initial points

        return np.array(tracked_points).squeeze()  # Return tracked points

    @staticmethod
    def align_frames_to_reference_using_given_shifts(frames: list, reference_frame: np.ndarray, delta_H: np.ndarray, delta_W: np.ndarray) -> (list, np.ndarray):
        #TODO: make robust to accept pytorch tensors instead of numpy arrays
        """
        Align frames based on given translation deltas and compute the averaged aligned frame.

        Args:
            frames (list): List of frames (numpy arrays) to align. Each frame shape is [H, W, C].
            reference_frame (np.ndarray): Reference frame with shape [H, W, C].
            delta_H (np.ndarray): Array of height translations for each frame relative to reference frame. Shape is [T].
            delta_W (np.ndarray): Array of width translations for each frame relative to reference frame. Shape is [T].

        Returns:
            list: List of aligned frames relative to the reference frame.
            np.ndarray: Averaged aligned frame.
        """

        ### This Is The Code Block: ###
        aligned_frames = []  # List to store aligned frames
        h, w = reference_frame.shape[:2]  # Get height and width from reference frame shape

        ### Looping Over Indices: ###
        for i, frame in enumerate(frames):  # Loop through each frame
            M = np.float32([[1, 0, delta_W[i]], [0, 1, delta_H[i]]])  # Create translation matrix for current frame
            aligned_frame = cv2.warpAffine(frame, M, (w, h))  # Apply translation to align frame
            aligned_frames.append(aligned_frame)  # Add aligned frame to list

        ### This Is The Code Block: ###
        averaged_frame = np.mean(aligned_frames, axis=0).astype(np.uint8)  # Compute the averaged frame
        return aligned_frames, averaged_frame  # Return aligned frames and averaged frame


    ### Function: generate_bounding_boxes_with_tracker ###
    @staticmethod
    def track_object_using_opencv_tracker(initial_bbox_XYWH, frames, tracker_type='CSRT'):
        """
        Generate bounding boxes for each frame based on the initial bounding box using an OpenCV tracker.

        Args:
            initial_bbox (tuple): The initial bounding box coordinates (x0, y0, w, h).
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
            tracker_type (str): Type of OpenCV tracker to use. Options are 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'.

        Returns:
            np.ndarray: Bounding boxes for each frame. Shape is [T, 4].
        """

        bounding_boxes = []  # List to store bounding boxes
        tracker = AlignClass.create_tracker(tracker_type)  # Create tracker

        # initial_bbox_cv = (initial_bbox[0], initial_bbox[1], initial_bbox[2] - initial_bbox[0], initial_bbox[3] - initial_bbox[1])  # Convert to OpenCV format (x, y, w, h)
        tracker.init(frames[0], initial_bbox_XYWH)  # Initialize tracker with the first frame and initial bounding box

        ### Looping Over Frames: ###
        for frame in frames:  # Loop through each frame
            success, bbox_cv = tracker.update(frame)  # Update tracker and get new bounding box
            if success:
                bbox = (int(bbox_cv[0]), int(bbox_cv[1]), int(bbox_cv[0] + bbox_cv[2]), int(bbox_cv[1] + bbox_cv[3]))  # Convert to (x0, y0, x1, y1)
            else:
                bbox = bounding_boxes[-1]  # If tracking fails, use the previous bounding box
            bounding_boxes.append(bbox)  # Append bounding box to list

        return np.array(bounding_boxes)  # Return bounding boxes for each frame

    ### Function: create_tracker ###
    @staticmethod
    def create_tracker(tracker_type):
        """
        Create an OpenCV tracker based on the specified type.

        Args:
            tracker_type (str): Type of OpenCV tracker to use. Options are 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'.

        Returns:
            tracker: OpenCV tracker object.
        """

        if tracker_type == 'BOOSTING':
            return cv2.legacy.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            return cv2.legacy.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            return cv2.legacy.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            return cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        elif tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        else:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")

    @staticmethod
    def apply_random_translation_to_image(image, max_translation=10):
        """
        Perform a random translation on the given image.

        Args:
            image (np.ndarray): The input image to be translated.
            max_translation (int): The maximum translation in both x and y directions.

        Returns:
            np.ndarray: The translated image.
        """
        height, width = image.shape[:2]

        ### Generating Random Translation Values: ###
        tx = get_random_number_in_range(-max_translation, max_translation)[0]  # Random translation in x direction
        ty = get_random_number_in_range(-max_translation, max_translation)[0]  # Random translation in y direction

        ### Creating the Translation Matrix: ###
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])  # Translation matrix

        ### Performing the Translation: ###
        translated_image = cv2.warpAffine(image, translation_matrix, (width, height))  # Translate image

        return translated_image  # Return translated image

    @staticmethod
    def apply_random_translation_to_images(image_list, max_translation=10):
        """
        Apply random global translation to each image in the list.

        Args:
            image_list (list): List of images (numpy arrays).
            max_translation (int): The maximum translation in both x and y directions.

        Returns:
            list: List of translated images.
        """
        ### Initializing the List to Store Translated Images: ###
        translated_images = []  # List to store translated images

        ### Looping Over Each Image: ###
        for image in image_list:  # Loop through each image in the list
            translated_image = AlignClass.apply_random_translation_to_image(image, max_translation)  # Translate image
            translated_images.append(translated_image)  # Append translated image to list

        return translated_images  # Return list of translated images

    ### Function: test_align_frames_homography ###
    @staticmethod
    def test_align_frames_homography(frames, flag_plot=False):
        """
        Test function for aligning frames using homography matrices.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
        """

        ### Apply random translation to frames: ###
        frames = AlignClass.apply_random_translation_to_images(frames, max_translation=10)  # Apply random translation to frames
        # frames_numpy = list_to_numpy(frames); imshow_video(frames_numpy, FPS=25, frame_stride=3)

        ### Extract Reference Frame and Calculate Homographies: ###
        ref_frame = frames[0]  # Reference frame

        ### Align Frames and Calculate Averaged Frame: ###
        # aligned_frames, avg_frame = align_frames_using_given_homographies(frames, ref_frame, homographies)  # Align frames
        aligned_frames, homography_matrices_list, last_frame_features = stabilize_frames_FeatureBased(frames,
                                      reference_frame=frames[0],
                                      flag_output_array_form='list',
                                      number_of_features_for_ORB=2500,
                                      max_iterations=5,
                                      inlier_threshold=2.5,
                                      c=4.685,
                                      interpolation='nearest',
                                      flag_registration_algorithm='regular',
                                      flag_perform_interpolation=True,
                                      last_frame_features=[None, None],
                                      flag_downsample_frames=False,
                                      binning_factor=1,
                                      flag_RGB2BW=False,
                                      flag_matching_crosscheck=False)
        # aligned_frames_numpy = list_to_numpy(aligned_frames)
        # avg_frame = np.mean(aligned_frames_numpy, axis=0).clip(0, 255).astype(np.uint8)  # Calculate averaged frame
        # imshow_video(aligned_frames_numpy, FPS=25, frame_stride=3)  #

        ### Align Frames Using Above Calculated Homographies: ###
        aligned_frames, averaged_frame = AlignClass.align_frames_using_given_homographies(frames, ref_frame, homography_matrices_list)  # Align
        aligned_frames_numpy = list_to_numpy(aligned_frames)
        avg_frame = np.mean(aligned_frames_numpy, axis=0).clip(0, 255).astype(np.uint8)  # Calculate averaged frame
        # imshow_video(aligned_frames_numpy, FPS=25, frame_stride=3)  #

        ### Display Aligned Frames: ###
        if flag_plot:
            imshow_video(aligned_frames_numpy, FPS=25, frame_stride=1)  #

            ### Display Averaged Frame: ###
            plt.figure()  # Create figure
            plt.imshow(cv2.cvtColor(avg_frame, cv2.COLOR_BGR2RGB))  # Display averaged frame
            plt.title('Averaged Frame')  # Title for the averaged frame
            plt.show()  # Show averaged frame

        return aligned_frames, avg_frame, homography_matrices_list

    ### Function: test_align_frame_crops_using_given_homographies ###
    @staticmethod
    def test_align_frame_crops_using_given_homographies(frames, flag_plot=False):
        """
        Test function for aligning crops using homography matrices and a bounding box.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
        """
        ### Extract Reference Frame and Bounding Box: ###
        ref_frame = frames[0]  # Reference frame
        bbox = draw_bounding_box(ref_frame)  # Draw bounding box

        ### Apply random translation to frames: ###
        frames = AlignClass.apply_random_translation_to_images(frames, max_translation=10)  # Apply random translation to frames
        # frames_numpy = list_to_numpy(frames); imshow_video(frames_numpy, FPS=25, frame_stride=3)

        ### Align Frames and Calculate Averaged Frame And Homography: ###
        aligned_frames, homography_matrices_list, last_frame_features = stabilize_frames_FeatureBased(frames,
                                                                                                      reference_frame=frames[0],
                                                                                                      flag_output_array_form='list',
                                                                                                      number_of_features_for_ORB=2500,
                                                                                                      max_iterations=5,
                                                                                                      inlier_threshold=2.5,
                                                                                                      c=4.685,
                                                                                                      interpolation='nearest',
                                                                                                      flag_registration_algorithm='regular',
                                                                                                      flag_perform_interpolation=True,
                                                                                                      last_frame_features=[None, None],
                                                                                                      flag_downsample_frames=False,
                                                                                                      binning_factor=1,
                                                                                                      flag_RGB2BW=False,
                                                                                                      flag_matching_crosscheck=False)
        aligned_frames_numpy = list_to_numpy(aligned_frames)
        avg_frame = np.mean(aligned_frames_numpy, axis=0).clip(0, 255).astype(np.uint8)  # Calculate averaged frame
        # imshow_video(aligned_frames, FPS=25, frame_stride=1)  #

        ### Align Crops and Calculate Averaged Crop: ###
        aligned_crops, avg_crop = AlignClass.align_frame_crops_using_given_homographies(frames, ref_frame, bbox, homography_matrices_list)  # Align crops

        ### Display Aligned Frames: ###
        if flag_plot:
            aligned_crops_numpy = list_to_numpy(aligned_crops)  # Convert aligned crops to numpy
            imshow_video(aligned_crops_numpy, FPS=25, frame_stride=1)  #

            ### Display Averaged Frame: ###
            plt.figure()  # Create figure
            plt.imshow(cv2.cvtColor(avg_crop, cv2.COLOR_BGR2RGB))  # Display averaged frame
            plt.title('Averaged Frame')  # Title for the averaged frame
            plt.show()  # Show averaged frame

        return aligned_crops, avg_crop, homography_matrices_list

    @staticmethod
    def find_homographies_from_points_list(predicted_points_array, reference_points, method='ransac', max_iterations=100, inlier_threshold=3):
        """
        Calculate homographies from a list of predicted points using the specified method.

        Formerly named: calculate_homographies

        Inputs:
        - reference_points: numpy array of reference points of shape [N, 2]
        - predicted_points_array: numpy array of predicted points for each frame of shape [T, N, 2] or list of [N, 2]
        - method: str, method to estimate homography ('ransac' or 'wls')

        Outputs:
        - homographies: list of homography matrices
        """

        ### Initialize List for Homographies: ###
        homographies = []  # Initialize list for homographies

        ### Looping Over Indices: ###
        for i in range(len(predicted_points_array)):  # Iterate through predicted points
            if method == 'ransac':  # If method is RANSAC
                H, _ = cv2.findHomography(predicted_points_array[i], reference_points, cv2.RANSAC, maxIters=max_iterations)  # Estimate homography using RANSAC
            elif method == 'least_squares':  # If method is
                H = AlignClass.find_homography_for_two_point_sets_simple_least_squares(predicted_points_array[i], reference_points)  # Estimate homography using LS
            elif method == 'weighted_least_squares':  # If method is WLS
                H = AlignClass.find_homography_for_two_point_sets_weighted_least_squares(predicted_points_array[i], reference_points)
            elif method == 'iterative_reweighted_least_squares':  # If method is WLS
                H = AlignClass.find_homography_for_two_point_sets_iterative_reweighted_least_squares(predicted_points_array[i],
                                                                                                     reference_points,
                                                                                                     max_iterations=max_iterations,
                                                                                                     inlier_threshold=inlier_threshold)
            homographies.append(H)  # Append homography

        return homographies  # Return list of homographies

    @staticmethod
    def find_homography_for_two_point_sets_simple_least_squares(src_points, dst_points):
        """
        Find homography using simple least squares.

        Formerly named: find_homography_wls

        Inputs:
        - src_points: numpy array of source points of shape [N, 2]
        - dst_points: numpy array of destination points of shape [N, 2]

        Outputs:
        - H: homography matrix of shape [3, 3]
        """

        ### Initialize Matrices A and B: ###
        A = []  # Initialize matrix A
        B = []  # Initialize matrix B

        ### Looping Over Indices: ###
        for i in range(len(src_points)):  # Iterate through source points
            x, y = src_points[i]  # Source point coordinates
            u, v = dst_points[i]  # Destination point coordinates
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])  # Append to A
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y])  # Append to A
            B.append(u)  # Append u to B
            B.append(v)  # Append v to B

        ### Convert A and B to Numpy Arrays: ###
        A = np.array(A)  # Convert A to numpy array
        B = np.array(B)  # Convert B to numpy array

        ### Solve for H: ###
        H = np.linalg.lstsq(A, B, rcond=None)[0]  # Solve for H
        H = np.append(H, 1).reshape(3, 3)  # Reshape H to 3x3

        return H  # Return homography matrix

    @staticmethod
    def find_homography_for_two_point_sets_weighted_least_squares(src_pts, dst_pts, weights):
        """
        Calculate homography matrix using weighted least squares.

        Parameters:
        - src_pts (np.ndarray): Source points.
        - dst_pts (np.ndarray): Destination points.
        - weights (np.ndarray): Weights for each point.

        Returns:
        - H (np.ndarray): Homography matrix.
        """

        A = []  ### Initialize list to store A matrix
        B = []  ### Initialize list to store B vector

        ### Construct A matrix and B vector: ###
        for (x1, y1), (x2, y2), w in zip(src_pts, dst_pts, weights):
            A.append([x1, y1, 1, 0, 0, 0, -w * x1 * x2, -w * y1 * x2])  ### Append row to A matrix
            A.append([0, 0, 0, x1, y1, 1, -w * x1 * y2, -w * y1 * y2])  ### Append row to A matrix
            B.append(x2)  ### Append x2 to B vector
            B.append(y2)  ### Append y2 to B vector

        A = np.array(A)  ### Convert A list to numpy array
        B = np.array(B)  ### Convert B list to numpy array

        ### Solve for h in Ah = B: ###
        h, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  ### Solve least squares problem
        H = np.append(h, 1).reshape(3, 3)  ### Reshape h to 3x3 homography matrix
        return H  ### Return homography matrix

    @staticmethod
    def find_homography_for_two_point_sets_iterative_reweighted_least_squares(src_pts, dst_pts, max_iterations=2000, inlier_threshold=5.0, c=4.685):
        """
        Aligns and refines the homography matrix using RANSAC and weighted least squares.

        Inputs:
        - src_pts: numpy array of source points of shape [N, 2]
        - dst_pts: numpy array of destination points of shape [N, 2]
        - max_iterations: int, maximum number of RANSAC iterations
        - inlier_threshold: float, threshold to determine inliers
        - c: float, tuning constant for Tukey's Biweight function

        Outputs:
        - H: numpy array, refined homography matrix of shape [3, 3]
        """

        ### Initialize Variables: ###
        for iteration in range(max_iterations):  # Iterate through maximum iterations

            ### Find Homography Using RANSAC: ###
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0,
                                         maxIters=2000)  # Find homography using RANSAC
            if H is None:  # If homography is None
                break  # Break the loop

            ### Get RANSAC Inliers: ###
            RANSAC_inliers = mask.ravel().astype(bool)  # Get RANSAC inliers
            num_inliers = np.sum(RANSAC_inliers)  # Count number of inliers
            if num_inliers < 4:  # If number of inliers is less than 4
                break  # Break the loop

            ### Transform Source Points and Calculate Residuals: ###
            transformed_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)  # Transform source points
            residuals = np.linalg.norm(src_pts - transformed_pts, axis=1)  # Calculate residuals
            outliers = residuals > inlier_threshold  # Identify outliers
            RANSAC_inliers[RANSAC_inliers] = ~outliers[RANSAC_inliers]  # Update RANSAC inliers

            ### Reweight Inliers Using Tukey's Biweight Function: ###
            weights = tukey_biweight(residuals, c)  # Reweight inliers using Tukey's Biweight function
            weights = weights / np.sum(weights)  # Normalize weights

            ### Get Inlier Points and Weights: ###
            src_pts_inliers = src_pts[RANSAC_inliers]  # Get inlier source points
            dst_pts_inliers = dst_pts[RANSAC_inliers]  # Get inlier destination points
            weights_inliers = weights[RANSAC_inliers]  # Get inlier weights

            ### Update Homography Using Weighted Least Squares: ###
            H = weighted_least_squares_homography_matrix_points(src_pts_inliers, dst_pts_inliers, weights_inliers)  # Update homography using weighted least squares

            ### Update Points and Weights for Next Iteration: ###
            src_pts = src_pts[RANSAC_inliers]  # Update source points for next iteration
            dst_pts = dst_pts[RANSAC_inliers]  # Update destination points for next iteration
            weights = weights[RANSAC_inliers]  # Update weights for next iteration

            ### Check for Convergence: ###
            if num_inliers == np.sum(RANSAC_inliers):  # If number of inliers does not change
                break  # Break the loop

        return H  # Return refined homography matrix


    @staticmethod
    def draw_bounding_boxes_on_images(images, bounding_boxes):
        """
        Draw bounding boxes on a list of images.

        Args:
            images (list): List of images (numpy arrays). Each image shape is [H, W, C].
            bounding_boxes (np.ndarray): Array of bounding boxes with shape [T, 4], where each row is [X0, Y0, X1, Y1].

        Returns:
            list: List of images with bounding boxes drawn on them.
        """
        ### This Is The Code Block: ###
        images_with_bboxes = []  # List to store images with bounding boxes

        ### Looping Over Each Image and Bounding Box: ###
        for i, (image, bbox) in enumerate(zip(images, bounding_boxes)):  # Loop through each image and bounding box
            x0, y0, x1, y1 = bbox  # Extract bounding box coordinates

            ### Copying Image to Avoid In-place Modification: ###
            image_with_bbox = image.copy()  # Copy the image to avoid in-place modification

            ### Drawing the Bounding Box on the Image: ###
            cv2.rectangle(image_with_bbox, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)  # Draw bounding box

            ### Appending Image with Bounding Box to List: ###
            images_with_bboxes.append(image_with_bbox)  # Append image with bounding box to list

        return images_with_bboxes  # Return list of images with bounding boxes

    ### Function: test_align_crops_in_frames_using_given_bounding_boxes ###
    @staticmethod
    def test_align_crops_in_frames_using_given_bounding_boxes(frames, flag_plot=False):
        """
        Test function for aligning crops using bounding boxes.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
        """
        ### Extract Reference Frame and Initial Bounding Box: ###
        ref_frame = frames[0]  # Reference frame
        initial_bbox_XYWH = draw_bounding_box(ref_frame)  # Draw initial bounding box

        ### Track Object Using OpenCV Tracker: ###
        bounding_boxes_array = AlignClass.track_object_using_opencv_tracker(initial_bbox_XYWH, frames)  # Generate bounding boxes for each frame

        ### Plot BB on Frames (DEBUG): ###
        frames_with_BB = AlignClass.draw_bounding_boxes_on_images(frames, bounding_boxes_array)  # Draw bounding boxes on frames
        # frames_with_BB_numpy = list_to_numpy(frames_with_BB)
        # imshow_video(frames_with_BB_numpy, FPS=25, frame_stride=3)

        ### Align Crops and Calculate Averaged Crop: ###
        aligned_crops, avg_crop = AlignClass.align_crops_in_frames_using_given_bounding_boxes(frames, bounding_boxes_array)  # Align crops

        ### Display Aligned Frames: ###
        if flag_plot:
            aligned_crops_numpy = list_to_numpy(aligned_crops)  # Convert aligned crops to numpy
            imshow_video(aligned_crops_numpy, FPS=25, frame_stride=1)  #

            ### Display Averaged Frame: ###
            plt.figure()  # Create figure
            plt.imshow(cv2.cvtColor(avg_crop, cv2.COLOR_BGR2RGB))  # Display averaged frame
            plt.title('Averaged Frame')  # Title for the averaged frame
            plt.show()  # Show averaged frame

    ### Function: test_align_crops_optical_flow ###
    @staticmethod
    def test_align_crops_optical_flow(frames, flag_plot=False, flag_pairwise_or_reference='reference'):
        """
        Test function for aligning crops using optical flow and a bounding box.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
        """
        ### Extract Reference Frame and Bounding Box: ###
        ref_frame = frames[0]  # Reference frame
        bbox = draw_bounding_box(ref_frame)  # Draw bounding box

        ### Get aligned crops using trackers before using optcal flow: ###
        aligned_crops, average_crop = AlignClass.align_frames_crops_using_opencv_tracker(frames, initial_bbox_XYWH=bbox)

        ### Crop to multiple of 8: ###
        (H_new, W_new) = AlignClass.adjust_shape_to_multiple(aligned_crops[0].shape[0:2], multiple=8, method='crop')
        aligned_crops = [crop_tensor(aligned_crops[i], (H_new, W_new)) for i in np.arange(len(aligned_crops))]

        ### If size is smaller then (224,224) to resize it: ###
        min_size = 224
        aspect_ratio = (W_new / H_new)
        if aspect_ratio < 1:
            W_new = int(min_size * aspect_ratio)
            W_new = (W_new // 8) * 8
            H_new = min_size
            H_new = (H_new // 8) * 8
        else:
            H_new = min_size
            W_new = int(H_new * aspect_ratio)
            W_new = (W_new // 8) * 8
        aligned_crops = [cv2.resize(aligned_crops[i], (W_new, H_new), interpolation=cv2.INTER_LINEAR) for i in np.arange(len(aligned_crops))]

        ### Calculate Optical Flow On Entire Frames Relative To First: ###
        if flag_pairwise_or_reference == 'pairwise':
            #### Calculating pairwise and then converting: ###
            # optical_flows = calculate_optical_flow_pairwise_opencv(frames)  # Calculate optical flows
            optical_flows = AlignClass.calculate_optical_flow_raft_pairwise(aligned_crops)
            # optical_flows = convert_pairwise_optical_flow_to_relative_to_first(optical_flows)
        elif flag_pairwise_or_reference == 'reference':
            ### Calculating Directly relative to first: ###
            # optical_flows = calculate_optical_flow_to_reference_frame_opencv(frames, frames[0])
            optical_flows, frames_list, reference_frame = AlignClass.calculate_optical_flow_raft_reference(aligned_crops, aligned_crops[0], return_numpy=True)
            optical_flows.append(np.zeros_like(optical_flows[0]))  # Append zeros for

        # ### Get Optical Flows Intensities (Debugging: ###
        # optical_flow_intensities = get_optical_flow_intensity(optical_flows)
        # optical_flow_intensities_numpy = list_to_numpy(optical_flow_intensities)
        # # imshow_video(optical_flow_intensities_numpy, FPS=25, frame_stride=1)
        # # imshow_video(list_to_numpy(frames), FPS=25, frame_stride=1)

        ### Align Crops and Calculate Averaged Crop: ###
        aligned_crops, avg_crop = AlignClass.align_frames_to_reference_using_given_optical_flow(aligned_crops, aligned_crops[0], optical_flows)  # Align crops

        ### Display Aligned Crops: ###
        if flag_plot:
            aligned_crops_numpy = list_to_numpy(aligned_crops)
            imshow_video(aligned_crops_numpy, FPS=2)
            average_aligned_crop = aligned_crops_numpy[40:45].astype(float).mean(0)
            imshow_np(average_aligned_crop/255)

    @staticmethod
    def draw_circles_on_image(points_array, input_image, color=(0, 255, 0), radius=5, thickness=2):
        """
        Draw circles on the input image at the specified points.

        Args:
            points_array (np.ndarray): Array of points with shape [N, 2].
            input_image (np.ndarray): The input image on which to draw the circles.
            color (tuple): Color of the circles (default is green).
            radius (int): Radius of the circles (default is 5).
            thickness (int): Thickness of the circle outlines (default is 2).

        Returns:
            np.ndarray: Image with circles drawn on it.
        """
        output_image = input_image.copy()  # Create a copy of the input image to draw on

        ### Drawing Circles: ###
        for point in points_array:  # Loop through each point
            x, y = int(point[0]), int(point[1])  # Extract coordinates and convert to integers
            cv2.circle(output_image, (x, y), radius, color, thickness)  # Draw circle at the specified point

        return output_image  # Return the image with circles drawn on it

    @staticmethod
    def draw_circles_on_images(points_arrays, images, color=(0, 255, 0), radius=5, thickness=2):
        """
        Draw circles on a list of images at the specified points for each image.

        Args:
            points_arrays (list of np.ndarray): List of arrays of points with shape [N, 2] for each image.
            images (list of np.ndarray): List of input images on which to draw the circles.
            color (tuple): Color of the circles (default is green).
            radius (int): Radius of the circles (default is 5).
            thickness (int): Thickness of the circle outlines (default is 2).

        Returns:
            list of np.ndarray: List of images with circles drawn on them.
        """
        output_images = []  # List to store output images

        ### Looping Over Images and Points Arrays: ###
        for points_array, image in zip(points_arrays, images):  # Loop through each image and corresponding points array
            output_image = AlignClass.draw_circles_on_image(points_array, image, color, radius, thickness)  # Draw circles on image
            output_images.append(output_image)  # Append output image to list

        return output_images  # Return list of output images with circles drawn on them


    ### Function: test_align_bounding_boxes_predicted_points ###
    @staticmethod
    def test_align_bounding_boxes_predicted_points(frames, flag_plot=False):
        """
        Test function for aligning bounding boxes using predicted points.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
        """
        ### Extract Reference Frame and Bounding Box: ###
        ref_frame = frames[0]  # Reference frame
        bbox_XYWH = draw_bounding_box(ref_frame)  # Draw bounding box
        bbox_XYXY = BB_convert_notation_XYWH_to_XYXY(bbox_XYWH)

        ### Get Points On BB: ###
        predicted_points = generate_points_in_BB(bbox_XYXY)  # Generate predicted points
        # image_with_points = AlignClass.draw_circles_on_image(predicted_points, ref_frame, color=(0, 255, 0), radius=5, thickness=2)
        # plt.imshow(image_with_points); plt.show()

        # ### Track points using optical flow: ###
        # tracked_points = AlignClass.track_points_using_optical_flow_opencv(frames, predicted_points)  # Track points across frames
        ### Track points using opencv tracker: ###
        bounding_boxes_array = AlignClass.track_object_using_opencv_tracker(bbox_XYWH, frames)

        ### Get Tracked Points From BB Locations: ###
        predicted_points = generate_points_in_BB(bounding_boxes_array)

        ### Show predicted points (DEBUG): ###
        frames_with_points = AlignClass.draw_circles_on_images(predicted_points, frames, color=(0, 255, 0), radius=5, thickness=2)
        # frames_with_points_numpy = list_to_numpy(frames_with_points)
        # imshow_video(frames_with_points_numpy, FPS=25, frame_stride=2)

        ### Align Bounding Boxes and Calculate Averaged Box: ###
        predicted_points_array = list_to_numpy(predicted_points)
        aligned_frames, aligned_frames_crops, averaged_aligned_frame, averaged_aligned_frame_crops = AlignClass.align_frames_using_predicted_points_per_frame_optical_flow(
                                                                                                                    frames,
                                                                                                                    ref_frame,
                                                                                                                    predicted_points_array,
                                                                                                                    method='interpolation')  # 'interpolation', 'tracker', 'predicted'.

        ### Display Aligned Bounding Boxes: ###
        if flag_plot:
            aligned_boxes_numpy = list_to_numpy(aligned_frames_crops)
            imshow_video(aligned_boxes_numpy, FPS=25, frame_stride=1)
            imshow_np(averaged_aligned_frame_crops, title='Averaged Aligned Frame')

    ### Function: test_align_frames_to_reference_using_given_optical_flow ###
    @staticmethod
    def test_align_frames_to_reference_using_given_optical_flow(frames):
        """
        Test function for aligning frames using optical flow.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
        """
        ### Extract Reference Frame and Calculate Optical Flows: ###
        ref_frame = frames[0]  # Reference frame
        optical_flows = AlignClass.calculate_optical_flow_pairwise_opencv(frames)  # Calculate optical flows

        ### Align Frames and Calculate Averaged Frame: ###
        aligned_frames, avg_frame = AlignClass.align_frames_to_reference_using_given_optical_flow(frames, ref_frame, optical_flows)  # Align frames

        ### Display Aligned Frames: ###
        for i, frame in enumerate(aligned_frames):  # Loop through each aligned frame
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Display aligned frame
            plt.title(f'Aligned Frame {i}')  # Title for the frame
            plt.show()  # Show frame

        ### Display Averaged Frame: ###
        plt.imshow(cv2.cvtColor(avg_frame, cv2.COLOR_BGR2RGB))  # Display averaged frame
        plt.title('Averaged Frame')  # Title for the averaged frame
        plt.show()  # Show averaged frame

    ### Function: test_align_frames_translation ###
    @staticmethod
    def test_align_frames_translation(frames, flag_plot=False):
        """
        Test function for aligning frames using translation.

        Args:
            frames (list): List of frames (numpy arrays). Each frame shape is [H, W, C].
        """
        ### Extract Reference Frame and Generate Translation Deltas: ###
        ref_frame = frames[0]  # Reference frame
        frames_translated = AlignClass.apply_random_translation_to_images(frames, max_translation=10)

        ### Perform CCC To Align Frames: ###
        aligned_frames, sy_torch, sx_torch = AlignClass.align_frames_to_reference_CCC(input_frames=frames_translated, reference_frame=frames_translated[0], device='cuda')

        ### Align Frames and Calculate Averaged Frame: ###
        delta_H = sy_torch.cpu().numpy()  # Convert sy_torch to float
        delta_W = sx_torch.cpu().numpy()  # Convert sy_torch to float
        aligned_frames, avg_frame = AlignClass.align_frames_to_reference_using_given_shifts(frames_translated, ref_frame, -delta_H, -delta_W)  # Align frames

        ### Display Aligned Frames: ###
        if flag_plot:
            imshow_video(aligned_frames, FPS=25, frame_stride=1)
            imshow_np(avg_frame, title='Averaged Aligned Frame')  # Display averaged frame

    @staticmethod
    def align_and_average_frames_using_ECC(frames, reference_frame=0, input_method=None, user_input=None):
        #TODO: implement ECC alignment and averaging here. don't forget expanding to super resolution
        bla = 1
        return stabilized_frames, average_frame

    @staticmethod
    def align_and_average_frames_using_SCC(frames, reference_frame=0, input_method=None, user_input=None):
        # TODO: implement SCC alignment and averaging here. don't forget expanding to super resolution
        bla = 1
        return stabilized_frames, average_frame

    @staticmethod
    def align_and_average_frames_using_FeatureBased(frames, reference_frame=None, input_method=None, user_input=None):
        #[input_method] = 'BB', 'polygon', or 'segmentation'
        # TODO: implement FeatureBased alignment and averaging here. don't forget expanding to super resolution

        ### Get Reference Frame: ###
        if reference_frame is None:
            reference_frame = frames[0]  # If reference_frame is not provided, use the first frame
        H,W = reference_frame.shape[0:2]  # Get frame dimensions

        ### Get Region From User: ###
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, grid_points, flag_no_input = user_input_to_all_input_types(
            user_input,
            input_method=input_method,
            input_shape=(H, W))

        ### Align Frames and Calculate Averaged Frame And Homography: ###
        #TODO: enable inserting into the function the necessary inputs like BB, polygon and segmentation mask to only take into account those areas
        aligned_frames, homography_matrices_list, last_frame_features = stabilize_frames_FeatureBased(frames,
                                                                                                      reference_frame=frames[0],
                                                                                                      flag_output_array_form='list',
                                                                                                      number_of_features_for_ORB=2500,
                                                                                                      max_iterations=5,
                                                                                                      inlier_threshold=2.5,
                                                                                                      c=4.685,
                                                                                                      interpolation='nearest',
                                                                                                      flag_registration_algorithm='regular',
                                                                                                      flag_perform_interpolation=True,
                                                                                                      last_frame_features=[None, None],
                                                                                                      flag_downsample_frames=False,
                                                                                                      binning_factor=1,
                                                                                                      flag_RGB2BW=False,
                                                                                                      flag_matching_crosscheck=False)
        aligned_frames_numpy = list_to_numpy(aligned_frames)
        avg_frame = np.mean(aligned_frames_numpy, axis=0).clip(0, 255).astype(np.uint8)  # Calculate averaged frame
        # imshow_video(aligned_frames, FPS=25, frame_stride=1)  #

        ### Align Crops and Calculate Averaged Crop: ###
        aligned_crops, average_crop = AlignClass.align_frame_crops_using_given_homographies(frames,
                                                                                        reference_frame,
                                                                                        initial_BB_XYXY,
                                                                                        homography_matrices_list)  # Align crops

        # ### Display Aligned Frames: ###
        # if flag_plot:
        #     aligned_crops_numpy = list_to_numpy(aligned_crops)  # Convert aligned crops to numpy
        #     imshow_video(aligned_crops_numpy, FPS=25, frame_stride=1)  #
        #
        #     ### Display Averaged Frame: ###
        #     plt.figure()  # Create figure
        #     plt.imshow(cv2.cvtColor(avg_crop, cv2.COLOR_BGR2RGB))  # Display averaged frame
        #     plt.title('Averaged Frame')  # Title for the averaged frame
        #     plt.show()  # Show averaged frame

        return aligned_crops, average_crop

    @staticmethod
    def align_and_average_frames_using_OpticalFlow(frames,
                                                   reference_frame=None,
                                                   input_method=None,
                                                   user_input=None,
                                                   optical_flow_method='LucasKanade',
                                                   flag_use_optical_flow_to_homography=False):

        ### Get Reference Frame: ###
        if reference_frame is None:
            reference_frame = frames[0]  # If reference_frame is not provided, use the first frame
        H, W = reference_frame.shape[0:2]  # Get frame dimensions

        ### Get Region From User: ###
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, grid_points, flag_no_input = user_input_to_all_input_types(
            user_input,
            input_method=input_method,
            input_shape=(H, W))

        ### Crop to multiple of 8: ###
        (H_new, W_new) = AlignClass.adjust_shape_to_multiple(frames[0].shape[0:2], multiple=8, method='crop')
        aligned_crops = [crop_tensor(frames[i], (H_new, W_new)) for i in np.arange(len(frames))]

        ### If size is smaller then (224,224) to resize it: ###
        min_size = 224
        aspect_ratio = (W_new / H_new)
        if aspect_ratio < 1:
            W_new = int(min_size * aspect_ratio)
            W_new = (W_new // 8) * 8
            H_new = min_size
            H_new = (H_new // 8) * 8
        else:
            H_new = min_size
            W_new = int(H_new * aspect_ratio)
            W_new = (W_new // 8) * 8
        frames = [cv2.resize(frames[i], (W_new, H_new), interpolation=cv2.INTER_LINEAR) for i in
                         np.arange(len(frames))]

        ### Get Optical Flow: ###
        #TODO: implement optical flow calculation here
        #TODO: add possibility of converting optical flow to homographies
        optical_flow, frames_list, reference_frame = AlignClass.calculate_optical_flow_raft_reference(frames, reference_frame, model_path=None, return_numpy=True)

        ### Use Optical Flow to Align Frames: ###
        #TODO: add scale_factor when aligning frames
        #TODO: add using optical flow method to calculate homographies and using homographies
        if flag_use_optical_flow_to_homography == False:
            aligned_crops, average_crop = AlignClass.align_frames_crops_to_reference_using_given_optical_flow(frames, reference_frame, optical_flow, initial_BB_XYXY)
        else:
            ### Use Optical Flow To Fit Homography And Use It To Align Crops: ###
            aligned_crops, average_crop = AlignClass.align_frames_crops_using_optical_flow_to_homography_matrix(frames, reference_frame, optical_flow, initial_BB_XYXY)

        return aligned_crops, average_crop
    
    @staticmethod
    def align_frames_using_predicted_points_per_frame(frames, reference_frame, predicted_points_list, alignment_method='optical_flow', homography_method='ransac'):
        # [alignment_method] = 'optical_flow', 'homography'
        if alignment_method == 'optical_flow':
            (aligned_frames,
             aligned_crops,
             averaged_aligned_frame,
             average_crop) = AlignClass.align_frames_using_predicted_points_per_frame_optical_flow(
                frames,
                reference_frame,
                predicted_points_list,
                method='optical_flow')
        elif alignment_method == 'homography':
            aligned_crops, average_crop = AlignClass.align_frames_using_predicted_points_per_frame_homography(frames,
                                                                                                              reference_frame,
                                                                                                              predicted_points_list,
                                                                                                              homography_method=homography_method)
        return aligned_crops, average_crop
    
    @staticmethod
    def align_and_average_frames_using_CoTracker(frames, 
                                                 reference_frame=None,
                                                 input_method=None, 
                                                 user_input=None, 
                                                 alignment_method='optical_flow', 
                                                 post_process_method=None, 
                                                 homography_method='ransac'):
        # [alignment_method] = 'homography', 'optical_flow'
        # [post_process_method] = 'contrast_optimization_homography', 'contrast_optimization_optical_flow'
        # [homography_method] = 'ransac', 'least_squares', 'weighted_least_squares', 'iterative_reweighted_least_squares'
        
        ### Get Reference Frame: ###
        if reference_frame is None:
            reference_frame = frames[0]  # If reference_frame is not provided, use the first frame
        H, W = reference_frame.shape[0:2]  # Get frame dimensions

        ### Get Region From User: ###
        initial_BB_XYXY, initial_polygon_points, initial_segmentation_mask, initial_grid_points, flag_no_input = user_input_to_all_input_types(
            user_input,
            input_method=input_method,
            input_shape=(H, W))

        ### Align Frames Using CoTracker: ###
        initial_points_list = initial_grid_points.tolist() #TODO: need to check this .tolist() because i think initial_grid_points is already a list
        frames_array = np.concatenate([numpy_unsqueeze(frames[i], 0) for i in range(len(frames))], axis=0)
        pred_tracks, pred_visibility, frames_array = track_points_in_video(frames_array,
                                                                           points=initial_points_list,
                                                                           grid_size=5,
                                                                           interactive=False,
                                                                           fill_all=False)
        predicted_points_list = [pred_tracks[i] for i in range(len(pred_tracks))]
        # imshow_video(frames_output, FPS=5)

        ### Perform Final Alignment (Post Processing): ###
        aligned_crops, average_crop = AlignClass.align_frames_using_predicted_points_per_frame(frames, 
                                                                                               reference_frame, 
                                                                                               predicted_points_list, 
                                                                                               alignment_method=alignment_method, 
                                                                                               homography_method=homography_method)
        ### Perform Post Processing (Contrast Optimization): ###
        if post_process_method is not None:
            aligned_crops, average_crop = AlignClass.optimize_images_for_maximym_contrast(aligned_crops,
                                                     alignment_method=post_process_method,
                                                     max_iterations=500,
                                                     learning_rate=0.0001,
                                                     weight_regularizer=0.0,
                                                     contrast_method='variance',  # 'default', 'variance'
                                                     device='cuda',
                                                     uniform_weights=True,
                                                     optical_flow_limit=1, 
                                                     size_fraction=0.7)
        
        return aligned_crops, average_crop
    
    @staticmethod
    def optimize_images_for_maximym_contrast(aligned_crops,
                                             alignment_method='contrast_optimization_optical_flow',
                                             max_iterations=500,
                                             learning_rate=0.0001,
                                             weight_regularizer=0.0,
                                             contrast_method='variance',  # 'default', 'variance'
                                             device='cuda',
                                             uniform_weights=True,
                                             optical_flow_limit=1,
                                             size_fraction=0.7):

        if alignment_method is not None:
            if alignment_method == 'contrast_optimization_optical_flow':
                optimized_flows, optimized_flows_intensities, optimized_weights, average_crop, aligned_crops = AlignClass.optimize_optical_flow_and_weights_torch(
                    aligned_crops,
                    max_iterations=500,
                    learning_rate=0.0001,
                    weight_regularizer=0.0,
                    contrast_method='variance',  # 'default', 'variance'
                    device='cuda',
                    uniform_weights=True,
                    optical_flow_limit=1)
                # imshow_video(list_to_numpy(warped_images) / 255, FPS=5, frame_stride=1, video_title='warped images')
                # imshow_np(weighted_average / 255)
            if alignment_method == 'contrast_optimization_homography':
                ### Optimize Homography Matrices For Maximum Contrast: ###
                optimized_homographies, optimized_weights, average_crop, aligned_crops = AlignClass.optimize_weights_and_homographies_torch(
                    aligned_crops,
                    max_iterations=300,
                    learning_rate=0.00003,
                    weight_regularizer=0.01,
                    contrast_method='variance',
                    device='cuda',
                    uniform_weights=True,
                    size_fraction=0.7)

            return aligned_crops, average_crop
        
        
    @staticmethod
    def align_frames_using_predicted_points_per_frame_homography(frames, reference_frame, predicted_points_list, homography_method='ransac', initial_BB_XYXY=None):
        # [homography_method] = 'ransac', 'least_squares', 'weighted_least_squares', 'iterative_reweighted_least_squares'
        ### Take Care Of Initial Inputs: ###
        initial_points = predicted_points_list[0]  # Get initial points
        if initial_BB_XYXY is None:
            H, W = reference_frame.shape[0:2]  # Get frame dimensions
            initial_BB_XYXY, segmentation_mask = points_to_bounding_box_and_mask(initial_points, (H,W))

        ### Get Homography From Current Points And Reference Points: ###
        H_list = AlignClass.find_homographies_from_points_list(predicted_points_list,
                                                               initial_points,
                                                               method=homography_method, # 'ransac', 'least_squares', 'weighted_least_squares', 'iterative_reweighted_least_squares'
                                                               max_iterations=2000,
                                                               inlier_threshold=3)

        ### Get Points After Applying Homography: ###
        predicted_points_list_after_homography = AlignClass.apply_homographies_to_list_of_points_arrays(predicted_points_list, H_list)


        ### Get and Align Crops Using Homographies: ###
        aligned_crops, averaged_crop = AlignClass.align_frame_crops_using_given_homographies(frames, frames[0],
                                                                                             initial_BB_XYXY,
                                                                                             H_list, size_factor=2)

        return aligned_crops, averaged_crop

    @staticmethod
    def test_align_crops_using_co_tracker(frames, 
                                          flag_plot=False, 
                                          flag_pre_align_with_tracker=False, 
                                          flag_plot_points_in_BB=False, 
                                          flag_plot_points_in_polygon=False, 
                                          flag_plot_points_predicted_from_cotracker=False, 
                                          flag_plot_points_predicted_from_optical_flow=False,
                                          flag_plot_points_predicted_from_homography=False):
        #(1). initialize BB, set points for tracking, track points, use optical flow to align and average
        #(2). initialize BB, set points for tracking, track points, find homography from points and align and average
        #(2). initialize BB, set points for tracking, track points, use optical flow interpolated between points and align and average
        #(3). initialize BB, set points for tracking, track points, find homography from points and align, use optical flow and weight optimization for max contrast
        #(4). initialize BB, set points for tracking, track points, find homography from points and align, use homography and weight optimization for max contrast
        #(5). initialize BB, draw polygon, set points for tracking, find homography from points and align and average
        #(5). initialize BB, draw polygon, set points for tracking, find homography from points and align and average, u

        ### Extract Reference Frame and Bounding Box: ###
        start_frame = 35
        frames = frames[start_frame:start_frame+15]
        ref_frame = frames[0]  # Reference frame
        initial_bbox_XYWH = draw_bounding_box(ref_frame)  # Draw bounding box
        initial_bbox_XYXY = BB_convert_notation_XYWH_to_XYXY(initial_bbox_XYWH)
        X0, Y0, X1, Y1 = initial_bbox_XYXY

        ### Get Points Grid On BB (For Following): ###
        initial_points = generate_points_in_BB(initial_bbox_XYXY)  # Generate predicted points
        if flag_plot_points_in_BB:
            image_with_points = AlignClass.draw_circles_on_image(initial_points, ref_frame, color=(0, 255, 0), radius=5, thickness=2)
            plt.imshow(image_with_points); plt.show()
        
        ### Draw Polygon Inside Initial Bounding Box: ###
        ref_frame_crop = ref_frame[Y0:Y1, X0:X1]  #
        polygon_points = select_points_and_build_polygon_opencv(ref_frame_crop.copy())
        ref_frame_crop_with_polygon = AlignClass.show_polygon_on_image(ref_frame_crop, polygon_points)

        ### Straighten Polygon To Rectangle: ###
        rectified_image, transform_matrix = AlignClass.straighten_polygon_to_rectangle(ref_frame_crop, polygon_points)
        # imshow_np(rectified_image)

        ### Get Points Grid In Polygon (For Following): ###
        points_in_polygon = generate_points_in_polygon(polygon_points, grid_size=5)
        if flag_plot_points_in_polygon:
            image_with_points = AlignClass.draw_circles_on_image(points_in_polygon, ref_frame_crop, color=(0, 255, 0), radius=2, thickness=2)
            plt.imshow(image_with_points); plt.show()        

        ### Track points using opencv tracker: ###
        if flag_pre_align_with_tracker:
            bounding_boxes_array = AlignClass.track_object_using_opencv_tracker(initial_bbox_XYWH, frames)

        ### Track points using co_tracker!!!!!: ###
        initial_points_list = initial_points.tolist()
        frames_array = np.concatenate([numpy_unsqueeze(frames[i],0) for i in range(len(frames))], axis=0)
        pred_tracks, pred_visibility, frames_array = track_points_in_video(frames_array, points=initial_points_list, grid_size=5, interactive=False, fill_all=False)
        predicted_points_list = [pred_tracks[i] for i in range(len(pred_tracks))]
        # imshow_video(frames_output, FPS=5)

        ### Plot Points With Arrows: ###
        if flag_plot_points_predicted_from_cotracker:
            ax = None
            for i in np.arange(len(predicted_points_list)):
                ax = AlignClass.plot_points_with_arrows(initial_points, predicted_points_list[i], ax)

        ### Show Points As Tracked With Co-Tracker: ###
        if flag_plot_points_predicted_from_cotracker:
            frames_with_points_list = []
            for i in np.arange(len(predicted_points_list)):
                current_frame_points = predicted_points_list[i]
                current_frame = frames[i]
                image_with_points = AlignClass.draw_circles_on_image(current_frame_points, current_frame, color=(0, 255, 0), radius=5, thickness=2)
                frames_with_points_list.append(image_with_points)
            frames_with_points_numpy = list_to_numpy(frames_with_points_list)
            # imshow_video(frames_with_points_numpy, FPS=1, frame_stride=1, video_title='tracked points with co-tracker')

        ### Get Homography From Current Points And Reference Points: ###
        H_list = AlignClass.find_homographies_from_points_list(predicted_points_list,
                                                               initial_points,
                                                               method='ransac', #'ransac', 'least_squares', 'weighted_least_squares', 'iterative_reweighted_least_squares'
                                                               max_iterations=2000,
                                                               inlier_threshold=3)

        ### Get Points After Applying Homography: ###
        predicted_points_list_after_homography = AlignClass.apply_homographies_to_list_of_points_arrays(predicted_points_list, H_list)

        ### Plot Points With Arrows After Homographies: ###
        if flag_plot_points_predicted_from_homography:
            ax = None
            for i in np.arange(len(predicted_points_list_after_homography)):
                ax = AlignClass.plot_points_with_arrows(initial_points, predicted_points_list_after_homography[i], ax)

        ### Get and Align Crops Using Homographies: ###
        aligned_crops, averaged_crop = AlignClass.align_frame_crops_using_given_homographies(frames, frames[0], initial_bbox_XYXY, H_list, size_factor=2)
        # imshow_video(list_to_numpy(aligned_crops), FPS=2, frame_stride=1)
        # imshow_np(aligned_crops[-2])
        # imshow_video(list_to_numpy(frames), FPS=5, frame_stride=1)
        # imshow_np(averaged_crop, title='Averaged Crop')

        ### Optimize Optical Flow and Weights For Maximum Contrast: ###
        optimized_flows, optimized_flows_intensities, optimized_weights, weighted_average, warped_images = AlignClass.optimize_optical_flow_and_weights_torch(
            aligned_crops,
            max_iterations=500,
            learning_rate=0.0001,
            weight_regularizer=0.0,
            contrast_method='variance',  #'default', 'variance'
            device='cuda',
            uniform_weights=True,
            optical_flow_limit=1)
        imshow_video(list_to_numpy(warped_images)/255, FPS=5, frame_stride=1, video_title='warped images')
        imshow_np(weighted_average/255)

        ### Optimize Homography Matrices For Maximum Contrast: ###
        optimized_homographies, optimized_weights, weighted_average, warped_images = AlignClass.optimize_weights_and_homographies_torch(
            aligned_crops,
            max_iterations=300,
            learning_rate=0.00003,
            weight_regularizer=0.01,
            contrast_method='variance',
            device='cuda',
            uniform_weights=True,
            size_fraction=1)
        imshow_video(list_to_numpy(aligned_crops), FPS=2, frame_stride=1, video_title='warped images')
        imshow_video(list_to_numpy(warped_images) / 255, FPS=2, frame_stride=1, video_title='warped images')
        imshow_np(weighted_average / 255)

        ### Align Bounding Boxes (Using Interpolated Optical Flow Between Predicted Points) and Calculate Averaged Box: ###
        predicted_points_array = list_to_numpy(predicted_points_list)
        aligned_frames, aligned_frames_crops, averaged_aligned_frame, averaged_aligned_frame_crops = AlignClass.align_frames_using_predicted_points_per_frame_optical_flow(
            frames,
            ref_frame,
            predicted_points_array,
            method='interpolation')  # 'interpolation', 'tracker', 'predicted'.

        ### Display Aligned Bounding Boxes: ###
        if flag_plot:
            aligned_boxes_numpy = list_to_numpy(aligned_frames_crops)
            imshow_video(aligned_boxes_numpy, FPS=5, frame_stride=1, video_title='aligned crops')
            imshow_np(averaged_aligned_frame_crops, title='Averaged Aligned Frame')
            imshow_np(frames[0][initial_bbox_XYXY[1]:initial_bbox_XYXY[3], initial_bbox_XYXY[0]:initial_bbox_XYXY[2]], title='Initial Crop')

    @staticmethod
    def straighten_polygon_to_rectangle(image, polygon):
        """
        Straightens a 4-vertex polygon to a rectangle aligned with the x and y Cartesian axes and returns the warped image.

        Inputs:
        - image: numpy array of shape [H, W, C], the original image containing the polygon.
        - polygon: list of tuples, each tuple contains (x, y) coordinates of the 4 vertices of the polygon.

        Outputs:
        - rectified_image: numpy array of shape [H', W', C], the straightened rectangle image.
        - transform_matrix: numpy array of shape [3, 3], the transformation matrix used to straighten the polygon.
        """
        ### This Is The Code Block: ###
        assert len(polygon) == 4, "Polygon must have exactly 4 vertices"  # Ensure the polygon has 4 vertices

        ### Convert polygon to numpy array ###
        polygon = np.array(polygon, dtype=np.float32)  # Convert polygon to numpy array of type float32

        ### Order the points in the polygon ###
        rect = AlignClass.order_points(polygon)  # Order the points in the polygon

        ### Compute the width and height of the new image ###
        width_a = np.linalg.norm(rect[0] - rect[1])  # Distance between top-left and top-right points
        width_b = np.linalg.norm(rect[2] - rect[3])  # Distance between bottom-left and bottom-right points
        max_width = max(int(width_a), int(width_b))  # Maximum width

        height_a = np.linalg.norm(rect[0] - rect[3])  # Distance between top-left and bottom-left points
        height_b = np.linalg.norm(rect[1] - rect[2])  # Distance between top-right and bottom-right points
        max_height = max(int(height_a), int(height_b))  # Maximum height

        ### Destination points for the transformed image ###
        dst = np.array([
            [0, 0],  # Top-left
            [max_width - 1, 0],  # Top-right
            [max_width - 1, max_height - 1],  # Bottom-right
            [0, max_height - 1]  # Bottom-left
        ], dtype=np.float32)

        ### Compute the perspective transform matrix ###
        transform_matrix = cv2.getPerspectiveTransform(rect, dst)  # Compute the perspective transform matrix

        ### Warp the image to obtain the rectified image ###
        rectified_image = cv2.warpPerspective(image, transform_matrix,
                                              (max_width, max_height))  # Apply the perspective transformation

        return rectified_image, transform_matrix  # Return the rectified image and the transformation matrix

    ### This Is The Function Description: ###
    @staticmethod
    def order_points(pts):
        """
        Orders the points in the polygon to a consistent order: top-left, top-right, bottom-right, bottom-left.

        Inputs:
        - pts: numpy array of shape [4, 2], the points of the polygon.

        Outputs:
        - rect: numpy array of shape [4, 2], the ordered points.
        """
        ### This Is The Code Block: ###
        rect = np.zeros((4, 2), dtype=np.float32)  # Initialize the ordered points array
        s = pts.sum(axis=1)  # Compute the sum of the x and y coordinates
        rect[0] = pts[np.argmin(s)]  # Top-left point has the smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right point has the largest sum

        diff = np.diff(pts, axis=1)  # Compute the difference between the x and y coordinates
        rect[1] = pts[np.argmin(diff)]  # Top-right point has the smallest difference
        rect[3] = pts[np.argmax(diff)]  # Bottom-left point has the largest difference

        return rect  # Return the ordered points

    @staticmethod
    def show_polygon_on_image(image, polygon):
        """
        Displays the image with the closed polygon overlay.

        Inputs:
        - image: numpy array of shape [H, W, C], the image on which the polygon will be displayed.
        - polygon: list of tuples, each tuple contains (x, y) coordinates of the points forming the closed polygon.

        Outputs:
        - None
        """
        ### This Is The Code Block: ###
        fig, ax = plt.subplots()  # Create a figure and axis
        ax.imshow(image)  # Display the image

        ### Copy the polygon to avoid modifying the original ###
        polygon_copy = copy.deepcopy(polygon)  # Create a deep copy of the polygon

        ### Ensure the polygon is closed ###
        if len(polygon_copy) > 0 and polygon_copy[0] != polygon_copy[-1]:
            polygon_copy.append(polygon_copy[0])  # Add the first point to the end if it's not already closed

        ### Draw the polygon ###
        ax.plot([p[0] for p in polygon_copy], [p[1] for p in polygon_copy], 'r-')  # Draw the polygon lines
        ax.plot([p[0] for p in polygon_copy], [p[1] for p in polygon_copy], 'ro')  # Draw the points of the polygon

        ax.set_title('Polygon Overlay')  # Set the title of the figure
        plt.show()  # Display the figure

    ### This Is The Function Description: ###
    @staticmethod
    def compute_weighted_average_torch(images, weights):
        """
        Computes the weighted average of a list of images using PyTorch.

        Inputs:
        - images: torch tensor of shape [N, C, H, W]
        - weights: torch tensor of shape [N]

        Outputs:
        - weighted_average: torch tensor of shape [C, H, W]
        """
        ### This Is The Code Block: ###
        weights_expanded = weights.view(-1, 1, 1, 1)  # Reshape weights to [N, 1, 1, 1]
        weighted_images = images * weights_expanded  # Apply weights to images
        weighted_sum = weighted_images.sum(dim=0)  # Sum weighted images along the first dimension
        weights_sum = weights.sum()  # Sum of weights
        weighted_average = weighted_sum / weights_sum  # Compute weighted average
        return weighted_average  # Return weighted average

    ### This Is The Function Description: ###
    @staticmethod
    def compute_contrast_torch(image, method='default'):
        """
        Computes the contrast of an image using PyTorch.

        Inputs:
        - image: torch tensor of shape [C, H, W]
        - method: str, method to compute contrast ('default' or 'variance')

        Outputs:
        - contrast: float
        """
        ### This Is The Code Block: ###
        C,H,W = image.shape
        if method == 'default':
            blurred = F.conv2d(image.unsqueeze(0), weight=AlignClass.gaussian_kernel(3, 1, C).to(image.device),
                               padding=1).squeeze(0)  # Apply Gaussian filter
            return torch.mean(torch.abs(image - blurred))  # Compute mean absolute difference

        elif method == 'variance':
            return torch.var(image) / torch.mean(image)  # Compute variance divided by mean

    ### This Is The Function Description: ###
    @staticmethod
    def gaussian_kernel(size, sigma, C=1):
        """
        Creates a Gaussian kernel using PyTorch.

        Inputs:
        - size: int, size of the kernel
        - sigma: float, standard deviation of the Gaussian

        Outputs:
        - kernel: torch tensor of shape [1, 1, size, size]
        """
        ### This Is The Code Block: ###
        x = torch.arange(-size // 2 + 1, size // 2 + 1).float().view(1, -1)  # Create a range of values for x
        y = torch.arange(-size // 2 + 1, size // 2 + 1).float().view(-1, 1)  # Create a range of values for y
        kernel = torch.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)  # Compute the Gaussian function
        kernel /= kernel.sum()  # Normalize the kernel
        return kernel.view(1, 1, size, size).repeat(1,C,1,1)  # Reshape to [1, 1, size, size]

    ### This Is The Function Description: ###
    @staticmethod
    def optimize_optical_flow_and_weights_torch(images,
                                                max_iterations=100,
                                                learning_rate=0.01,
                                                weight_regularizer=0.01,
                                                contrast_method='default',
                                                device='cpu',
                                                uniform_weights=False,
                                                optical_flow_limit=1):
        """
        Optimizes optical flow maps and pixel weights to minimize the contrast of the weighted average of the images using PyTorch.

        Inputs:
        - images: list of numpy arrays, each of shape [C, H, W]
        - max_iterations: int, maximum number of iterations for the optimization
        - learning_rate: float, learning rate for the optimization
        - weight_regularizer: float, regularization term for the pixel weights
        - contrast_method: str, method to compute contrast ('default' or 'variance')
        - device: str, device to run the optimization on ('cpu' or 'cuda')
        - uniform_weights: bool, if True, constrain weights to be uniform across each image

        Outputs:
        - optimized_flows: list of numpy arrays, each of shape [H, W, 2], the optimized optical flow maps
        - optimized_weights: numpy array of shape [N] or [C, H, W], the optimized pixel weights
        - weighted_average: numpy array of shape [C, H, W], the final weighted average image
        - warped_images: list of numpy arrays, each of shape [C, H, W], the warped images
        """
        ### This Is The Code Block: ###
        device = torch.device(device)  # Set the device
        H, W, C = images[0].shape  # Get dimensions of the images
        num_images = len(images)  # Number of images

        ### Convert images to torch tensors ###
        images_torch = torch.stack([numpy_to_torch(img) for img in images]).to(device)  # Shape [N, C, H, W]

        ### Initialize optical flow maps and weights ###
        flows = torch.nn.Parameter(torch.zeros((num_images, 2, H, W), dtype=torch.float32, device=device))  # Initialize flows to zero
        if uniform_weights:
            weights = torch.nn.Parameter(torch.ones(num_images, dtype=torch.float32, device=device) / num_images)  # Uniform weights per image
        else:
            weights = torch.nn.Parameter(torch.ones((C, H, W), dtype=torch.float32, device=device) / num_images)  # Initialize weights to uniform

        ### Create meshgrid once ###
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))  # Create meshgrid
        grid = torch.stack((grid_x, grid_y), dim=0).float()  # Shape [2, H, W]

        ### Create optimizer ###
        optimizer = optim.Adam([flows, weights], lr=learning_rate)  # Create Adam optimizer

        ### Optimization loop ###
        for iteration in range(max_iterations):  ### Looping Over Indices: ###
            optimizer.zero_grad()  # Zero the gradients

            ### Apply optical flows to images ###
            flow_grid = grid.unsqueeze(0) + flows  # Shape [N, 2, H, W]
            flow_grid = flow_grid.permute(0, 2, 3, 1)  # Shape [N, H, W, 2]
            flow_grid[..., 0] = (flow_grid[..., 0] / (W - 1) * 2) - 1  # Normalize flow grid x to [-1, 1]
            flow_grid[..., 1] = (flow_grid[..., 1] / (H - 1) * 2) - 1  # Normalize flow grid y to [-1, 1]
            warped_images = F.grid_sample(images_torch, flow_grid, align_corners=True)  # Shape [N, C, H, W]

            ### Compute weighted average ###
            weighted_average = AlignClass.compute_weighted_average_torch(warped_images, weights)  # Compute weighted average of warped images

            ### Compute contrast ###
            contrast = AlignClass.compute_contrast_torch(weighted_average,
                                                         method=contrast_method)  # Compute contrast of the weighted average image

            ### Add regularization to weights ###
            regularization = weight_regularizer * torch.sum(weights ** 2)  # Compute regularization term
            loss = -contrast + regularization  # Compute total loss (negative contrast to minimize)

            ### Backpropagation ###
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the parameters

            ### Clamp optical flow to [-2, 2] ###
            with torch.no_grad():  # Ensure that the clamp operation does not track gradients
                flows.data = torch.clamp(flows.data, -optical_flow_limit, optical_flow_limit)  # Clamp the flow values

            ### Clamp weights to be non-negative and within [0, 1] ###
            with torch.no_grad():  # Ensure that the clamp operation does not track gradients
                weights.data = torch.clamp(weights.data, 0, 1)  # Clamp the weight values

        ### Convert flows and weights back to numpy arrays ###
        optimized_flows = flows.detach().cpu().numpy().transpose(0, 3, 1, 2)  # Convert flows to numpy arrays, shape [N, H, W, 2]
        optimized_flows = optimized_flows.transpose(0, 2, 3, 1)  # Convert
        optimized_flows_list = [optimized_flows[i] for i in range(len(optimized_flows))]
        optimized_flows_intensities = AlignClass.get_optical_flow_intensity(optimized_flows_list)  # Convert flow intensity to numpy arrays, shape [N, H, W]
        optimized_weights = weights.detach().cpu().numpy()  # Convert weights to numpy arrays
        weighted_average = torch_to_numpy(weighted_average)  # Convert weighted average to numpy arrays, shape [C, H, W]
        warped_images = torch_to_numpy(warped_images)  # Convert warped images to numpy arrays, shape [N, C, H, W]
        warped_images = [warped_images[i] for i in range(len(warped_images))]
        return optimized_flows, optimized_flows_intensities, optimized_weights, weighted_average, warped_images  # Return optimized flows, weights, weighted average, and warped images

    @staticmethod
    def optimize_weights_and_homographies_torch(frames,
                                                max_iterations=100,
                                                learning_rate=0.01,
                                                weight_regularizer=0.01,
                                                contrast_method='default',
                                                device='cpu',
                                                uniform_weights=False,
                                                size_fraction=0.5):
        """
        Optimizes the weights and homography matrices for each frame to minimize the contrast of the weighted average of the images using PyTorch.

        Inputs:
        - frames: list of numpy arrays, each of shape [C, H, W]
        - max_iterations: int, maximum number of iterations for the optimization
        - learning_rate: float, learning rate for the optimization
        - weight_regularizer: float, regularization term for the pixel weights
        - contrast_method: str, method to compute contrast ('default' or 'variance')
        - device: str, device to run the optimization on ('cpu' or 'cuda')
        - uniform_weights: bool, if True, constrain weights to be uniform across each image
        - size_fraction: float, the fraction of the center crop size relative to the original frame size

        Outputs:
        - optimized_homographies: list of numpy arrays, each of shape [3, 3], the optimized homography matrices
        - optimized_weights: numpy array of shape [N] or [C, H, W], the optimized pixel weights
        - weighted_average: numpy array of shape [C, H, W], the final weighted average image
        - warped_images: list of numpy arrays, each of shape [C, H, W], the warped images
        """
        ### This Is The Code Block: ###
        device = torch.device(device)  # Set the device
        H, W, C = frames[0].shape  # Get dimensions of the frames
        num_images = len(frames)  # Number of frames

        ### Convert frames to torch tensors ###
        frames_torch = torch.stack([numpy_to_torch(frame) for frame in frames]).to(device)  # Shape [N, C, H, W]

        ### Initialize homography matrices and weights ###
        homographies = torch.nn.Parameter(torch.eye(3).unsqueeze(0).repeat(num_images, 1, 1).to(device))  # Initialize homographies to identity
        if uniform_weights:
            weights = torch.nn.Parameter(
                torch.ones(num_images, dtype=torch.float32, device=device) / num_images)  # Uniform weights per image
        else:
            weights = torch.nn.Parameter(
                torch.ones((C, H, W), dtype=torch.float32, device=device) / num_images)  # Initialize weights to uniform

        ### Calculate center crop coordinates ###
        crop_width = int(W * size_fraction)  # Calculate crop width
        crop_height = int(H * size_fraction)  # Calculate crop height
        crop_x0 = (W - crop_width) // 2  # Calculate crop top-left x-coordinate
        crop_y0 = (H - crop_height) // 2  # Calculate crop top-left y-coordinate
        crop_x1 = crop_x0 + crop_width  # Calculate crop bottom-right x-coordinate
        crop_y1 = crop_y0 + crop_height  # Calculate crop bottom-right y-coordinate

        ### Create optimizer ###
        optimizer = optim.Adam([homographies, weights], lr=learning_rate)  # Create Adam optimizer

        ### Optimization loop ###
        for iteration in range(max_iterations):  ### Looping Over Indices: ###
            optimizer.zero_grad()  # Zero the gradients

            ### Apply homographies to frames ###
            warped_images = []
            for i in range(num_images):
                grid = F.affine_grid(homographies[i][:2].unsqueeze(0), frames_torch[i].unsqueeze(0).size(), align_corners=True)  # Create affine grid
                warped_image = F.grid_sample(frames_torch[i].unsqueeze(0), grid, align_corners=True).squeeze(0)  # Apply homography
                warped_images.append(warped_image)
            warped_images = torch.stack(warped_images)  # Shape [N, C, H, W]

            ### Compute weighted average on the center crop ###
            center_crops = warped_images[:, :, crop_y0:crop_y1, crop_x0:crop_x1]  # Extract center crops
            weighted_average = AlignClass.compute_weighted_average_torch(center_crops,
                                                                         weights)  # Compute weighted average of center crops

            ### Compute contrast ###
            contrast = AlignClass.compute_contrast_torch(weighted_average,
                                                         method=contrast_method)  # Compute contrast of the weighted average image

            ### Add regularization to weights ###
            regularization = weight_regularizer * torch.sum(weights ** 2)  # Compute regularization term
            loss = -contrast + regularization  # Compute total loss (negative contrast to minimize)

            ### Backpropagation ###
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the parameters

        ### Convert homographies and weights back to numpy arrays ###
        optimized_homographies = homographies.detach().cpu().numpy()  # Convert homographies to numpy arrays, shape [N, 3, 3]
        optimized_weights = weights.detach().cpu().numpy()  # Convert weights to numpy arrays
        weighted_average = torch_to_numpy(weighted_average)  # Convert weighted average to numpy arrays, shape [C, H, W]
        warped_images = torch_to_numpy(warped_images)  # Convert warped images to numpy arrays, shape [N, C, H, W]
        warped_images = [warped_images[i] for i in range(len(warped_images))]
        return optimized_homographies, optimized_weights, weighted_average, warped_images  # Return optimized homographies, weights, weighted average, and warped images


    @staticmethod
    def compute_weighted_average(images, weights):
        """
        Computes the weighted average of a list of images.

        Inputs:
        - images: list of numpy arrays, each of shape [H, W, C]
        - weights: numpy array of shape [H, W, C]

        Outputs:
        - weighted_average: numpy array of shape [H, W, C]
        """
        ### This Is The Code Block: ###
        weighted_sum = np.zeros_like(images[0], dtype=np.float32)  # Initialize weighted sum to zero, shape [H, W, C]
        for img, w in zip(images, weights):  ### Looping Over Indices: ###
            weighted_sum += img * w  # Add weighted image to the sum
        return weighted_sum / np.sum(weights, axis=0)  # Divide by sum of weights to get the average

    ### This Is The Function Description: ###
    @staticmethod
    def compute_contrast(image, method='default'):
        """
        Computes the contrast of an image using the specified method.

        Inputs:
        - image: numpy array of shape [H, W, C]
        - method: str, method to compute contrast ('default' or 'variance')

        Outputs:
        - contrast: float
        """
        ### This Is The Code Block: ###
        if method == 'default':
            blurred = AlignClass.gaussian_filter(image, sigma=1)  # Apply Gaussian filter to the image
            return np.mean(np.abs(image - blurred))  # Compute mean absolute difference

        elif method == 'variance':
            return np.var(image) / np.mean(image)  # Compute variance divided by mean

    ### This Is The Function Description: ###
    @staticmethod
    def objective_function(params, images, H, W, C, num_images, contrast_method):
        """
        Objective function to be minimized.

        Inputs:
        - params: flattened array of optical flows and weights
        - images: list of numpy arrays, each of shape [H, W, C]
        - H: int, height of the images
        - W: int, width of the images
        - C: int, number of channels in the images
        - num_images: int, number of images
        - contrast_method: str, method to compute contrast ('default' or 'variance')

        Outputs:
        - contrast: float
        """
        ### This Is The Code Block: ###
        flows = params[:num_images * H * W * 2].reshape(
            (num_images, H, W, 2))  # Reshape first part of params to flows, shape [num_images, H, W, 2]
        weights = params[num_images * H * W * 2:].reshape(
            (H, W, C))  # Reshape second part of params to weights, shape [H, W, C]

        ### Apply optical flows to images ###
        warped_images = []  # Initialize list for warped images
        for i in range(num_images):  ### Looping Over Indices: ###
            flow = flows[i]  # Get the flow for the i-th image
            img = images[i]  # Get the i-th image
            flow_map = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1) + flow  # Create flow map
            warped_img = cv2.remap(img, flow_map.astype(np.float32), None,
                                   cv2.INTER_LINEAR)  # Apply the flow to the image
            warped_images.append(warped_img)  # Append warped image to the list

        ### Compute weighted average ###
        weighted_average = AlignClass.compute_weighted_average(warped_images, weights)  # Compute weighted average of warped images

        ### Compute contrast ###
        contrast = AlignClass.compute_contrast(weighted_average,
                                    method=contrast_method)  # Compute contrast of the weighted average image

        return contrast  # Return the contrast value

    ### This Is The Function Description: ###
    @staticmethod
    def optimize_optical_flow_and_weights_scipy(images,
                                                max_iterations=100,
                                                weight_regularizer=0.01,
                                                contrast_method='default'):
        """
        Optimizes optical flow maps and pixel weights to minimize the contrast of the weighted average of the images.

        Inputs:
        - images: list of numpy arrays, each of shape [H, W, C]
        - max_iterations: int, maximum number of iterations for the optimization
        - weight_regularizer: float, regularization term for the pixel weights
        - contrast_method: str, method to compute contrast ('default' or 'variance')

        Outputs:
        - optimized_flows: list of numpy arrays, each of shape [H, W, 2], the optimized optical flow maps
        - optimized_weights: numpy array of shape [H, W, C], the optimized pixel weights
        """
        H, W, C = images[0].shape  # Get dimensions of the images
        num_images = len(images)  # Number of images

        ### Initialize optical flow maps and weights ###
        initial_flows = np.zeros((num_images, H, W, 2),
                                 dtype=np.float32).flatten()  # Initialize flows to zero, flattened
        initial_weights = (np.ones((H, W, C),
                                   dtype=np.float32) / num_images).flatten()  # Initialize weights to uniform, flattened
        initial_params = np.concatenate([initial_flows, initial_weights])  # Combine flows and weights into one array

        ### Define bounds for the optimization ###
        bounds = [(-2, 2)] * num_images * H * W * 2 + [
            (0, 1)] * H * W * C  # Bounds for flows [-2, 2] and weights [0, 1]

        ### Perform optimization using SciPy ###
        result = minimize(AlignClass.objective_function,
                          initial_params,
                          args=(images, H, W, C, num_images, contrast_method),
                          method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iterations})

        optimized_params = result.x  # Get optimized parameters
        optimized_flows = optimized_params[:num_images * H * W * 2].reshape(
            (num_images, H, W, 2))  # Reshape to get optimized flows
        optimized_weights = optimized_params[num_images * H * W * 2:].reshape(
            (H, W, C))  # Reshape to get optimized weights

        return optimized_flows, optimized_weights  # Return optimized flows and weights


    @staticmethod
    def apply_homography_to_points_array(points, homography):
        """
        Applies a homography matrix to a set of points.

        Inputs:
        - points: numpy array of shape [N, 2], where N is the number of points
        - homography: numpy array of shape [3, 3], the homography matrix

        Outputs:
        - transformed_points: numpy array of shape [N, 2], the transformed points
        """

        # Convert points to homogeneous coordinates
        num_points = points.shape[0]
        homogeneous_points = np.hstack([points, np.ones((num_points, 1))])  # Shape [N, 3]

        # Apply homography matrix
        transformed_points_homogeneous = np.dot(homography, homogeneous_points.T).T  # Shape [N, 3]

        # Convert back to Cartesian coordinates
        transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2].reshape(-1,
                                                                                                                  1)  # Shape [N, 2]

        return transformed_points

    @staticmethod
    def apply_homographies_to_list_of_points_arrays(points_list, homographies_list):
        """
        Applies a list of homography matrices to a list of point arrays.

        Inputs:
        - points_list: list of numpy arrays, each of shape [N, 2], where N is the number of points
        - homographies: list of numpy arrays, each of shape [3, 3], the homography matrices

        Outputs:
        - transformed_points_list: list of numpy arrays, each of shape [N, 2], the transformed points
        """

        # Check if the length of points_list and homographies are the same
        if len(points_list) != len(homographies_list):
            raise ValueError("The length of points_list and homographies must be the same.")

        # Apply homography to each set of points
        transformed_points_list = [AlignClass.apply_homography_to_points_array(points, homography) for points, homography in
                                   zip(points_list, homographies_list)]

        return transformed_points_list
    @staticmethod
    def plot_points_with_arrows(reference_points, new_points, ax=None, pause_time=0.5, flag_plot=True):
        """
        Plots two sets of points with arrows indicating the transformation from reference points to new points and updates the plot over time.

        Inputs:
        - reference_points: numpy array of reference points of shape [N, 2]
        - new_points: numpy array of new points of shape [N, 2]
        - ax: matplotlib axes object, default is None. If None, create a new figure and axes.
        - pause_time: float, pause time between updates in seconds
        - flag_plot: bool, if True, display the plot; if False, do not display the plot

        Outputs:
        - ax: matplotlib axes object
        """

        # Check if the input arrays are empty
        if reference_points.size == 0 or new_points.size == 0:
            print("Input arrays are empty.")
            return ax

        # Check if the input arrays have the correct shape
        if reference_points.shape != new_points.shape:
            print("Input arrays must have the same shape.")
            return ax

        # Initialize the plot if ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_title('Reference Points and New Points with Arrows')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True)
            ax.axis('equal')

        # Clear current plot
        ax.clear()  # Clear the current plot

        # Plot reference points
        ax.scatter(reference_points[:, 0], reference_points[:, 1], color='blue', label='Reference Points')

        # Plot new points
        ax.scatter(new_points[:, 0], new_points[:, 1], color='red', label='New Points')

        # Draw arrows from reference points to new points
        for i in range(len(reference_points)):
            ax.arrow(reference_points[i, 0], reference_points[i, 1],
                     new_points[i, 0] - reference_points[i, 0], new_points[i, 1] - reference_points[i, 1],
                     color='green', head_width=0.05, length_includes_head=True)

        # Customize plot
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

        if flag_plot:
            plt.draw()
            plt.pause(pause_time)

        return ax

    ### Wrapper Function: test_wrapper ###
    @staticmethod
    def my_test_wrapper(movie_path, test_function_method):
        """
        Wrapper function to test different frame alignment methods using a movie file or image folder.

        Args:
            movie_path (str): Path to the movie file or images folder.
            test_name (str): Name of the test function to run.

        """
        ### Load Frames from Movie or Image Folder: ###
        frames = []  # List to store frames
        if os.path.isdir(movie_path):  # Check if movie_path is a directory
            image_files = sorted(os.listdir(movie_path))  # List image files in the directory
            for image_file in image_files:  # Loop through each image file
                img = cv2.imread(os.path.join(movie_path, image_file))  # Read image
                frames.append(img)  # Append image to frames list
        else:  # If movie_path is a file
            cap = cv2.VideoCapture(movie_path)  # Open video file
            while cap.isOpened():  # Loop through each frame in the video
                ret, frame = cap.read()  # Read frame
                if not ret:  # Break if no frame is read
                    break
                frames.append(frame)  # Append frame to frames list
            cap.release()  # Release video capture

        ### Mapping test_name to Corresponding Test Function: ###
        test_functions = {
            'align_frames_homography': AlignClass.test_align_frames_homography,
            'align_frame_crops_using_given_homographies': AlignClass.test_align_frame_crops_using_given_homographies,
            'align_crops_in_frames_using_given_bounding_boxes': AlignClass.test_align_crops_in_frames_using_given_bounding_boxes,
            'align_crops_optical_flow': AlignClass.test_align_crops_optical_flow,  #TODO: needs to improve
            'align_bounding_boxes_predicted_points': AlignClass.test_align_bounding_boxes_predicted_points,
            'align_frames_to_reference_using_given_optical_flow': AlignClass.test_align_frames_to_reference_using_given_optical_flow,
            'align_frames_translation': AlignClass.test_align_frames_translation,
            'align_crops_using_co_tracker': AlignClass.test_align_crops_using_co_tracker,
        }

        ### Execute the Chosen Test Function: ###
        if test_function_method in test_functions:  # Check if test_name is valid
            test_functions[test_function_method](frames)  # Call the corresponding test function
        else:  # If test_name is not valid
            print(f"Invalid test name: {test_function_method}")  # Print error message


### Main Function: ###
if __name__ == '__main__':
    pass
#     torch.cuda.set_per_process_memory_fraction(0.8, 0)  # Use 50% of the GPU 0's memory
#     movie_path = r'C:\Users\dudyk\Documents\RDND_dudy\SHABAK/scene_10.mp4'
#     test_function_method = 'align_crops_using_co_tracker'
#     AlignClass.my_test_wrapper(movie_path, test_function_method)
#
#
# #
