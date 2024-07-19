
from RapidBase.Utils.Classical_DSP.ECC import *
from RapidBase.Anvil.alignments import align_to_reference_frame_circular_cc
# from vidgear.gears.stabilizer import *
# import kornia.feature as KF

from RapidBase.Utils.Classical_DSP.missile_detection_utils import *

# # initialize params #
# stabilizer_obj = Stabilizer(smoothing_radius=25, border_type="black", border_size=0, crop_n_zoom=False)
# kornia_matcher = KF.LoFTR(pretrained="outdoor")

### ECC params ###
number_of_pixels_to_use_for_ECC = 100000
number_of_iteration_per_level_ECC = 50
number_of_frames_per_batch_ECC = 10
max_pixel_difference_ECC = 0.0002

# opencv feature params #
number_of_features_for_orb = 500

def stabilize_featurebased_opencv_single_frame(current_frame, last_frame, last_frame_features=False):
    """
    Stabilize the current frame based on features matched with the last frame using OpenCV.

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The previous frame used for matching features.
    - last_frame_features (bool or list): List containing the flag for using features from the last frame and the features themselves.

    Returns:
    - current_frame_stabilized_to_last (np.ndarray): The stabilized current frame.
    - last_frame_features (list): Updated list of last frame features.
    """

    ### Get Homography Matrix: ###
    homography_matrix, last_frame_features = calculate_frame_movement_homography_mat(
        current_frame, last_frame,
        number_of_features_for_ORB=number_of_features_for_orb,  ### Use a predefined number of ORB features
        last_frame_features=last_frame_features  ### Use features from the last frame if available
    )

    ### Interpolate Stabilized Image: ###
    if np.any(homography_matrix):  ### Check if the homography matrix is not empty
        current_frame_stabilized_to_last = cv2.warpPerspective(
            current_frame, homography_matrix,  ### Use the homography matrix to warp the current frame
            (current_frame.shape[1], current_frame.shape[0]),  ### Set the output size to the original frame size
            flags=cv2.INTER_NEAREST  ### Use nearest neighbor interpolation
        )
        return current_frame_stabilized_to_last, last_frame_features  ### Return the stabilized frame and features

    return current_frame, last_frame_features  ### Return the original frame and features if homography matrix is empty



def stabilize_featurebased_opencv_binned_2x2(current_frame, last_frame, last_frame_features=False):
    """
    Stabilize the current frame based on features matched with the last frame using OpenCV and 2x2 binning.

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The previous frame used for matching features.
    - last_frame_features (bool or list): List containing the flag for using features from the last frame and the features themselves.

    Returns:
    - current_frame_stabilized_to_last (np.ndarray): The stabilized current frame.
    - last_frame_features (list): Updated list of last frame features.
    """

    ### Get Homography Matrix for binned images: ###
    current_frame_binned = bin_2x2(current_frame).astype(np.uint8)  ### Bin the current frame using 2x2 binning
    last_frame_binned = bin_2x2(last_frame).astype(np.uint8)
    homography_matrix, last_frame_features = calculate_frame_movement_homography_mat(
        current_frame_binned, last_frame_binned,  ### Calculate homography for binned images
        number_of_features_for_ORB=number_of_features_for_orb,  ### Use a predefined number of ORB features
        last_frame_features=last_frame_features  ### Use features from the last frame if available
    )
    homography_matrix = adjust_binned_homography_matrix(homography_matrix, binning_factor=2)  ### Adjust homography for binning

    ### Interpolate Stabilized Image: ###
    if np.any(homography_matrix):  ### Check if the homography matrix is not empty
        current_frame_stabilized_to_last = cv2.warpPerspective(
            current_frame, homography_matrix,  ### Use the homography matrix to warp the current frame
            (current_frame.shape[1], current_frame.shape[0]),  ### Set the output size to the original frame size
            flags=cv2.INTER_NEAREST  ### Use nearest neighbor interpolation
        )
        return current_frame_stabilized_to_last, last_frame_features  ### Return the stabilized frame and features

    return current_frame, last_frame_features  ### Return the original frame and features if homography matrix is empty


def stabilize_featurebased_opencv_binned_2x2_with_H_matrix(current_frame, last_frame, last_frame_features=False):
    """
    Stabilize the current frame based on features matched with the last frame using OpenCV and 2x2 binning,
    and return the homography matrix.

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The previous frame used for matching features.
    - last_frame_features (bool or list): List containing the flag for using features from the last frame and the features themselves.

    Returns:
    - current_frame_stabilized_to_last (np.ndarray): The stabilized current frame.
    - last_frame_features (list): Updated list of last frame features.
    - homography_matrix (np.ndarray): The homography matrix used for stabilization.
    """

    ### Get Homography Matrix for binned images: ###
    current_frame_binned = bin_2x2(current_frame).astype(np.uint8)  ### Bin the current frame using 2x2 binning
    last_frame_binned = bin_2x2(last_frame).astype(np.uint8)  ### Bin the last frame using 2x2 binning
    homography_matrix, last_frame_features = calculate_frame_movement_homography_mat(
        current_frame_binned, last_frame_binned,  ### Calculate homography for binned images
        number_of_features_for_ORB=number_of_features_for_orb,  ### Use a predefined number of ORB features
        last_frame_features=last_frame_features  ### Use features from the last frame if available
    )
    homography_matrix = adjust_binned_homography_matrix(homography_matrix, binning_factor=2)  ### Adjust homography for binning

    ### Interpolate Stabilized Image: ###
    if np.any(homography_matrix):  ### Check if the homography matrix is not empty
        current_frame_stabilized_to_last = cv2.warpPerspective(
            current_frame, homography_matrix,  ### Use the homography matrix to warp the current frame
            (current_frame.shape[1], current_frame.shape[0]),  ### Set the output size to the original frame size
            flags=cv2.INTER_NEAREST  ### Use nearest neighbor interpolation
        )
        return current_frame_stabilized_to_last, last_frame_features, homography_matrix  ### Return the stabilized frame, features, and homography matrix

    return current_frame, last_frame_features, homography_matrix  ### Return the original frame, features, and homography matrix if homography matrix is empty


def compute_optical_flow_vectorized(frame, H):
    """
    Compute the optical flow (delta_x, delta_y) for each pixel in the frame using the given homography matrix.

    Parameters:
    - frame (np.ndarray): HxW numpy array representing the image frame.
    - H (np.ndarray): 3x3 homography matrix.

    Returns:
    - optical_flow (np.ndarray): HxWx2 numpy array where each element contains (delta_x, delta_y) for each pixel.
    """
    H_inv = np.linalg.inv(H)  ### Invert the homography matrix
    height, width = frame.shape[:2]  ### Get the height and width of the frame

    ### Create a grid of (x, y) coordinates ###
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    ### Flatten the coordinate grids ###
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    ### Create a matrix of homogeneous coordinates ###
    ones = np.ones_like(x_flat)
    original_points = np.vstack((x_flat, y_flat, ones)).T

    ### Apply the inverse homography matrix to all points ###
    transformed_points = H_inv @ original_points.T

    ### Normalize homogeneous coordinates ###
    transformed_points /= transformed_points[2, :]

    ### Compute delta_x and delta_y ###
    delta_x = transformed_points[0, :] - x_flat
    delta_y = transformed_points[1, :] - y_flat

    ### Reshape to the original frame shape ###
    optical_flow = np.stack((delta_x, delta_y), axis=-1).reshape(height, width, 2)

    return optical_flow  ### Return the computed optical flow


def stabilize_featurebased_kornia(last_frame, current_frame):
    """
    Stabilize the current frame using Kornia for feature extraction and matching.

    Parameters:
    - last_frame (np.ndarray): The previous frame used for stabilization.
    - current_frame (np.ndarray): The current frame to be stabilized.

    Returns:
    - current_tensor_stabilized (np.ndarray): The stabilized current frame.
    """

    ### Convert images to grayscale tensors ###
    reference_tensor = kornia.image_to_tensor(last_frame, keepdim=False)
    current_tensor = kornia.image_to_tensor(current_frame, keepdim=False)
    input_dict = {
        "image0": kornia.color.rgb_to_grayscale(reference_tensor).float().cuda(),
        "image1": kornia.color.rgb_to_grayscale(current_tensor).float().cuda()
    }

    ### Run Kornia LOFTR features extractor and matcher ###
    with torch.inference_mode():
        correspondences = kornia_matcher.forward(input_dict)
    mkpts0 = correspondences["keypoints0"]
    mkpts1 = correspondences["keypoints1"]

    ### Estimate homography and warp image accordingly ###
    if len(mkpts0.shape) == 2:
        mkpts0 = mkpts0.unsqueeze(0)
    if len(mkpts1.shape) == 2:
        mkpts1 = mkpts1.unsqueeze(0)
    try:
        h = kornia.geometry.homography.find_homography_dlt(mkpts0, mkpts1)
        current_tensor_warped = kornia.geometry.warp_perspective(current_tensor, h, dsize=
        (reference_tensor.shape[-2], reference_tensor.shape[-1]))
        current_tensor_stabilized = kornia.tensor_to_image(current_tensor_warped)
    except:
        current_tensor_stabilized = current_frame

    return current_tensor_stabilized  ### Return the stabilized frame



def bin_2x2(image: np.ndarray) -> np.ndarray:
    """
    Perform 2x2 binning on the input image.

    Parameters:
    - image (np.ndarray): Input image array of shape (height, width, channels).

    Returns:
    - binned_image (np.ndarray): Binned image array of shape (height//2, width//2, channels).
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]  ### If the image is grayscale, add a channel dimension for consistency

    height, width, channels = image.shape  ### Get the height, width, and channels of the image
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError \
            ("Image dimensions must be even for 2x2 binning.")  ### Ensure the dimensions are even for 2x2 binning

    binned_image = image.reshape((height // 2, 2, width // 2, 2, channels)).mean(axis=(1, 3))  ### Reshape and average

    if binned_image.shape[-1] == 1:
        binned_image = binned_image[:, :, 0]  ### If the original image was grayscale, remove the added channel dimension

    return binned_image  ### Return the binned image

def adjust_binned_homography_matrix(H_binned, binning_factor):
    """
    Adjust the homography matrix for binning.

    Parameters:
    - H_binned (np.ndarray): Homography matrix for binned images.
    - binning_factor (int): Binning factor used for binning.

    Returns:
    - H_original (np.ndarray): Adjusted homography matrix for the original image.
    """
    S = np.array([
        [binning_factor, 0, 0],
        [0, binning_factor, 0],
        [0, 0, 1]
    ])  ### Create the scaling matrix

    H_original = np.dot(np.dot(S, H_binned), np.linalg.inv(S))  ### Calculate the homography matrix for the original image

    return H_original  ### Return the adjusted homography matrix


def initialize_features(previous_image: np.ndarray):
    """
    Initialize and detect features in the previous image for tracking.

    Parameters:
    - previous_image (np.ndarray): Grayscale image of the previous frame.

    Returns:
    - prev_pts (np.ndarray): The detected feature points in the previous image.
    """
    feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)  ### Set parameters for feature detection
    prev_pts = cv2.goodFeaturesToTrack(previous_image, mask=None, **feature_params)  ### Detect good features to track
    return prev_pts  ### Return the detected feature points



def track_and_warp(previous_image: np.ndarray, current_image: np.ndarray, prev_pts):
    """
    Track features from the previous image to the current image and calculate the homography matrix.
    Warp the current image to align with the previous image using the homography matrix.

    Parameters:
    - previous_image (np.ndarray): Grayscale image of the previous frame.
    - current_image (np.ndarray): Grayscale image of the current frame.
    - prev_pts (np.ndarray): Feature points detected in the previous image.

    Returns:
    - homography_matrix (np.ndarray): The 3x3 homography matrix.
    - next_pts (np.ndarray): The tracked feature points in the current image.
    """
    if prev_pts is None:
        prev_pts = initialize_features(previous_image)  ### Initialize features if not provided

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  ### Set parameters for Lucas-Kanade optical flow
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(previous_image, current_image, prev_pts, None, **lk_params)  ### Calculate optical flow

    good_prev_pts = prev_pts[status == 1]  ### Select good points in the previous image
    good_next_pts = next_pts[status == 1]  ### Select good points in the current image

    if len(good_next_pts) < 300:  ### If not enough good points, reinitialize
        good_prev_pts = initialize_features(previous_image)
        good_next_pts = cv2.calcOpticalFlowPyrLK(previous_image, current_image, good_prev_pts, None, **lk_params)[0]

    try:
        homography_matrix, _ = cv2.findHomography(good_prev_pts, good_next_pts, cv2.RANSAC, 5.0)  ### Compute the homography matrix
        return homography_matrix, good_next_pts  ### Return the homography matrix and good points

    except cv2.error:
        return None, good_next_pts  ### Return None and good points if homography calculation fails


def stabilize_featurebased_vidgear(current_frame, last_frame):
    """
    Stabilize the current frame using VidGear.

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The previous frame used for stabilization.

    Returns:
    - current_frame_stabilized_to_last (np.ndarray): The stabilized current frame.
    """
    current_frame_stabilized_to_last = stabilizer_obj.stabilize \
        (current_frame)  ### Stabilize the current frame using VidGear
    if not (current_frame_stabilized_to_last is None):  ### Check if stabilization was successful
        return current_frame_stabilized_to_last  ### Return the stabilized frame

    return current_frame  ### Return the original frame if stabilization fails


def stabilize_ecc(current_frame, last_frame):
    """
    Stabilize the current frame using Enhanced Correlation Coefficient (ECC).

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The previous frame used for stabilization.

    Returns:
    - input_tensor_stabilized_numpy (np.ndarray): The stabilized current frame.
    """
    reference_tensor = numpy_to_torch(last_frame).unsqueeze(0)  ### Convert last frame to tensor and add batch dimension
    input_tensor = numpy_to_torch(current_frame).unsqueeze \
        (0)  ### Convert current frame to tensor and add batch dimension

    input_tensor_stabilized = stabilize_images_ecc(input_tensor,
                                                   number_of_iteration_per_level_ECC=number_of_iteration_per_level_ECC,
                                                   number_of_pixels_to_use_for_ECC=number_of_pixels_to_use_for_ECC,
                                                   number_of_frames_per_batch_ECC=number_of_frames_per_batch_ECC,
                                                   max_pixel_difference_ECC=max_pixel_difference_ECC,
                                                   reference_tensor=RGB2BW(reference_tensor))  ### Stabilize using ECC

    input_tensor_stabilized_numpy = torch_to_numpy \
        (input_tensor_stabilized[0])  ### Convert stabilized tensor to numpy array
    return input_tensor_stabilized_numpy  ### Return the stabilized frame



def stabilize_ccc(current_frame, last_frame):
    """
    Stabilize the current frame using Circular Cross-Correlation (CCC).

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The previous frame used for stabilization.

    Returns:
    - current_frame_stabilized_to_last (np.ndarray): The stabilized current frame.
    """
    reference_tensor = numpy_to_torch(last_frame).unsqueeze(0)  ### Convert last frame to tensor and add batch dimension
    input_tensor = numpy_to_torch(current_frame).unsqueeze \
        (0)  ### Convert current frame to tensor and add batch dimension

    current_frame_stabilized_to_last_torch, _, _, _ = align_to_reference_frame_circular_cc(input_tensor, reference_tensor)  ### Align using CCC
    if current_frame_stabilized_to_last_torch.any():  ### Check if stabilization was successful
        current_frame_stabilized_to_last = cv2.cvtColor \
            (current_frame_stabilized_to_last_torch.squeeze().cpu().detach().numpy(), cv2.COLOR_GRAY2BGR)  ### Convert tensor to BGR image
        return current_frame_stabilized_to_last  ### Return the stabilized frame

    return current_frame  ### Return the original frame if stabilization fails


def stabilize_sparse_opticalflow_opencv(current_frame, last_frame, prev_pts=None):
    """
    Stabilize the current frame using sparse optical flow and OpenCV.

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The previous frame used for stabilization.
    - prev_pts (np.ndarray): Previous feature points.

    Returns:
    - current_frame_stabilized_to_last (np.ndarray): The stabilized current frame.
    - prev_pts (np.ndarray): Updated previous feature points.
    """
    homography_matrix, prev_pts = track_and_warp(last_frame[:, :, 0], current_frame[:, :, 0], prev_pts)  ### Track and warp using optical flow

    if len(prev_pts.shape) != 3:  ### Ensure prev_pts is 3-dimensional
        prev_pts = prev_pts.reshape(-1, 1, 2)

    if np.any(homography_matrix):  ### Check if homography matrix is not empty
        height, width, _ = last_frame.shape  ### Get the height and width of the last frame
        current_frame_stabilized_to_last = cv2.warpPerspective(current_frame, homography_matrix,
        (width, height))  ### Warp the current frame to align with the last frame
        return current_frame_stabilized_to_last, prev_pts  ### Return the stabilized frame and updated points

    return current_frame, prev_pts  ### Return the original frame and points if homography matrix is empty


def calculate_frame_movement_homography_mat(frame_to_stabilize, reference_frame, number_of_features_for_ORB=500,
                                            last_frame_features=[None, None, None]):
    """
    Calculate homography matrix between two frames using ORB features.

    Parameters:
    - frame_to_stabilize (np.ndarray): The frame to be stabilized.
    - reference_frame (np.ndarray): The reference frame.
    - number_of_features_for_ORB (int): Number of ORB features to detect.
    - last_frame_features (list): Features from the last frame [return_features, keypoints, descriptors].

    Returns:
    - H (np.ndarray): Homography matrix.
    - last_frame_features (list): Updated features from the last frame.
    """

    ### Initialize ORB Detector and BFMatcher: ###
    detector = cv2.ORB_create(number_of_features_for_ORB)  ### Create ORB detector with the specified number of features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  ### Create BFMatcher with Hamming norm and cross-check

    ### Detect Features and Descriptors: ###
    gtic()
    kp1, des1 = detector.detectAndCompute(frame_to_stabilize,
                                          None)  ### Detect keypoints and descriptors in frame_to_stabilize
    if np.any(last_frame_features[1]):
        kp2, des2 = last_frame_features[1:]  ### Use features from the last frame if available
    else:
        print('detecting reference frame features')
        kp2, des2 = detector.detectAndCompute(reference_frame,
                                              None)  ### Detect keypoints and descriptors in reference_frame
    gtoc('detect features')

    ### Match features: ###
    try:
        gtic()
        matches = matcher.match(des1, des2)  ### Match descriptors between frame_to_stabilize and reference_frame
        gtoc('matching')
        gtic()
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  ### Source points from matches
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1,
                                                                            2)  ### Destination points from matches
        gtoc('float')
        gtic()
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  ### Find homography using RANSAC
        gtoc('find homography')

        if last_frame_features[0]:
            last_frame_features[1] = kp1  ### Update keypoints in last_frame_features
            last_frame_features[2] = des1  ### Update descriptors in last_frame_features
            return H, last_frame_features  ### Return homography matrix and updated last_frame_features

        return H, last_frame_features  ### Return homography matrix and updated last_frame_features

    except Exception as e:
        print(f"Error during matching: {e}")  ### Print error message if exception occurs
        if last_frame_features[0]:
            last_frame_features[1] = kp1  ### Update keypoints in last_frame_features
            last_frame_features[2] = des1  ### Update descriptors in last_frame_features
            return None, last_frame_features  ### Return None and updated last_frame_features
        return None, None  ### Return None if no homography matrix can be found


def H_matrix_convert_from_pairwise_to_reference_frame(H_matrices_list):
    """
    Convert a list of pairwise homography matrices to reference frame homographies.

    Parameters:
    - H_matrices_list (list or np.ndarray): List of pairwise homography matrices.

    Returns:
    - H_to_first_frame (np.ndarray): Homography matrices to the first frame.
    """

    ### Check if H_matrices_list is a list of matrices and convert to numpy array if needed: ###
    if isinstance(H_matrices_list, list):
        H_matrices = np.array(H_matrices_list)  ### Convert list to numpy array
    else:
        H_matrices = H_matrices_list  ### Use the provided numpy array

    ### Initialize the result array: ###
    H_to_first_frame = np.zeros_like(H_matrices)  ### Initialize H_to_first_frame with zeros

    ### First frame's homography matrix is identity: ###
    H_to_first_frame[0] = np.eye(3)  ### Set the first homography matrix to the identity matrix

    ### Iterate through the matrices to calculate cumulative product: ###
    for t in range(1, H_matrices.shape[0]):
        H_to_first_frame[t] = H_to_first_frame[t - 1] @ H_matrices[
            t - 1]  ### Calculate cumulative product of homography matrices

    return H_to_first_frame  ### Return the homography matrices to the first frame


def calculate_frame_movement_homography_mat_dudy(frame_to_stabilize, reference_frame, number_of_features_for_ORB=500,
                                                 last_frame_features=[None, None], flag_matching_crosscheck=True):
    """
    Calculate homography matrix between two frames using ORB features (Dudy's version).

    Parameters:
    - frame_to_stabilize (np.ndarray): The frame to be stabilized.
    - reference_frame (np.ndarray): The reference frame.
    - number_of_features_for_ORB (int): Number of ORB features to detect.
    - last_frame_features (list): Features from the last frame [return_features, keypoints, descriptors].

    Returns:
    - H (np.ndarray): Homography matrix.
    - last_frame_features (list): Updated features from the last frame.
    """

    ### Initialize ORB Detector and BFMatcher: ###
    detector = cv2.ORB_create(number_of_features_for_ORB)  ### Create ORB detector with the specified number of features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING,
                            crossCheck=flag_matching_crosscheck)  ### Create BFMatcher with Hamming norm and cross-check

    ### Detect Features and Descriptors: ###
    kp1, des1 = detector.detectAndCompute(frame_to_stabilize,
                                          None)  ### Detect keypoints and descriptors in frame_to_stabilize
    if np.any(last_frame_features[1]):
        kp2, des2 = last_frame_features[1:]  ### Use features from the last frame if available
    else:
        print('calculate reference features')
        kp2, des2 = detector.detectAndCompute(reference_frame,
                                              None)  ### Detect keypoints and descriptors in reference_frame

    ### Match features: ###
    try:
        gtic()
        matches = matcher.match(des1, des2)  ### Match descriptors between frame_to_stabilize and reference_frame
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  ### Source points from matches
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1,
                                                                            2)  ### Destination points from matches
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  ### Find homography using RANSAC
        gtoc('matching')

        if last_frame_features[0]:
            last_frame_features[1] = kp1  ### Update keypoints in last_frame_features
            last_frame_features[2] = des1  ### Update descriptors in last_frame_features
            return H, last_frame_features  ### Return homography matrix and updated last_frame_features

        return H, last_frame_features  ### Return homography matrix and updated last_frame_features

    except Exception as e:
        print(f"Error during matching: {e}")  ### Print error message if exception occurs
        if last_frame_features[0]:
            last_frame_features[1] = kp1  ### Update keypoints in last_frame_features
            last_frame_features[2] = des1  ### Update descriptors in last_frame_features
            return None, last_frame_features  ### Return None and updated last_frame_features
        return None, None  ### Return None if no homography matrix can be found


def tukey_biweight(residuals, c=4.685):
    """
    Calculate Tukey's Biweight weights.

    Parameters:
    - residuals (np.ndarray): Residuals to calculate weights for.
    - c (float): Scaling factor for Tukey's biweight function.

    Returns:
    - weights (np.ndarray): Tukey's biweight weights.
    """

    residuals = residuals / c  ### Scale residuals by constant c
    mask = np.abs(residuals) <= 1  ### Create mask for residuals within threshold
    weights = np.zeros_like(residuals)  ### Initialize weights with zeros
    weights[mask] = (1 - residuals[mask] ** 2) ** 2  ### Calculate weights using Tukey's biweight function
    return weights  ### Return the calculated weights


def weighted_least_squares_homography_matrix_points(src_pts, dst_pts, weights):
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


def plot_keypoints_with_weights(frame, keypoints, weights, title, weights_size_base=100):
    """
    Plot keypoints with weights on the given frame.

    Parameters:
    - frame (np.ndarray): The image frame.
    - keypoints (np.ndarray): Array of keypoints.
    - weights (np.ndarray): Array of weights for each keypoint.
    - title (str): Title of the plot.
    - weights_size_base (int): Base size for scaling weights.

    Returns:
    - None
    """

    plt.imshow(frame, cmap='gray')  ### Display the frame in grayscale
    x, y = keypoints[:, 0], keypoints[:, 1]  ### Extract x and y coordinates from keypoints
    plt.scatter(x, y, s=weights * weights_size_base, c='red',
                alpha=0.6)  ### Scatter plot of keypoints with sizes proportional to weights
    plt.title(title)  ### Set the title of the plot
    plt.show()  ### Display the plot


def plot_keypoints_and_matches(frame1, frame2, keypoints1, keypoints2, matches, inliers, title):
    """
    Plot keypoints and matches between two frames.

    Parameters:
    - frame1 (np.ndarray): The first frame.
    - frame2 (np.ndarray): The second frame.
    - keypoints1 (np.ndarray): Keypoints in the first frame.
    - keypoints2 (np.ndarray): Keypoints in the second frame.
    - matches (list): List of matches between keypoints.
    - inliers (np.ndarray): Boolean array indicating inliers.
    - title (str): Title for the plot.
    """

    ### Combine the two frames side by side ###
    combined_image = np.hstack((frame1, frame2))  ### Create a combined image by horizontally stacking the frames
    plt.imshow(combined_image, cmap='gray')  ### Display the combined image in grayscale

    offset = frame1.shape[1]  ### Calculate the offset for the second frame

    ### Extract inliers and outliers ###
    keypoints1_inliers = keypoints1[inliers]  ### Keypoints in frame1 that are inliers
    keypoints2_inliers = keypoints2[inliers]  ### Keypoints in frame2 that are inliers
    keypoints1_outliers = keypoints1[~inliers]  ### Keypoints in frame1 that are outliers
    keypoints2_outliers = keypoints2[~inliers]  ### Keypoints in frame2 that are outliers

    ### Plot inlier matches ###
    plt.plot(np.vstack([keypoints1_inliers[:, 0], keypoints2_inliers[:, 0] + offset]),
             np.vstack([keypoints1_inliers[:, 1], keypoints2_inliers[:, 1]]),
             color='green', alpha=0.6)  ### Plot green lines for inlier matches

    ### Plot outlier matches ###
    plt.plot(np.vstack([keypoints1_outliers[:, 0], keypoints2_outliers[:, 0] + offset]),
             np.vstack([keypoints1_outliers[:, 1], keypoints2_outliers[:, 1]]),
             color='blue', alpha=0.6)  ### Plot blue lines for outlier matches

    ### Plot keypoints ###
    plt.scatter(keypoints1[:, 0], keypoints1[:, 1], s=20,
                c=['green' if inlier else 'blue' for inlier in inliers], alpha=0.6)  ### Plot keypoints in frame1
    plt.scatter(keypoints2[:, 0] + offset, keypoints2[:, 1], s=20,
                c=['green' if inlier else 'blue' for inlier in inliers], alpha=0.6)  ### Plot keypoints in frame2

    plt.title(title)  ### Set the title of the plot
    plt.show()  ### Display the plot


def find_homography_with_canny(frame_to_stabilize, reference_frame, low_threshold=50, high_threshold=150,
                               max_iterations=2000, flag_matching_strategy=0):
    """
    Find homography matrix between two frames using Canny edge detection and RANSAC.

    Parameters:
    - frame_to_stabilize (np.ndarray): The frame to be stabilized.
    - reference_frame (np.ndarray): The reference frame.
    - low_threshold (int): Low threshold for Canny edge detection.
    - high_threshold (int): High threshold for Canny edge detection.
    - max_iterations (int): Maximum iterations for RANSAC.
    - flag_matching_strategy (int): Flag to choose matching strategy (0 or 1).

    Returns:
    - H (np.ndarray): Homography matrix.
    - mask (np.ndarray): Mask of inliers.
    """

    ### Detect edges using Canny edge detection ###
    edges1 = cv2.Canny(cv2.blur(frame_to_stabilize[:, :, 0], (5, 5)), low_threshold,
                       high_threshold)  ### Detect edges in frame_to_stabilize
    edges2 = cv2.Canny(cv2.blur(reference_frame[:, :, 0], (5, 5)), low_threshold,
                       high_threshold)  ### Detect edges in reference_frame

    ### Extract edge points ###
    points1 = np.column_stack(np.nonzero(edges1)).astype(float)  ### Extract edge points from frame_to_stabilize
    points2 = np.column_stack(np.nonzero(edges2)).astype(float)  ### Extract edge points from reference_frame

    ### Matching strategy 0: Ensure the same number of points by random sampling ###
    if flag_matching_strategy == 0:
        if len(points1) > len(points2):
            points1 = points1[
                np.random.choice(points1.shape[0], len(points2), replace=False)]  ### Randomly sample points1
        else:
            points2 = points2[
                np.random.choice(points2.shape[0], len(points1), replace=False)]  ### Randomly sample points2

        src_pts = np.array([pt for pt in points1], dtype=np.float32).reshape(-1, 1, 2)  ### Source points
        dst_pts = np.array([pt for pt in points2], dtype=np.float32).reshape(-1, 1, 2)  ### Destination points

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0,
                                     maxIters=max_iterations)  ### Find homography matrix using RANSAC

    ### Matching strategy 1: Use ORB keypoints and descriptors ###
    else:
        keypoints1 = [cv2.KeyPoint(pt[1], pt[0], 1) for pt in points1]  ### Create KeyPoint objects for points1
        keypoints2 = [cv2.KeyPoint(pt[1], pt[0], 1) for pt in points2]  ### Create KeyPoint objects for points2

        orb = cv2.ORB_create()  ### Initialize ORB detector

        keypoints1, descriptors1 = orb.compute(frame_to_stabilize, keypoints1)  ### Compute descriptors for keypoints1
        keypoints2, descriptors2 = orb.compute(reference_frame, keypoints2)  ### Compute descriptors for keypoints2

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  ### Initialize BFMatcher
        matches = matcher.match(descriptors1, descriptors2)  ### Match descriptors

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1,
                                                                                   2)  ### Source points from matches
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1,
                                                                                   2)  ### Destination points from matches

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0,
                                     maxIters=max_iterations)  ### Find homography matrix using RANSAC

    return H, mask  ### Return the homography matrix and inlier mask


def calculate_frame_movement_homography_mat_dudy_iterative_reweighing(frame_to_stabilize, reference_frame,
                                                                      number_of_features_for_ORB=500,
                                                                      last_frame_features=[None, None],
                                                                      max_iterations=5, inlier_threshold=2.5, c=4.685):
    """
    Calculate frame movement using homography and iterative reweighting.

    Parameters:
    - frame_to_stabilize (np.ndarray): The frame to be stabilized.
    - reference_frame (np.ndarray): The reference frame.
    - number_of_features_for_ORB (int): Number of ORB features to detect.
    - last_frame_features (list): Features from the last frame.
    - max_iterations (int): Maximum number of iterations.
    - inlier_threshold (float): Threshold to identify inliers.
    - c (float): Constant for Tukey's biweight function.

    Returns:
    - H (np.ndarray): Homography matrix.
    - last_frame_features (list): Updated features from the last frame.
    """

    detector = cv2.ORB_create(number_of_features_for_ORB)  ### Initialize ORB detector
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  ### Initialize BFMatcher

    kp1, des1 = detector.detectAndCompute(frame_to_stabilize,
                                          None)  ### Detect keypoints and descriptors in frame_to_stabilize
    if np.any(last_frame_features[1]):
        kp2, des2 = last_frame_features[1:]  ### Use features from the last frame if available
    else:
        kp2, des2 = detector.detectAndCompute(reference_frame,
                                              None)  ### Detect keypoints and descriptors in reference_frame

    try:
        matches = matcher.match(des1, des2)  ### Match descriptors
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)  ### Source points from matches
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)  ### Destination points from matches

        for iteration in range(max_iterations):
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0,
                                         maxIters=2000)  ### Find homography using RANSAC
            if H is None:
                break

            RANSAC_inliers = mask.ravel().astype(bool)  ### RANSAC inliers
            num_inliers = np.sum(RANSAC_inliers)
            if num_inliers < 4:
                break

            transformed_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1,
                                                                                             2)  ### Transform source points
            residuals = np.linalg.norm(src_pts - transformed_pts, axis=1)  ### Calculate residuals
            outliers = residuals > inlier_threshold  ### Identify outliers
            RANSAC_inliers[RANSAC_inliers] = ~outliers[RANSAC_inliers]  ### Update RANSAC inliers

            weights = tukey_biweight(residuals, c)  ### Reweight inliers using Tukey's Biweight function
            weights = weights / np.sum(weights)  ### Normalize weights

            src_pts_inliers = src_pts[RANSAC_inliers]  ### Get inlier source points
            dst_pts_inliers = dst_pts[RANSAC_inliers]  ### Get inlier destination points
            weights_inliers = weights[RANSAC_inliers]  ### Get inlier weights

            H = weighted_least_squares_homography_matrix_points(src_pts_inliers, dst_pts_inliers,
                                                                weights_inliers)  ### Update homography using weighted least squares

            src_pts = src_pts[RANSAC_inliers]  ### Update source points for next iteration
            dst_pts = dst_pts[RANSAC_inliers]  ### Update destination points for next iteration
            weights = weights[RANSAC_inliers]  ### Update weights for next iteration

            if num_inliers == np.sum(RANSAC_inliers):
                break

        if last_frame_features[0]:
            last_frame_features[1] = kp1
            last_frame_features[2] = des1
            return H, last_frame_features

        return H, last_frame_features

    except Exception as e:
        print(f"Error during matching: {e}")
        if last_frame_features[0]:
            last_frame_features[1] = kp1
            last_frame_features[2] = des1
            return None, last_frame_features
        return None, None


def fast_binning_2D_PixelBinning2(input_tensor, binning_size, overlap_size):
    """
    Perform fast 2D pixel binning on the input tensor or numpy array.

    Parameters:
    - input_tensor (torch.Tensor or np.ndarray): Input tensor or array of shape [T,H,W], [T,C,H,W], [B,T,C,H,W], or [H,W].
    - binning_size (int): Size of the binning window.
    - overlap_size (int): Overlap size between bins.

    Returns:
    - binned_matrix_final (torch.Tensor or np.ndarray): Binned tensor or array.
    """
    # Convert numpy array to torch tensor if needed
    is_numpy = isinstance(input_tensor, np.ndarray)
    if is_numpy:
        input_tensor = numpy_to_torch(input_tensor)

    # Determine the full shape and the number of dimensions
    shape_len = len(input_tensor.shape)
    step_size = binning_size - overlap_size

    if shape_len == 2:  # [H,W]
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add dummy batch and channel dimensions
    elif shape_len == 3:  # [T,H,W] or [H,W,C]
        if input_tensor.shape[0] != 3:
            input_tensor = input_tensor.unsqueeze(1)  # Add dummy channel dimension
    elif shape_len == 4:  # [T,C,H,W]
        pass  # Do nothing
    elif shape_len == 5:  # [B,T,C,H,W]
        pass  # Do nothing
    else:
        raise ValueError("Unsupported input shape")

    # B, T, C, H, W = input_tensor.shape[-5:]
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    H_final = 1 + (H - binning_size) // step_size
    W_final = 1 + (W - binning_size) // step_size

    column_cumsum = torch.cat(
        (torch.zeros((*input_tensor.shape[:-1], 1)).to(input_tensor.device), torch.cumsum(input_tensor, -1)), -1)
    column_binning = column_cumsum[..., binning_size::step_size] - column_cumsum[..., :-binning_size:step_size]

    row_cumsum = torch.cat(
        (torch.zeros((*input_tensor.shape[:-2], 1, W_final)).to(input_tensor.device), torch.cumsum(column_binning, -2)),
        -2)
    binned_matrix_final = row_cumsum[..., binning_size::step_size, :] - row_cumsum[..., :-binning_size:step_size, :]

    # Remove dummy dimensions if needed
    if shape_len == 2:
        binned_matrix_final = binned_matrix_final.squeeze(0).squeeze(0)
    elif shape_len == 3:
        binned_matrix_final = binned_matrix_final.squeeze(1) if input_tensor.shape[0] != 3 else binned_matrix_final

    return torch_to_numpy(binned_matrix_final) if is_numpy else binned_matrix_final


def stabilize_batch_of_frames_opencv_binned(frames, reference_frame=None):
    stabilized_frames = np.empty_like(frames)
    last_frame_features = [False, None, None]

    if reference_frame is None:
        reference_frame = frames[len(frames) // 2]

    for frame_index, frame in enumerate(frames):
        stabilized_frame, _, _ = stabilize_featurebased_opencv_binned_2x2_with_H_matrix(frame, reference_frame,
                                                                                        last_frame_features)
        stabilized_frames[frame_index] = stabilized_frame

    return stabilized_frames


def stabilize_frames_FeatureBased(frames_to_stabilize,
                                  reference_frame=None,
                                  flag_output_array_form='list',
                                  number_of_features_for_ORB=500,
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
                                  flag_matching_crosscheck=False):
    """
    Stabilize a series of frames using feature-based homography iterative reweighing or Dudy's method.

    Parameters:
    - frames_to_stabilize (list or np.ndarray or torch.Tensor): Input frames to be stabilized.
    - reference_frame (np.ndarray, optional): Reference frame for stabilization. Defaults to the first frame.
    - flag_output_array_form (str): Output format for stabilized frames. Options are 'list', 'numpy', or 'torch'.
    - number_of_features_for_ORB (int): Number of ORB features to detect.
    - max_iterations (int): Maximum number of iterations for iterative reweighing.
    - inlier_threshold (float): Threshold to identify inliers.
    - c (float): Constant for Tukey's biweight function.
    - interpolation (str): Interpolation method for warping. Options are 'nearest' or 'bilinear'.
    - flag_registration_algorithm (str): Algorithm to use for registration. Options are 'dudy' or 'iterative_reweighing'.
    - flag_perform_interpolation (bool): Whether to perform perspective warp or not.
    - last_frame_features (list): Features from the last frame. Default is [None, None].
    - flag_downsample_frames (bool): Whether to downsample frames before stabilization.
    - binning_factor (int): Factor for downsampling.

    Returns:
    - stabilized_frames_to_reference_list (list or np.ndarray or torch.Tensor): Stabilized frames in the specified format.
    - homography_matrices_list (list or np.ndarray): List of homography matrices used for stabilization.
    - last_frame_features (list): Updated list of last frame features.
    """

    def to_numpy_array(frames):
        """
        Convert frames to numpy array if they are not already in that form and determine dimensions.

        Parameters:
        - frames (list or np.ndarray or torch.Tensor): Input frames.

        Returns:
        - frames (np.ndarray): Frames as numpy array.
        - single_frame_input (bool): Flag indicating if the input is a single frame.
        - T (int): Number of frames (time dimension).
        - H (int): Height of frames.
        - W (int): Width of frames.
        - C (int): Number of channels in frames.
        """
        if isinstance(frames, list):
            frames = np.array(
                [frame.squeeze() if frame.ndim > 2 and frame.shape[-1] == 1 else frame for frame in frames])
        elif isinstance(frames, torch.Tensor):
            frames = frames.squeeze().cpu().numpy()

        single_frame_input = False
        if frames.ndim == 2:  # Single grayscale frame
            frames = frames[np.newaxis, ..., np.newaxis]
            single_frame_input = True
        elif frames.ndim == 3:
            if frames.shape[-1] in {1, 3}:  # Single color or grayscale frame with channel dimension
                frames = frames[np.newaxis, ...]
                single_frame_input = True
            elif frames.shape[0] not in {1, 3}:  # Multiple frames
                single_frame_input = False
                frames = numpy_unsqueeze(frames, -1)  # [T,H,W] -> [T,H,W,C]
            else:  # Single frame but ambiguous shape (e.g., (3, H, W))
                frames = frames[np.newaxis, ...]
                single_frame_input = True

        T, H, W, C = frames.shape
        return frames, single_frame_input, T, H, W, C

    def to_output_format(frames, output_format):
        """
        Convert frames to the specified output format.

        Parameters:
        - frames (np.ndarray): Input frames.
        - output_format (str): Desired output format ('list', 'numpy', 'torch').

        Returns:
        - frames (list or np.ndarray or torch.Tensor): Frames in the specified format.
        """
        if output_format == 'list':
            return [frame for frame in frames]
        elif output_format == 'numpy':
            return np.array(frames)
        elif output_format == 'torch':
            return torch.tensor(frames).unsqueeze(1).float()
        else:
            raise ValueError(f"Invalid output format: {output_format}")

    def adjust_binned_homography_matrix(H_binned, binning_factor):
        """
        Adjust homography matrix from binned to original size.

        Parameters:
        - H_binned (np.ndarray): Binned homography matrix.
        - binning_factor (int): Binning factor.

        Returns:
        - np.ndarray: Adjusted homography matrix.
        """
        S = np.array([
            [binning_factor, 0, 0],
            [0, binning_factor, 0],
            [0, 0, 1]
        ])
        return np.dot(np.dot(S, H_binned), np.linalg.inv(S))

    def fast_binning_2D_PixelBinning(input_tensor, binning_size, overlap_size):
        """
        Perform fast binning on the input tensor.

        Parameters:
        - input_tensor (torch.Tensor or np.ndarray): Input tensor to be binned.
        - binning_size (int): Size of the binning window.
        - overlap_size (int): Size of the overlap between bins.

        Returns:
        - binned_matrix_final (torch.Tensor or np.ndarray): Binned tensor.
        """
        flag_squeeze = False
        if isinstance(input_tensor, np.ndarray):
            dtype = input_tensor.dtype
            if len(input_tensor.shape) == 3:
                if (input_tensor.shape[-1] == 3 or input_tensor.shape[-1] == 1):  # [H,W,1] OR [H,W,3]
                    input_tensor = numpy_to_torch(input_tensor).unsqueeze(0)  # -> [1,C,H,W]
                    flag_squeeze = True
                else:  # [T,H,W]
                    input_tensor = torch.tensor(input_tensor)
            else:
                input_tensor = numpy_to_torch(input_tensor)
            is_numpy = True
        else:
            is_numpy = False

        (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
        step_size = binning_size - overlap_size
        H_final = 1 + np.int16((H - binning_size) / step_size)
        W_final = 1 + np.int16((W - binning_size) / step_size)

        if shape_len == 2:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            column_cumsum = torch.cat(
                (torch.zeros((1, 1, H, 1)).to(input_tensor.device), torch.cumsum(input_tensor, -1)), -1)
            column_binning = column_cumsum[..., binning_size::step_size] - column_cumsum[..., :-binning_size:step_size]
            row_cumsum = torch.cat(
                (torch.zeros((1, 1, 1, W_final)).to(input_tensor.device), torch.cumsum(column_binning, -2)), -2)
            binned_matrix_final = row_cumsum[..., binning_size::step_size, :] - row_cumsum[...,
                                                                                :-binning_size:step_size, :]
            binned_matrix_final = binned_matrix_final.squeeze(0).squeeze(0)
        elif shape_len == 3:
            column_cumsum = torch.cat((torch.zeros((T, H, 1)).to(input_tensor.device), torch.cumsum(input_tensor, -1)),
                                      -1)
            column_binning = column_cumsum[..., binning_size::step_size] - column_cumsum[..., :-binning_size:step_size]
            row_cumsum = torch.cat(
                (torch.zeros((T, 1, W_final)).to(input_tensor.device), torch.cumsum(column_binning, -2)), -2)
            binned_matrix_final = row_cumsum[..., binning_size::step_size, :] - row_cumsum[...,
                                                                                :-binning_size:step_size, :]
        elif shape_len == 4:
            column_cumsum = torch.cat(
                (torch.zeros((T, C, H, 1)).to(input_tensor.device), torch.cumsum(input_tensor, -1)), -1)
            column_binning = column_cumsum[..., binning_size::step_size] - column_cumsum[..., :-binning_size:step_size]
            row_cumsum = torch.cat(
                (torch.zeros((T, C, 1, W_final)).to(input_tensor.device), torch.cumsum(column_binning, -2)), -2)
            binned_matrix_final = row_cumsum[..., binning_size::step_size, :] - row_cumsum[...,
                                                                                :-binning_size:step_size, :]
        elif shape_len == 5:
            column_cumsum = torch.cat(
                (torch.zeros((B, T, C, H, 1)).to(input_tensor.device), torch.cumsum(input_tensor, -1)), -1)
            column_binning = column_cumsum[..., binning_size::step_size] - column_cumsum[..., :-binning_size:step_size]
            row_cumsum = torch.cat(
                (torch.zeros((B, T, C, 1, W_final)).to(input_tensor.device), torch.cumsum(column_binning, -2)), -2)
            binned_matrix_final = row_cumsum[..., binning_size::step_size, :] - row_cumsum[...,
                                                                                :-binning_size:step_size, :]

        ### Normalize to perform mean and not sum: ###
        binned_matrix_final = binned_matrix_final / binning_factor ** 2

        ### Return to numpy if necessary: ###
        if flag_squeeze:
            binned_matrix_final = binned_matrix_final.squeeze(0)
        if is_numpy:
            binned_matrix_final = torch_to_numpy(binned_matrix_final)
            binned_matrix_final = binned_matrix_final.astype(dtype)
        return binned_matrix_final

    ### RGB-BW: ###
    if flag_RGB2BW:
        if type(frames_to_stabilize) == list:
            dtype = frames_to_stabilize[0].dtype
            frames_to_stabilize = [RGB2BW(current_frame).astype(dtype) for current_frame in frames_to_stabilize]
        else:
            dtype = frames_to_stabilize.dtype
            frames_to_stabilize = RGB2BW(frames_to_stabilize).dtype

    ### Convert input frames to numpy array and extract dimensions ###
    # TODO: there is no real need to turn them into one numpy array if this is a list....i need to rewrite this function pretty much.
    gtic()
    frames_np, single_frame_input, T, H, W, C = to_numpy_array(frames_to_stabilize)
    gtoc('to numpy array')

    ### Convert reference frame to numpy array if provided ###
    if reference_frame is None:
        reference_frame = frames_np[0]
    else:
        reference_frame, _, _, _, _, _ = to_numpy_array(reference_frame)

    ### Downsample frames if flag is set ###
    # TODO: on CPU this is very inefficient and requires going to pytorch and back and it's stupid. rewrite this to use numpy or opencv or VPI functions.
    gtic()
    if flag_downsample_frames:
        frames_np_binned = fast_binning_2D_PixelBinning(frames_np, binning_factor, 0)
        reference_frame_binned = fast_binning_2D_PixelBinning(reference_frame, binning_factor, 0)
    else:
        frames_np_binned = frames_np
        reference_frame_binned = reference_frame
    gtoc('binning')

    ### Initialize lists to store stabilized frames and homography matrices ###
    stabilized_frames_to_reference_list = []
    homography_matrices_list = []

    ### Set interpolation method ###
    if interpolation == 'nearest':
        interpolation_flag = cv2.INTER_NEAREST
    elif interpolation == 'bilinear':
        interpolation_flag = cv2.INTER_LINEAR
    else:
        raise ValueError(f"Invalid interpolation method: {interpolation}")

    ### Function to calculate homography matrix ###
    def calculate_homography(current_frame, reference_frame, last_frame_features, flag_matching_crosscheck):
        if flag_registration_algorithm == 'iterative_reweighing':
            return calculate_frame_movement_homography_mat_dudy_iterative_reweighing(
                current_frame, reference_frame,
                number_of_features_for_ORB=number_of_features_for_ORB,
                last_frame_features=last_frame_features,
                max_iterations=max_iterations,
                inlier_threshold=inlier_threshold,
                c=c
            )
        elif flag_registration_algorithm == 'regular':
            return calculate_frame_movement_homography_mat_dudy(
                current_frame, reference_frame,
                number_of_features_for_ORB=number_of_features_for_ORB,
                last_frame_features=last_frame_features,
                flag_matching_crosscheck=flag_matching_crosscheck
            )
        else:
            raise ValueError(f"Invalid registration algorithm: {flag_registration_algorithm}")

    ### Loop over frames and stabilize each to reference frame ###
    for t in range(T):
        current_frame = frames_np_binned[t] if T > 1 else frames_np_binned

        ### Ensure correct shape for homography calculation ###
        # TODO: change this to a function which takes care of everything, including if we're dealing with a list
        if current_frame.ndim == 4:
            current_frame = current_frame.squeeze(0)
        if reference_frame_binned.ndim == 4:
            reference_frame_binned = reference_frame_binned.squeeze(0)

        ### Get Homography Matrix ###
        # gtic()
        homography_matrix, last_frame_features = calculate_homography(current_frame,
                                                                      reference_frame_binned,
                                                                      last_frame_features,
                                                                      flag_matching_crosscheck)
        # gtoc('calculate homography')
        homography_matrix = adjust_binned_homography_matrix(homography_matrix, binning_factor)
        homography_matrices_list.append(homography_matrix)

        ### Ensure correct shape for warping ###
        original_frame = frames_np[t] if T > 1 else frames_np
        if original_frame.ndim == 4:
            original_frame = original_frame.squeeze(0)

        ### Warp current frame to reference if flag_perform_interpolation is True ###
        gtic()
        if flag_perform_interpolation and homography_matrix is not None:
            stabilized_frame = cv2.warpPerspective(original_frame, homography_matrix, (W, H), flags=interpolation_flag)
        else:
            stabilized_frame = original_frame
        gtoc('warp')

        ### Append stabilized frame to list ###
        stabilized_frames_to_reference_list.append(stabilized_frame)

    ### Convert stabilized frames to specified format ###
    stabilized_frames_to_reference_list = to_output_format(stabilized_frames_to_reference_list, flag_output_array_form)

    ### Handle single frame output ###
    if single_frame_input:
        stabilized_frames_to_reference_list = stabilized_frames_to_reference_list[0]
        homography_matrices_list = homography_matrices_list[0]

    return stabilized_frames_to_reference_list, homography_matrices_list, last_frame_features


def warp_images_to_first_frame(images, homographies_to_first_frame):
    # TODO: extend this as much as possible
    """
    Warp each image in a sequence to the perspective of the first frame.

    Parameters:
    - images (np.ndarray): The sequence of images to warp (T, H, W).
    - homographies_to_first_frame (list or np.ndarray): List of homography matrices to the first frame.

    Returns:
    - warped_images (np.ndarray): The sequence of images warped to the first frame's perspective.
    """

    T, H, W = images.shape  ### Extract the dimensions of the input images
    warped_images = np.zeros_like(images)  ### Initialize an array to store the warped images

    ### Warp each image to the first frame's perspective ###
    for t in range(T):  ### Iterate over each image in the sequence
        warped_images[t] = cv2.warpPerspective(images[t], homographies_to_first_frame[t], (W, H))  ### Apply homography

    return warped_images  ### Return the array of warped images


class OpticalFlowFromHomography:
    """
    Calculate optical flow between frames using homography matrices.

    Attributes:
    - height (int): Height of the images.
    - width (int): Width of the images.
    - y (np.ndarray): Meshgrid y-coordinates.
    - x (np.ndarray): Meshgrid x-coordinates.
    - coords (np.ndarray): Homogeneous coordinates for each pixel.
    """

    def __init__(self, image_shape):
        """
        Initialize the optical flow calculator with the image shape.

        Parameters:
        - image_shape (tuple): Shape of the image (height, width).
        """

        self.height, self.width = image_shape  ### Store the image dimensions
        self.y, self.x = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')  ### Create meshgrid
        self.coords = np.stack([self.x, self.y, np.ones_like(self.x)], axis=-1).reshape(-1,
                                                                                        3).T  ### Create homogeneous coordinates

    def compute_flow(self, homography):
        """
        Compute the optical flow for a single homography matrix.

        Parameters:
        - homography (np.ndarray): Homography matrix.

        Returns:
        - flow (np.ndarray): Optical flow vectors (height, width, 2).
        """

        homography_inv = np.linalg.inv(homography)  ### Compute the inverse of the homography matrix

        ### Transform coordinates using the homography matrix ###
        new_coords = homography @ self.coords  ### Apply the homography transformation
        new_coords /= new_coords[2, :]  ### Normalize the coordinates

        ### Calculate optical flow ###
        flow_x = new_coords[0, :].reshape(self.height, self.width) - self.x  ### Calculate x-component of flow
        flow_y = new_coords[1, :].reshape(self.height, self.width) - self.y  ### Calculate y-component of flow

        return np.stack([flow_x, flow_y], axis=-1)  ### Return the optical flow vectors

    def compute_flow_sequence(self, images, homography_matrices):
        """
        Compute the optical flow for a sequence of homography matrices.

        Parameters:
        - images (np.ndarray): The sequence of images (T, H, W).
        - homography_matrices (list or np.ndarray): List of homography matrices.

        Returns:
        - optical_flows (np.ndarray): Optical flow vectors for each frame (T, H, W, 2).
        """

        T, H, W = images.shape  ### Extract the dimensions of the input images
        if (H, W) != (self.height, self.width):
            raise ValueError("Image shape does not match initialized shape")  ### Check for shape mismatch

        optical_flows = np.zeros((T, H, W, 2))  ### Initialize an array to store the optical flows

        ### Compute optical flow for each frame ###
        for t in range(T):  ### Iterate over each frame
            optical_flows[t] = self.compute_flow(homography_matrices[t])  ### Compute optical flow

        return optical_flows  ### Return the optical flow vectors

    def visualize_flow(self, optical_flow, frame_idx=0, use_quiver=True, fps=10):
        """
        Visualize the optical flow for a specific frame.

        Parameters:
        - optical_flow (np.ndarray): Optical flow vectors (T, H, W, 2).
        - frame_idx (int): Index of the frame to visualize.
        - use_quiver (bool): Flag to use quiver plot.
        - fps (int): Frames per second for the visualization.
        """

        if use_quiver:
            self._plot_optical_flow_quiver(optical_flow, frame_idx, fps)  ### Plot using quiver
        else:
            self._plot_optical_flow_intensity(optical_flow, frame_idx, fps)  ### Plot using intensity

    def visualize_flow_for_homography(self, homographies, use_quiver=True, fps=10):
        """
        Visualize the optical flow for given homography matrices.

        Parameters:
        - homographies (np.ndarray): Homography matrices.
        - use_quiver (bool): Flag to use quiver plot.
        - fps (int): Frames per second for the visualization.
        """

        if homographies.ndim == 2:
            optical_flow = self.compute_flow(homographies)  ### Compute optical flow for a single homography
            if use_quiver:
                self._plot_optical_flow_quiver([optical_flow], fps=fps)  ### Plot using quiver
            else:
                self._plot_optical_flow_intensity([optical_flow], fps=fps)  ### Plot using intensity
        elif homographies.ndim == 3:
            optical_flows = [self.compute_flow(homographies[i]) for i in
                             range(homographies.shape[0])]  ### Compute optical flow for each homography
            if use_quiver:
                self._plot_optical_flow_quiver(optical_flows, fps=fps)  ### Plot using quiver
            else:
                self._plot_optical_flow_intensity(optical_flows, fps=fps)  ### Plot using intensity
        else:
            raise ValueError("Invalid homography matrix dimensions")  ### Raise error for invalid dimensions

    def _plot_optical_flow_quiver(self, optical_flows, frame_idx=None, fps=10):
        """
        Plot optical flow using quiver plot.

        Parameters:
        - optical_flows (list or np.ndarray): List of optical flow vectors.
        - frame_idx (int): Index of the frame to visualize.
        - fps (int): Frames per second for the visualization.
        """

        fig, ax = plt.subplots()  ### Create a figure and axis
        q = ax.quiver(optical_flows[0][..., 0], optical_flows[0][..., 1])  ### Initialize quiver plot
        ax.invert_yaxis()  ### Invert the y-axis for proper orientation
        ax.set_title("Optical Flow Quiver")  ### Set the title

        for i in range(len(optical_flows)):  ### Iterate over each frame
            q.set_UVC(optical_flows[i][..., 0], optical_flows[i][..., 1])  ### Update quiver plot
            ax.set_title(f"Optical Flow Frame {i}")  ### Update title
            plt.pause(1 / fps)  ### Pause for the specified frame rate

        plt.show()  ### Display the plot

    def _plot_optical_flow_intensity(self, optical_flows, frame_idx=None, fps=10):
        """
        Plot optical flow using intensity plot.

        Parameters:
        - optical_flows (list or np.ndarray): List of optical flow vectors.
        - frame_idx (int): Index of the frame to visualize.
        - fps (int): Frames per second for the visualization.
        """

        fig, ax = plt.subplots()  ### Create a figure and axis
        intensity = np.sqrt(
            optical_flows[0][..., 0] ** 2 + optical_flows[0][..., 1] ** 2)  ### Compute intensity of flow
        im = ax.imshow(intensity, cmap='gray')  ### Initialize intensity plot
        ax.set_title("Optical Flow Intensity")  ### Set the title

        for i in range(len(optical_flows)):  ### Iterate over each frame
            intensity = np.sqrt(optical_flows[i][..., 0] ** 2 + optical_flows[i][..., 1] ** 2)  ### Update intensity
            im.set_array(intensity)  ### Update intensity plot
            ax.set_title(f"Optical Flow Intensity Frame {i}")  ### Update title
            plt.pause(1 / fps)  ### Pause for the specified frame rate

        plt.show()  ### Display the plot


def transform_point_using_homography_matrix(homography_matrix, point):
    """
    Transform a point using a given homography matrix.

    Parameters:
    - homography_matrix (np.ndarray): The homography matrix.
    - point (tuple or list): The point (x, y) to transform.

    Returns:
    - point_transformed (np.ndarray): The transformed point.
    """

    point_array = np.float32([[point]])  ### Convert point to required shape and type
    point_transformed = cv2.perspectiveTransform(point_array, homography_matrix)  ### Apply homography transformation

    return point_transformed  ### Return the transformed point


def transform_points_using_homography_matrix(homography_matrix, points):
    """
    Transform a set of points using a given homography matrix.

    Parameters:
    - homography_matrix (np.ndarray): The homography matrix.
    - points (list or np.ndarray): The points to transform.

    Returns:
    - points_transformed (np.ndarray): The transformed points.
    """

    points_array = np.float32(points).reshape(-1, 1, 2)  ### Convert points to required shape and type
    points_transformed = cv2.perspectiveTransform(points_array, homography_matrix)  ### Apply homography transformation
    points_transformed = points_transformed.reshape(-1, 2)  ### Reshape the output to match input shape

    return points_transformed  ### Return the transformed points


def calculate_shifts_between_points_sets(original_points, transformed_points):
    """
    Calculate the shifts between corresponding points in two sets.

    Parameters:
    - original_points (list or np.ndarray): The original set of points.
    - transformed_points (list or np.ndarray): The transformed set of points.

    Returns:
    - min_shift (float): The minimum shift between corresponding points.
    - max_shift (float): The maximum shift between corresponding points.
    - avg_shift (float): The average shift between corresponding points.
    """

    ### Convert to numpy arrays for easy manipulation ###
    original_points = np.array(original_points)  ### Convert original points to numpy array
    transformed_points = np.array(transformed_points)  ### Convert transformed points to numpy array

    ### Calculate the shifts between corresponding points ###
    shifts = np.linalg.norm(transformed_points - original_points, axis=1)  ### Calculate the Euclidean distance

    ### Calculate minimum, maximum, and average shift ###
    min_shift = np.min(shifts)  ### Calculate the minimum shift
    max_shift = np.max(shifts)  ### Calculate the maximum shift
    avg_shift = np.mean(shifts)  ### Calculate the average shift

    return min_shift, max_shift, avg_shift  ### Return the calculated shifts


def plot_matching_features(image1, image2):
    """
    Plot matching features between two images using ORB detector and BFMatcher.

    Parameters:
    - image1 (np.ndarray): The first image.
    - image2 (np.ndarray): The second image.

    Returns:
    - None
    """

    ### Initialize the ORB detector
    orb_detector = cv2.ORB_create()

    ### Find the keypoints and descriptors with ORB for both images
    keypoints1, descriptors1 = orb_detector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb_detector.detectAndCompute(image2, None)

    ### Create BFMatcher object with Hamming distance and cross-check enabled
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ### Match descriptors between the two images
    matches = bf_matcher.match(descriptors1, descriptors2)

    ### Sort the matches based on their distance (best matches first)
    matches = sorted(matches, key=lambda match: match.distance)

    ### Draw the first 10 matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    ### Display the matched image
    plt.figure(figsize=(10, 5))  ### Set the figure size
    plt.imshow(matched_image)  ### Display the matched image
    plt.title('Matching Features')  ### Set the title of the plot
    plt.show()  ### Show the plot


def do_registration(source_point, homography_matrix, trajectory_line_end, frame_bbox):
    """
    Perform point registration using a homography matrix.

    Parameters:
    - source_point (list): Source point coordinates [x, y].
    - homography_matrix (np.ndarray): Homography matrix.
    - trajectory_line_end (list): End point of the trajectory line [x, y].
    - frame_bbox (tuple): Bounding box of the frame (xmin, ymin, xmax, ymax).

    Returns:
    - Tuple of registered source point and trajectory line end point or (None, None) if points are outside the frame.
    """
    ### Apply homography transformation to the source point and trajectory line end ###
    source_point = transform_point_using_homography_matrix(homography_matrix.squeeze(),
                                                           [source_point[0], source_point[1]]).squeeze().astype(int)
    trajectory_line_end = transform_point_using_homography_matrix(homography_matrix.squeeze(), [trajectory_line_end[0],
                                                                                                trajectory_line_end[
                                                                                                    1]]).squeeze().astype(
        int)

    ### Check if the transformed points are within the frame bounding box ###
    if is_point_in_bbox(source_point, frame_bbox) and is_point_in_bbox(trajectory_line_end, frame_bbox):
        return source_point, trajectory_line_end  ### Return the registered points if they are within the frame ###
    else:
        return None, None  ### Return None if points are outside the frame ###


def stabilize_CCC(current_frame, last_frame):
    """
    Stabilize the current frame using circular cross-correlation (CCC) method.

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The last reference frame.

    Returns:
    - Stabilized current frame.
    """
    ### Convert frames to tensor format for CCC ###
    reference_tensor = numpy_to_torch(last_frame).unsqueeze(0)
    input_tensor = numpy_to_torch(current_frame).unsqueeze(0)

    ### Perform alignment using CCC ###
    current_frame_stabilized_to_last_torch, _, _, _ = align_to_reference_frame_circular_cc(input_tensor,
                                                                                           reference_tensor)

    ### Check if stabilization was successful ###
    if current_frame_stabilized_to_last_torch.any():
        ### Convert stabilized frame back to numpy and BGR format ###
        current_frame_stabilized_to_last = cv2.cvtColor(
            current_frame_stabilized_to_last_torch.squeeze().cpu().detach().numpy(), cv2.COLOR_GRAY2BGR)
        return current_frame_stabilized_to_last  ### Return stabilized frame ###
    else:
        return current_frame  ### Return original frame if stabilization failed ###


def stabilize_FeatureBased_OpenCV(current_frame, last_frame, last_frame_features=False):
    """
    Stabilize the current frame using feature-based OpenCV method.

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The last reference frame.
    - last_frame_features (bool or list): Features from the last frame.

    Returns:
    - Stabilized current frame and updated last frame features.
    """
    ### Get Homography Matrix ###
    homography_matrix, last_frame_features = calculate_frame_movement_homography_mat(
        current_frame, last_frame,
        number_of_features_for_ORB=NUMBER_OF_FEATURES_FOR_ORB,
        last_frame_features=last_frame_features
    )

    ### Check if homography matrix was calculated successfully ###
    if np.any(homography_matrix):
        ### Warp current frame to last frame's perspective ###
        current_frame_stabilized_to_last = cv2.warpPerspective(current_frame, homography_matrix,
                                                               (current_frame.shape[1], current_frame.shape[0]))
        return current_frame_stabilized_to_last, last_frame_features  ### Return stabilized frame and updated features ###
    else:
        return current_frame, last_frame_features  ### Return original frame and features if stabilization failed ###


def stabilize_FeatureBased_OpenCV_dudy_homography(current_frame, last_frame, last_frame_features=False,
                                                  number_of_features=1000):
    """
    Stabilize the current frame using Dudy's feature-based OpenCV method.

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The last reference frame.
    - last_frame_features (bool or list): Features from the last frame.
    - number_of_features (int): Number of ORB features to detect.

    Returns:
    - Stabilized current frame, updated last frame features, and homography matrix.
    """
    ### Get Homography Matrix ###
    homography_matrix, last_frame_features = calculate_frame_movement_homography_mat_dudy(
        current_frame, last_frame,
        number_of_features_for_ORB=number_of_features,
        last_frame_features=last_frame_features
    )

    ### Check if homography matrix was calculated successfully ###
    if np.any(homography_matrix):
        ### Warp current frame to last frame's perspective ###
        current_frame_stabilized_to_last = cv2.warpPerspective(current_frame, homography_matrix,
                                                               (current_frame.shape[1], current_frame.shape[0]))
        return current_frame_stabilized_to_last, last_frame_features, homography_matrix  ### Return stabilized frame, updated features, and homography matrix ###
    else:
        return current_frame, last_frame_features, homography_matrix  ### Return original frame, features, and homography matrix if stabilization failed ###


def stabilize_FeatureBased_OpenCV_dudy_homography_iterative_reweighing(current_frame, last_frame,
                                                                       last_frame_features=False,
                                                                       number_of_features=1000, max_iterations=5,
                                                                       inlier_threshold=2.5):
    """
    Stabilize the current frame using Dudy's iterative reweighing feature-based OpenCV method.

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The last reference frame.
    - last_frame_features (bool or list): Features from the last frame.
    - number_of_features (int): Number of ORB features to detect.
    - max_iterations (int): Maximum number of iterations for reweighing.
    - inlier_threshold (float): Threshold to identify inliers.

    Returns:
    - Stabilized current frame, updated last frame features, and homography matrix.
    """
    ### Get Homography Matrix ###
    homography_matrix, last_frame_features = calculate_frame_movement_homography_mat_dudy_iterative_reweighing(
        current_frame, last_frame,
        number_of_features_for_ORB=number_of_features,
        last_frame_features=last_frame_features,
        max_iterations=max_iterations,
        inlier_threshold=inlier_threshold
    )

    ### Check if homography matrix was calculated successfully ###
    if np.any(homography_matrix):
        ### Warp current frame to last frame's perspective ###
        current_frame_stabilized_to_last = cv2.warpPerspective(current_frame, homography_matrix,
                                                               (current_frame.shape[1], current_frame.shape[0]))
        return current_frame_stabilized_to_last, last_frame_features, homography_matrix  ### Return stabilized frame, updated features, and homography matrix ###
    else:
        return current_frame, last_frame_features, homography_matrix  ### Return original frame, features, and homography matrix if stabilization failed ###


def stabilize_FeatureBased_OpenCV_dudy(current_frame, last_frame, last_frame_features=False):
    """
    Stabilize the current frame using Dudy's feature-based OpenCV method.

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The last reference frame.
    - last_frame_features (bool or list): Features from the last frame.

    Returns:
    - Stabilized current frame and updated last frame features.
    """
    ### Get Homography Matrix ###
    homography_matrix, last_frame_features = calculate_frame_movement_homography_mat_dudy(
        current_frame, last_frame,
        number_of_features_for_ORB=NUMBER_OF_FEATURES_FOR_ORB,
        last_frame_features=last_frame_features
    )

    ### Check if homography matrix was calculated successfully ###
    if np.any(homography_matrix):
        ### Warp current frame to last frame's perspective ###
        current_frame_stabilized_to_last = cv2.warpPerspective(current_frame, homography_matrix,
                                                               (current_frame.shape[1], current_frame.shape[0]))
        return current_frame_stabilized_to_last, last_frame_features  ### Return stabilized frame and updated features ###
    else:
        return current_frame, last_frame_features  ### Return original frame and features if stabilization failed ###


def stabilize_FeatureBased_OpenCV_Batch(current_frames_list, reference_frame=None):
    """
    Stabilize a batch of frames using feature-based OpenCV method.

    Parameters:
    - current_frames_list (list): List of frames to be stabilized.
    - reference_frame (np.ndarray, optional): Reference frame for stabilization. Defaults to the first frame in the list.

    Returns:
    - List of stabilized frames.
    """
    last_frame_features = [None, None, None]  ### Initialize last frame features ###
    if reference_frame is None:
        reference_frame = current_frames_list[0]  ### Use first frame as reference if not provided ###
    frames_stabilized_list = []  ### Initialize list to store stabilized frames ###

    ### Loop over frames and stabilize each to reference frame ###
    for frame_index in np.arange(1, len(current_frames_list)):
        current_frame = current_frames_list[frame_index]  ### Get current frame to stabilize ###
        current_frame_stabilized, last_frame_features = stabilize_FeatureBased_OpenCV(current_frame,
                                                                                      reference_frame)  ### Stabilize current frame ###
        frames_stabilized_list.append(current_frame_stabilized)  ### Append stabilized frame to list ###

    return frames_stabilized_list  ### Return list of stabilized frames ###


def adjust_binned_homography_matrix(H_binned, binning_factor):
    """
    Adjust the homography matrix for binning.

    Parameters:
    - H_binned (np.ndarray): Homography matrix for binned images.
    - binning_factor (int): Binning factor used for binning.

    Returns:
    - H_original (np.ndarray): Adjusted homography matrix for the original image.
    """
    S = np.array([
        [binning_factor, 0, 0],
        [0, binning_factor, 0],
        [0, 0, 1]
    ])  ### Create the scaling matrix

    H_original = np.dot(np.dot(S, H_binned),
                        np.linalg.inv(S))  ### Calculate the homography matrix for the original image

    return H_original


def bin_2x2(image: np.ndarray) -> np.ndarray:
    """
    Perform 2x2 binning on the input image.

    Parameters:
    - image (np.ndarray): Input image array of shape (height, width, channels).

    Returns:
    - binned_image (np.ndarray): Binned image array of shape (height//2, width//2, channels).
    """
    ### If the image is grayscale, add a channel dimension for consistency
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    height, width, channels = image.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Image dimensions must be even for 2x2 binning.")

    ### Reshape and average
    binned_image = image.reshape((height // 2, 2, width // 2, 2, channels)).mean(axis=(1, 3))

    ### If the original image was grayscale, remove the added channel dimension
    if binned_image.shape[-1] == 1:
        binned_image = binned_image[:, :, 0]

    return binned_image


def stabilize_featurebased_opencv_binned_2x2_with_H_matrix(current_frame, last_frame, last_frame_features=False):
    """
    Stabilize the current frame based on features matched with the last frame using OpenCV and 2x2 binning,
    and return the homography matrix.

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The previous frame used for matching features.
    - last_frame_features (bool or list): List containing the flag for using features from the last frame and the features themselves.

    Returns:
    - current_frame_stabilized_to_last (np.ndarray): The stabilized current frame.
    - last_frame_features (list): Updated list of last frame features.
    - homography_matrix (np.ndarray): The homography matrix used for stabilization.
    """

    ### Get Homography Matrix for binned images: ###
    current_frame_binned = bin_2x2(current_frame).astype(np.uint8)
    last_frame_binned = bin_2x2(last_frame).astype(np.uint8)
    homography_matrix, last_frame_features = calculate_frame_movement_homography_mat(current_frame_binned,
                                                                                     last_frame_binned,
                                                                                     number_of_features_for_ORB=NUMBER_OF_FEATURES_FOR_ORB,
                                                                                     last_frame_features=last_frame_features)
    ### Adjust homography for binning
    homography_matrix = adjust_binned_homography_matrix(homography_matrix, binning_factor=2)

    ### Interpolate Stabilized Image: ###
    if np.any(homography_matrix):
        current_frame_stabilized_to_last = cv2.warpPerspective(current_frame, homography_matrix,
                                                               (current_frame.shape[1], current_frame.shape[0]),
                                                               flags=cv2.INTER_NEAREST)

        return current_frame_stabilized_to_last, last_frame_features, homography_matrix

    return current_frame, last_frame_features, homography_matrix


# def calculate_frame_movement_homography_mat(frame_to_stabilize, refrence_frame, number_of_features_for_ORB=500, last_frame_features = [None, None, None]):
#     ### Initialize Detector and Matcher: ###
#     detector = cv2.ORB_create(number_of_features_for_ORB)
#     matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#     # last_frame_features is a list built like this [return_the_features(bool), the_frame_kp(nd.array), the_descriptor_of_thos_kp(nd.array)]
#     ### Detect Features and Descriptors: ###
#     gtic()
#     kp1, des1 = detector.detectAndCompute(frame_to_stabilize, None)
#     if np.any(last_frame_features[1]):
#         kp2, des2 = last_frame_features[1:]
#     else:
#         kp2, des2 = detector.detectAndCompute(refrence_frame, None)
#     gtoc('get features')

#     ### Match features: ###
#     try:
#         gtic()
#         matches = matcher.match(des1, des2)
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#         gtoc('matching')
#         gtic()
#         H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         gtoc('find homography')

#         if last_frame_features[0]:
#             last_frame_features[1] = kp1
#             last_frame_features[2] = des1
#             return H, last_frame_features

#         return H, [None, None, None]

#     except:

#         if last_frame_features[0]:
#             last_frame_features[1] = kp1
#             last_frame_features[2] = des1
#             return None, last_frame_features

#         return None, None


def adjust_binned_homography_matrix(H_binned, binning_factor):
    """
    Adjust the homography matrix for binning.

    Parameters:
    - H_binned (np.ndarray): Homography matrix for binned images.
    - binning_factor (int): Binning factor used for binning.

    Returns:
    - H_original (np.ndarray): Adjusted homography matrix for the original image.
    """
    S = np.array([
        [binning_factor, 0, 0],
        [0, binning_factor, 0],
        [0, 0, 1]
    ])  ### Create the scaling matrix

    H_original = np.dot(np.dot(S, H_binned),
                        np.linalg.inv(S))  ### Calculate the homography matrix for the original image

    return H_original


def bin_2x2(image: np.ndarray) -> np.ndarray:
    """
    Perform 2x2 binning on the input image.

    Parameters:
    - image (np.ndarray): Input image array of shape (height, width, channels).

    Returns:
    - binned_image (np.ndarray): Binned image array of shape (height//2, width//2, channels).
    """
    ### If the image is grayscale, add a channel dimension for consistency
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    height, width, channels = image.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Image dimensions must be even for 2x2 binning.")

    ### Reshape and average
    binned_image = image.reshape((height // 2, 2, width // 2, 2, channels)).mean(axis=(1, 3))

    ### If the original image was grayscale, remove the added channel dimension
    if binned_image.shape[-1] == 1:
        binned_image = binned_image[:, :, 0]

    return binned_image


def stabilize_featurebased_opencv_binned_2x2_with_H_matrix(current_frame, last_frame, last_frame_features=False):
    """
    Stabilize the current frame based on features matched with the last frame using OpenCV and 2x2 binning,
    and return the homography matrix.

    Parameters:
    - current_frame (np.ndarray): The current frame to be stabilized.
    - last_frame (np.ndarray): The previous frame used for matching features.
    - last_frame_features (bool or list): List containing the flag for using features from the last frame and the features themselves.

    Returns:
    - current_frame_stabilized_to_last (np.ndarray): The stabilized current frame.
    - last_frame_features (list): Updated list of last frame features.
    - homography_matrix (np.ndarray): The homography matrix used for stabilization.
    """

    ### Get Homography Matrix for binned images: ###
    current_frame_binned = bin_2x2(current_frame).astype(np.uint8)
    last_frame_binned = bin_2x2(last_frame).astype(np.uint8)
    homography_matrix, last_frame_features = calculate_frame_movement_homography_mat(current_frame_binned,
                                                                                     last_frame_binned,
                                                                                     number_of_features_for_ORB=NUMBER_OF_FEATURES_FOR_ORB,
                                                                                     last_frame_features=last_frame_features)
    ### Adjust homography for binning
    homography_matrix = adjust_binned_homography_matrix(homography_matrix, binning_factor=2)

    ### Interpolate Stabilized Image: ###
    if np.any(homography_matrix):
        current_frame_stabilized_to_last = cv2.warpPerspective(current_frame, homography_matrix,
                                                               (current_frame.shape[1], current_frame.shape[0]),
                                                               flags=cv2.INTER_NEAREST)

        return current_frame_stabilized_to_last, last_frame_features, homography_matrix

    return current_frame, last_frame_features, homography_matrix


