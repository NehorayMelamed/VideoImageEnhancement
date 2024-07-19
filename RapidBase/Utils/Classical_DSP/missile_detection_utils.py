
import numpy as np
import matplotlib.pyplot as plt
import cv2
from RapidBase.Utils.IO.tic_toc import *
import csv
import os
from threading import Thread
import threading
# from blobs import *
from pathlib import Path
import re
from RapidBase.Utils.IO.Path_and_Reading_utils import path_create_path_if_none_exists


from typing import Tuple
import numpy as np
from scipy import ndimage
from skimage import measure, morphology
import cv2
# from RapidBase.import_all import *

# from settings import *


def get_blob_indices(binary_mask):
    """Gets a binary mask of an image and returns the indices of the true parts by number of objects
    differentiates between objects by checking adjacent pixels (up, under, left, right)

    Args:
        binary_mask (numpy_array): numpy array in shape of (height, width) of ones and zeros (true, false)

    Returns:
        list of tuples: holds each object in an index of itself with which is a list of tuples of integers of the
        y, x (height, width) of the pixels
    """
    labeled_mask, num_labels = ndimage.label(binary_mask)
    blob_indices = np.argwhere(labeled_mask == (np.bincount(labeled_mask.flatten())[1:].argmax()+ 1))

    # blob_indices = []
    # for label in range(1, num_labels + 1):
    #     blob_indices.append(np.argwhere(labeled_mask == label))
    return blob_indices

def get_blob_centroid(blob: np.ndarray) -> Tuple[int, int]:
    """Calculates the centroid of a blob of coordinates.

    Args:
        blob (np.ndarray): A 2D array of shape (n, 2) containing (x, y) coordinates.

    Returns:
        tuple: A tuple (centroid_y, centroid_x) representing the centroid.
    """
    return np.mean(blob[:, 0]).astype(np.uint32), np.mean(blob[:, 1]).astype(np.uint32)


def get_blob_dimensions(blob: np.ndarray) -> Tuple[int, int]:
    """Calculates the minimum height and width of a blob of coordinates.

    Args:
        blob (np.ndarray): A 2D array of shape (n, 2) containing (x, y) coordinates.

    Returns:
        tuple: A tuple (min_height, min_width) representing the minimum width and height.
    """
    return np.max(blob[:, 0]) - np.min(blob[:, 0]), np.max(blob[:, 1]) - np.min(blob[:, 1])


def get_blob_bbox(blob: np.ndarray) -> Tuple[int, int, int, int]:
    """Finds a bounding box wraps the given blob

    Args:
        blob: (np.ndarray): A 2D array of shape (n, 2) containing (x, y) coordinates.

    Returns:
        tuple: A tuple representing the bounding box in xyxy format
    """
    centroid = get_blob_centroid(blob)
    height, width = get_blob_dimensions(blob)
    x0 = centroid[1] - width // 2
    y0 = centroid[0] - height // 2
    x1 = centroid[1] + width // 2
    y1 = centroid[0] + height // 2
    return x0, y0, x1, y1


def get_hot_zone_blob_bbox(blob: np.ndarray) -> Tuple[int, int, int, int]:
    """Finds a bounding box wraps the given blob

    Args:
        blob: (np.ndarray): A 2D array of shape (n, 2) containing (x, y) coordinates.

    Returns:
        tuple: A tuple representing the bounding box in xyxy format
    """
    centroid = get_blob_centroid(blob)
    height, width = get_blob_dimensions(blob)
    x0 = centroid[1] - width // 2
    y0 = centroid[0] - height // 2
    x1 = centroid[1] + width // 2
    y1 = centroid[0] + height // 2
    return x0 - 10 , y0 - 10 , x1 + 10 , y1 + 10



def scale_array_to_range(ndarray: np.ndarray, min_max_values_to_scale_to=(0, 255)):
    """Scaling an array to a given min max values (defaults to 0-1)
    Note: we add to max 1e-16 because it deals with the case that the entire array is the same value (a division by 0)
    """
    return (ndarray - ndarray.min()) / (ndarray.max() - ndarray.min() + 1e-16) * \
        (min_max_values_to_scale_to[1] - min_max_values_to_scale_to[0]) + min_max_values_to_scale_to[0]


def euclidean_distance(ndarray1: np.ndarray, ndarray2: np.ndarray) -> np.ndarray:
    """Returns a ndarray that represents the distance between the two given arrays using Euclidean distance"""
    return np.sqrt(np.sum((ndarray1 - ndarray2 )**2, axis=-1)).astype(np.uint64)


def min_enclosing_circle_circularity(contour):
    """
    Calculate the circularity of a contour based on the minimum enclosing circle.

    Parameters:
    - contour (np.ndarray): Contour of the object.

    Returns:
    - area_ratio (float): Ratio of the contour's area to the enclosing circle's area.
    """

    ### Calculate the minimum enclosing circle: ###
    (x, y), radius = cv2.minEnclosingCircle(contour)  ### Get the center (x, y) and radius of the enclosing circle
    circle_area = np.pi * (radius ** 2)  ### Calculate the area of the enclosing circle

    ### Calculate the blob's area: ###
    blob_area = cv2.contourArea(contour)  ### Get the area of the contour

    ### Calculate the ratio of the blob's area to the enclosing circle's area: ###
    area_ratio = blob_area / circle_area  ### Compute the area ratio
    # print("perfect circle vs our blob algo: ", area_ratio)
    return area_ratio  ### Return the area ratio



def elliptical_fit_circularity(contour):
    """
    Calculate the circularity of a contour based on the elliptical fit.

    Parameters:
    - contour (np.ndarray): Contour of the object.

    Returns:
    - aspect_ratio (float): Ratio of the minor axis to the major axis of the fitted ellipse.
    """

    ### Fit an ellipse to the contour: ###
    ellipse = cv2.fitEllipse(contour)  ### Fit an ellipse to the contour

    ### Extract the major and minor axis lengths: ###
    axis_major = max(ellipse[1])  ### Get the major axis length
    axis_minor = min(ellipse[1])  ### Get the minor axis length

    ### Calculate the aspect ratio (minor/major); perfect circle = 1: ###
    aspect_ratio = axis_minor / axis_major  ### Compute the aspect ratio
    # print("elipsoid algo: ", aspect_ratio)
    return aspect_ratio  ### Return the aspect ratio



def circularity_by_old_algo(contour):
    """
    Calculate the circularity of a contour using the old algorithm.

    Parameters:
    - contour (np.ndarray): Contour of the object.

    Returns:
    - current_circularity (float): Circularity value based on the contour area and perimeter.
    """

    ### Calculate the area and perimeter of the contour: ###
    area = cv2.contourArea(contour)  ### Get the area of the contour
    perimeter = cv2.arcLength(contour, True)  ### Get the perimeter of the contour

    ### Calculate circularity: ###
    current_circularity = 4 * np.pi * (area / (perimeter ** 2))  ### Compute circularity using the formula
    # print("current algo: ",current_circularity)
    return current_circularity  ### Return the circularity value



def detect_pseudo_circles(image, frame_index, current_image_diff, current_image, circularity_threshold=0.725, csv_writer=None):
    # TODO: why do we have image and current_image???
    """
    Detect pseudo-circles in the image based on circularity threshold.

    Parameters:
    - image (np.ndarray): Input image.
    - frame_index (int): Index of the current frame.
    - current_image_diff (np.ndarray): Image difference used for additional conditions.
    - current_image (np.ndarray): Original current image.
    - circularity_threshold (float): Threshold for circularity.
    - csv_writer (csv.writer, optional): CSV writer for logging results.

    Returns:
    - circle_cXcYRT (list): List containing x, y, radius, frame_index, circularity, average_diff, average_pixel_value.
    """

    circle_cXcYRT = [None]  ### Initialize the result list with None
    max_circularity_in_image = 0  ### Initialize the maximum circularity in the image

    ### Check if image has a single channel: ###
    if image.shape[-1] == 1:
        image = image[:, :, 0]  ### Convert to single channel if necessary

    ### Scale the image to the range of 0-255: ###
    image = scale_array_to_range(image.astype(np.uint8)).astype(np.uint8)  ### Scale the image array

    ### Find contours in the image: ###
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  ### Retrieve contours

    ### Iterate through each contour: ###
    for contour in contours:
        if len(contour) >= 5:  ### Ensure contour has enough points for ellipse fitting
            current_circularity_eliptical = elliptical_fit_circularity \
                (contour)  ### Calculate elliptical fit circularity

            current_circularity = round(current_circularity_eliptical, 3)  ### Round the circularity value
            temp = np.zeros_like \
                (current_image_diff)  ### Create a temporary array for contour drawing. TODO: no need to reinitialize this again and again
            cv2.drawContours(temp, [contour], 0, (255, 255, 255), -1)  ### Draw the contour on the temporary array
            average_diff = np.mean \
                (current_image_diff[temp == 255])  ### Calculate the average difference inside the contour

            temp_image = np.zeros_like \
                (current_image)  ### Create a temporary array for contour drawing. TODO: no need to reinitialize this again and again
            cv2.drawContours(temp_image, [contour], 0, (255, 255, 255), -1)  ### Draw the contour on the temporary array
            cond = np.logical_and(temp_image == 255, current_image > 125)[:, :, 0]  ### Condition for pixel values
            average_pixel_value = np.mean(current_image[cond])  ### Calculate the average pixel value inside the contour

            ### Check if the current contour meets the circularity threshold and other conditions: ###
            if current_circularity >= circularity_threshold and current_circularity > max_circularity_in_image and average_diff > 30:
                (x, y), radius = cv2.minEnclosingCircle(contour)  ### Get the minimum enclosing circle
                circle_cXcYRT = [x, y, max(radius, 2), frame_index, current_circularity, average_diff, average_pixel_value]  ### Update result list
                max_circularity_in_image = current_circularity  ### Update maximum circularity in the image

    return circle_cXcYRT  ### Return the result list


def is_point_in_bbox(point, bounding_box):
    """
    Check if a point is inside a given bounding box.

    Parameters:
    - point (tuple): The point (x, y) to check.
    - bounding_box (tuple): The bounding box defined by (x_min, y_min, x_max, y_max).

    Returns:
    - bool: True if the point is inside the bounding box, False otherwise.
    """

    ### Check if the point lies within the bounding box ###
    return bounding_box[0] <= point[0] <= bounding_box[2] and bounding_box[1] <= point[1] <= bounding_box[3]  ### Return True if the point is inside the bounding box


def get_time_of_videos_in_folder(folder_path):
    """
    Calculate the total time of all videos in a folder.

    Parameters:
    - folder_path (str): The path to the folder containing video files.

    Returns:
    - None
    """

    total_time = 0  ### Initialize total time

    for file_name in os.listdir(folder_path):  ### Iterate over all files in the folder
        video_path = os.path.join(folder_path, file_name)  ### Construct the full path to the video file
        video_capture = cv2.VideoCapture(video_path)  ### Open the video file
        total_time += int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  ### Add the frame count to the total time

    total_time_seconds = total_time / 25  ### Convert total frames to seconds (assuming 25 FPS)
    print("Total time of videos: ", total_time_seconds)  ### Print the total time in seconds



def get_hot_zone(frame):
    """
    Identify the hot zone in a given frame based on a threshold.

    Parameters:
    - frame (np.ndarray): The frame data.

    Returns:
    - y_indices (np.ndarray): The y-coordinates of the hot zone pixels.
    - x_indices (np.ndarray): The x-coordinates of the hot zone pixels.
    """

    ### Identify the hot zone pixels ###
    binary_hot_zone = frame > HOT_ZONE_THRESHOLD  ### Apply the threshold to get a binary mask
    y_indices, x_indices = np.where(binary_hot_zone)  ### Get the indices of the hot zone pixels

    return y_indices, x_indices  ### Return the indices of the hot zone pixels



def get_hot_zone_bbox(hot_zone_frame):
    """
    Calculate the bounding box of the hot zone in a frame.

    Parameters:
    - hot_zone_frame (np.ndarray): The frame data.

    Returns:
    - bbox (np.ndarray): The bounding box of the hot zone.
    """

    ### Identify the hot zone outliers ###
    hot_zone_outliers = hot_zone_frame > HOT_ZONE_THRESHOLD  ### Apply the threshold to get outliers

    if hot_zone_outliers.sum() > 0:  ### Check if there are any hot zone outliers
        hot_zone_blob = get_blob_indices(hot_zone_outliers)  ### Get the blob indices
        if len(hot_zone_blob) > 20:  ### Check if the blob is significant
            return np.array(get_hot_zone_blob_bbox(hot_zone_blob))  ### Return the bounding box of the hot zone blob

    return np.array([0, 0, 0, 0])  ### Return an empty bounding box if no significant hot zone



def get_bounding_box_hot_zone(y_indices, x_indices):
    """
    Calculate the bounding box for the hot zone given the indices.

    Parameters:
    - y_indices (np.ndarray): The y-coordinates of the hot zone pixels.
    - x_indices (np.ndarray): The x-coordinates of the hot zone pixels.

    Returns:
    - bbox (np.ndarray): The bounding box of the hot zone.
    """

    if any(y_indices) and any(x_indices):  ### Check if there are any hot zone pixels
        x_min, x_max = x_indices.min(), x_indices.max()  ### Calculate the x boundaries of the bounding box
        y_min, y_max = y_indices.min(), y_indices.max()  ### Calculate the y boundaries of the bounding box

        ### Return the bounding box expanded by 5 pixels in all directions ###
        return np.array([x_min - 5, y_min - 5, x_max + 5, y_max + 5], dtype=np.int16)

    return np.array([0, 0, 0, 0])  ### Return an empty bounding box if no hot zone pixels


def write_batch(batch_frames, writer_properties, current_frame_index, hot_zone_bbox=None, mark_type='traj'): # TODO: come up with better descriptive function name
    """
    Write a batch of frames to a video file and optionally save hot zone data.

    Parameters:
    - batch_frames (list or np.ndarray): The batch of frames to write.
    - writer_properties (tuple): Properties for the video writer.
    - current_frame_index (int): The current frame index.
    - hot_zone_bbox (np.ndarray): The bounding box of the hot zone (optional).
    - mark_type (str): The type of marking for the video (default is 'traj').

    Returns:
    - None
    """

    ### Convert batch to a numpy array ###
    batch_array = np.array(batch_frames)  ### Convert the batch of frames to a numpy array

    ### Construct the video file name ###
    video_name = f"{writer_properties[1]}_{mark_type}_{str(current_frame_index - 100)}-{current_frame_index}.mp4" # TODO: why the -100??

    ### Save hot zone data if available ###
    if hot_zone_bbox is not None and np.any \
            (hot_zone_bbox):  ### Check if hot zone bounding box is provided and not empty
        hot_zone_path = os.path.join(writer_properties[0], f"{video_name.split('.')[0]}_hot_zones.npy")
        np.save(hot_zone_path, hot_zone_bbox)  ### Save the hot zone bounding box

    ### Save numpy array if mark type is not 'regular' ###
    if "regular" not in mark_type:  ### Check if the mark type is not 'regular'  #TODO: what do you mean regular? what's going on here?
        numpy_path = os.path.join(writer_properties[0], f"{video_name.split('.')[0]}.npy")
        np.save(numpy_path, batch_array)  ### Save the batch array as a numpy file
        video_writer = cv2.VideoWriter(os.path.join(writer_properties[0], video_name),
                                       writer_properties[2],
                                       writer_properties[3],
                                       writer_properties[4], 0)  ### Initialize video writer with grayscale flag
    else:
        video_writer = cv2.VideoWriter(os.path.join(writer_properties[0], video_name),
                                       writer_properties[2],
                                       writer_properties[3],
                                       writer_properties[4])  ### Initialize video writer without grayscale flag

    ### Write each frame to the video file ###
    for frame in batch_array:
        video_writer.write(frame.astype(np.uint8))  ### Write the frame after converting to uint8

    video_writer.release()  ### Release the video writer



def points_distance_from_line(points, line_point1, line_point2):
    """
    Calculate the perpendicular distance of each point from a line defined by two points.

    Parameters:
    - points (np.ndarray): An array of points (x, y) to measure the distance from the line.
    - line_point1 (tuple): The first point (x1, y1) defining the line.
    - line_point2 (tuple): The second point (x2, y2) defining the line.

    Returns:
    - distances (np.ndarray): The perpendicular distances of each point from the line.
    """

    x1, y1 = line_point1  ### Extract coordinates of the first point defining the line
    x2, y2 = line_point2  ### Extract coordinates of the second point defining the line
    x0, y0 = points[:, 0], points[:, 1]  ### Extract x and y coordinates of the points

    ### Calculate the numerator of the distance formula
    numerator = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)

    ### Calculate the denominator of the distance formula
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    ### Calculate the perpendicular distances
    distances = numerator / denominator

    return distances  ### Return the array of distances


def clean_log(directory_path, file_list):
    """
    Remove log files in a directory if their count exceeds 50.

    Parameters:
    - directory_path (str): The root directory where log files are located.
    - file_list (list): A list of log files to manage.

    Returns:
    - None
    """

    while len(file_list) > 50:  ### Continue cleaning if there are more than 50 files
        try:
            file_list = os.listdir(directory_path)  ### List all files in the directory
            os.remove(os.path.join(directory_path, file_list[0]))  ### Remove the oldest file
        except Exception as e:  ### Catch exceptions during file removal
            print(f"Error removing file: {e}")
            pass  ### Ignore errors and continue



def increment_path(file_path, allow_existing=False, separator='', create_dir=False):
    """
    Increment file path to avoid overwriting existing files.

    Parameters:
    - file_path (str): The original file path.
    - allow_existing (bool): If False, increment the file path to avoid overwriting.
    - separator (str): Separator to use between the file name and increment number.
    - create_dir (bool): If True, create directories in the file path.

    Returns:
    - Path: The incremented file path.
    """

    path = Path(file_path)  ### Convert string path to Path object

    if path.exists() and not allow_existing:  ### Check if path exists and overwriting is not allowed
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (
            path, '')  ### Separate base path and suffix

        ### Iterate to find a unique path
        for n in range(2, 9999):
            new_path = f'{path}{separator}{n}{suffix}'  ### Construct new path with increment
            if not os.path.exists(new_path):  ### Check if the new path does not exist
                break
        path = Path(new_path)  ### Update path to new unique path

    if create_dir:  ### Create directories if required
        path.mkdir(parents=True, exist_ok=True)  ### Create directories in the path

    return path  ### Return the incremented or original path


def remove_non_english_letters(input_str):
    """
    Remove all non-English letters from the input string.

    Parameters:
    - input_str (str): The original string.

    Returns:
    - str: The cleaned string containing only English letters.
    """

    pattern = re.compile(r'[^a-zA-Z]')  ### Define pattern to match non-English letters
    cleaned_string = re.sub(pattern, '', input_str)  ### Replace non-English letters with an empty string

    return cleaned_string  ### Return the cleaned string



from scipy.spatial.distance import cdist
# from sklearn.cluster import KMeans
from RapidBase.import_all import *
from RapidBase.Anvil.alignments import align_to_reference_frame_circular_cc

def calculate_ransac_heuristics(model_accuracy, valid_trajectory_points, check_algo=False):
    """
    Calculate heuristics for RANSAC-based trajectory validation.

    Parameters:
    - model_accuracy (float): Accuracy of the model.
    - valid_trajectory_points (np.ndarray): Array of valid trajectory points.
    - check_algo (bool): Flag to indicate if algorithm check is needed.

    Returns:
    - Various heuristics and validation results.
    """

    ### Calculate Velocity ###
    delta_x = np.diff(valid_trajectory_points[:, 0])  ### Calculate x differences between consecutive points
    delta_y = np.diff(valid_trajectory_points[:, 1])  ### Calculate y differences between consecutive points
    delta_t = np.diff(valid_trajectory_points[:, 2])  ### Calculate time differences between consecutive points
    velocity = (np.sqrt(delta_x**2 + delta_y**2) / delta_t).reshape(1, -1)  ### Calculate velocity for each segment

    valid_points_based_on_velocity = np.append(np.logical_and(1 <= velocity, velocity < LINE_MIN_DIST * 2.5), True)  ### Filter points based on velocity
    valid_trajectory_points = valid_trajectory_points[valid_points_based_on_velocity]  ### Keep only valid points

    ### Calculate Distance Between Points ###
    distance_between_points = cdist(valid_trajectory_points[:, :2], valid_trajectory_points[:, :2])  ### Calculate pairwise distances

    ### Filter Out Distant Points ###
    is_point_out_there = np.logical_not \
        (np.sum(np.logical_not(distance_between_points > 100), axis=0) == 1)  ### Identify outlier points
    valid_trajectory_points = valid_trajectory_points[is_point_out_there]  ### Keep only non-outlier points

    if len(valid_trajectory_points) < 3:
        return False, None, None, None, None  ### Return if not enough valid points

    ### Calculate Line Start and End ###
    line_start = valid_trajectory_points[valid_trajectory_points[:, -1].argmin(), :2]  ### Find the starting point of the line
    line_end = valid_trajectory_points[valid_trajectory_points[:, -1].argmax(), :2]  ### Find the ending point of the line
    total_dist = euclidean_distance(line_start, line_end)  ### Calculate the total distance between start and end

    ### Calculate Average Differences ###
    x_diff_average = delta_x.mean()  ### Calculate average x difference
    y_diff_average = delta_y.mean()  ### Calculate average y difference

    ### Calculate Directions ###
    line_direction = line_end - line_start  ### Calculate line direction
    x_direction, y_direction = line_direction[0], line_direction[1]  ### Extract x and y direction components

    delta_x = np.diff(valid_trajectory_points[: ,0])
    delta_y = np.diff(valid_trajectory_points[:, 1])
    x_direction_positive = x_direction > 0  ### Check if x direction is positive
    y_direction_positive = y_direction > 0  ### Check if y direction is positive

    ### Filter Points Based on Direction ###
    x_doesnt_impact = [x_diff_average < abs(3)] * len(delta_x)  ### Check if x direction impacts the trajectory
    y_doesnt_impact = [y_diff_average < abs(3)] * len(delta_y)  ### Check if y direction impacts the trajectory

    x_points_in_correct_direction = np.logical_or((delta_x > 0) == x_direction_positive, x_doesnt_impact)  ### Filter points based on x direction
    y_points_in_correct_direction = np.logical_or((delta_y > 0) == y_direction_positive, y_doesnt_impact)  ### Filter points based on y direction
    valid_trajectory_points_based_on_direction = np.logical_and(x_points_in_correct_direction, y_points_in_correct_direction)  ### Combine direction filters
    valid_trajectory_points = valid_trajectory_points[np.append(valid_trajectory_points_based_on_direction, True)]  ### Keep only valid points based on direction

    if len(valid_trajectory_points) < 3:
        return False, None, None, None, None  ### Return if not enough valid points

    ### Calculate Average Velocity ###
    average_velocity = total_dist / (valid_trajectory_points[:, -1].max() - valid_trajectory_points[:, -1].min())  ### Calculate average velocity

    ### Cluster Valid Points ###
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto", max_iter=300).fit \
        (valid_trajectory_points[:, :2])  ### Cluster valid points
    kmeans_centers = kmeans.cluster_centers_  ### Get cluster centers

    ### Calculate Average Distance from Cluster Centers ###
    distance_matrix_list = [cdist(kmeans_centers[i].reshape(-1, 2), valid_trajectory_points[kmeans.labels_ == i, :2]) for i in range(2)]  ### Calculate distance matrices
    average_dist_from_centers = [distance_matrix_list[i].sum() / (kmeans.labels_ == i).sum() for i in range(2)]  ### Calculate average distances

    ### Calculate Velocity per Cluster ###
    velocity_per_cluster = [
        cdist(valid_trajectory_points[kmeans.labels_ == i, :2], valid_trajectory_points[kmeans.labels_ == i, :2]).max() /
        abs(valid_trajectory_points[kmeans.labels_ == i, 2][-1] - valid_trajectory_points[kmeans.labels_ == i, 2]
            [0] + 0.0001)
        for i in range(2)
    ]  ### Calculate velocity for each cluster

    ### Validate Clusters Based on Velocity ###
    valid_clusters = np.array(velocity_per_cluster) > 1  ### Check if clusters have valid velocities
    valid_points_in_valid_clusters = [kmeans.labels_ == i for i in range(2) if valid_clusters[i]]  ### Get valid points in valid clusters

    valid_points_in_valid_clusters.append(np.array(0))  ### Append 0 to ensure list is not empty

    if valid_clusters.sum() == 2:
        valid_points_in_valid_clusters = np.expand_dims \
            (np.logical_or(np.array(valid_points_in_valid_clusters[0]), np.array(valid_points_in_valid_clusters[1])), axis=0)  ### Combine valid points from both clusters

    ### Check Trajectory Validity ###
    line_long_enough = total_dist > LINE_MIN_DIST  ### Check if line is long enough
    cluster_centers_large_enough = max \
        (average_dist_from_centers) > LINE_MIN_DIST / 3  ### Check if cluster centers are far enough apart
    enough_valid_points_in_valid_clusters = valid_points_in_valid_clusters[0].sum() > MIN_RANSAC_SAMPLES - 1  ### Check if there are enough valid points in clusters
    enough_valid_velocity_points = valid_points_based_on_velocity.sum() > MIN_RANSAC_SAMPLES - 1  ### Check if there are enough valid velocity points
    cluster_velocity_very_large = max(velocity_per_cluster) > \
                (LINE_MIN_DIST / 4.5)  ### Check if cluster velocity is very large
    trajectory_characteristics_makes_to_much_sense = (
                valid_clusters.sum() == 2 and valid_points_based_on_velocity.sum() > 10 and model_accuracy > 0.85 and average_velocity > 1.5)  ### Check if trajectory characteristics make sense

    all_must_pass_params = (
                line_long_enough and cluster_centers_large_enough and enough_valid_points_in_valid_clusters and enough_valid_velocity_points)  ### Combine all must-pass parameters
    must_pass_one = (
                cluster_velocity_very_large or trajectory_characteristics_makes_to_much_sense)  ### Combine must-pass one conditions

    if all_must_pass_params and must_pass_one:
        return True, line_start, line_end, kmeans, valid_trajectory_points  ### Return if trajectory is valid

    elif check_algo and all_must_pass_params and must_pass_one:
        return True, total_dist, max(average_dist_from_centers), valid_points_in_valid_clusters[0].sum(), valid_points_based_on_velocity.sum(), \
            max(velocity_per_cluster), valid_clusters.sum(), average_velocity  ### Return detailed results for algorithm check

    return False, None, None, None, None  ### Return if trajectory is invalid


import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


def find_smooth_rise_fall_segment(input_array, K, threshold, sigma=2):
    """
    Find a segment within an input array that is approximately K samples long,
    exhibits rising and falling with a difference above a certain threshold, and does so smoothly.

    Parameters:
    - input_array (np.ndarray): The input array.
    - K (int): The approximate length of the segment to find.
    - threshold (float): The threshold value for the difference between initial and peak values.
    - sigma (float): Standard deviation for Gaussian kernel used in smoothing.

    Returns:
    - segment_indices (tuple): The start and end indices of the detected segment.
    - None: If no such segment is found.
    """

    # Step 1: Smooth the input array to eliminate abrupt outliers
    smoothed_array = gaussian_filter1d(input_array, sigma=sigma)

    # Step 2: Slide over the array to find segments of length K
    for start_idx in range(len(smoothed_array) - K + 1):
        segment = smoothed_array[start_idx:start_idx + K]

        # Step 3: Check if the segment has a rise and fall pattern
        peak_idx = np.argmax(segment)
        if peak_idx == 0 or peak_idx == K - 1:
            continue  # Skip if the peak is at the edges

        # Step 4: Check if the difference between initial and peak values is above the threshold
        initial_value = segment[0]
        peak_value = segment[peak_idx]
        if peak_value - initial_value < threshold:
            continue

        # Step 5: Check if there is a fall after the peak
        rising_part = np.all(np.diff(segment[:peak_idx]) > 0)
        falling_part = np.all(np.diff(segment[peak_idx:]) < 0)
        if rising_part and falling_part:
            return start_idx, start_idx + K

    return None  # No suitable segment found


def find_smooth_rise_fall_segments_3d(input_array, K, threshold, sigma=2):
    """
    Find segments within a 3D input array along the T dimension that are approximately K samples long,
    exhibit rising and falling with a difference above a certain threshold, and do so smoothly.

    Parameters:
    - input_array (np.ndarray): The 3D input array of shape (T, H, W).
    - K (int): The approximate length of the segment to find.
    - threshold (float): The threshold value for the difference between initial and peak values.
    - sigma (float): Standard deviation for Gaussian kernel used in smoothing.

    Returns:
    - results (np.ndarray): A 2D array of shape (H, W) where each element indicates whether a suitable segment was found.
    """

    # Convert the input array to a PyTorch tensor
    input_tensor = torch.tensor(input_array, dtype=torch.float32)

    # Apply Gaussian smoothing along the T dimension
    smoothed_tensor = torch.from_numpy(gaussian_filter1d(input_tensor.numpy(), sigma=sigma, axis=0))

    # Initialize results tensor
    T, H, W = smoothed_tensor.shape
    results = torch.zeros((H, W), dtype=torch.bool)

    # Slide a window of length K along the T dimension
    for start_idx in range(T - K + 1):
        segment = smoothed_tensor[start_idx:start_idx + K, :, :]

        # Find the peak index for each pixel location
        peak_indices = torch.argmax(segment, dim=0)

        # Create a mask for valid peaks (not at the edges)
        valid_peaks = (peak_indices > 0) & (peak_indices < K - 1)

        # Extract initial and peak values
        initial_values = segment[0, :, :]
        peak_values = torch.gather(segment, 0, peak_indices.unsqueeze(0)).squeeze(0)

        # Check if the difference between initial and peak values is above the threshold
        threshold_mask = (peak_values - initial_values) >= threshold

        # Create masks for rising and falling parts
        rising_parts = torch.all(segment[:peak_indices, :, :] < segment[1:peak_indices + 1, :, :], dim=0)
        falling_parts = torch.all(segment[peak_indices:, :, :] > segment[peak_indices + 1:, :, :], dim=0)

        # Combine all masks to identify valid segments
        valid_segments = valid_peaks & threshold_mask & rising_parts & falling_parts
        results |= valid_segments

    return results.numpy()






