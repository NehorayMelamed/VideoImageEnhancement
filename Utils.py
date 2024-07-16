


### Imports: ###
from scipy.interpolate import griddata
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize, warp
from skimage.measure import ransac
from skimage.transform import AffineTransform, ProjectiveTransform
from skimage.feature import match_descriptors, ORB
from skimage.color import rgb2gray

from skimage.measure import find_contours  # for finding contours in the segmentation mask
from skimage.draw import polygon  # for creating the segmentation mask with 1s inside the polygon

### Rapid Base Imports: ###
import matplotlib
from RapidBase.import_all import *
from RapidBase.Anvil._transforms.shift_matrix_subpixel import _shift_matrix_subpixel_fft_batch_with_channels
import torchvision.transforms as transforms
from torchvision.models.optical_flow import raft_large

matplotlib.use('TkAgg')
# from VideoImageEnhancement.Tracking.co_tracker.co_tracker import *
from scipy.optimize import minimize
import matplotlib.path as mpath

### Wiener Filtering: ###
import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import data, restoration
from skimage.restoration import unsupervised_wiener
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from skimage.restoration import estimate_sigma  # For estimating noise standard deviation
import torch
from typing import Tuple
from torch import Tensor
from skimage.util import img_as_float
from PARAMETER import *
start = -1
end = -1
from checkpoints.FlowFormer.core.FlowFormer.LatentCostFormer.transformer import FlowFormer
from checkpoints.FlowFormer.configs.sintel import get_cfg
from checkpoints.irr.models.IRR_PWC import PWCNet

def polygon_to_bounding_box_and_mask(polygon_points, input_shape):
    """
    This function takes a list of polygon points and returns:
    1. The minimum containing bounding box in the format (X0, Y0, X1, Y1).
    2. A binary segmentation mask of the input shape with 1s inside the polygon.

    Former function name: N/A

    Parameters:
    polygon_points (list): A list of tuples representing the polygon points.
                           Each tuple contains two integers (x, y).
    input_shape (tuple): A tuple representing the shape of the input (height, width).

    Returns:
    bounding_box (tuple): A tuple of four integers (X0, Y0, X1, Y1) representing the bounding box coordinates.
                          X0, Y0 are the coordinates of the top-left corner.
                          X1, Y1 are the coordinates of the bottom-right corner.
    segmentation_mask (np.ndarray): A binary mask of the same shape as the input, with the polygon area filled with ones.
                                    Shape is (height, width).
    """

    ### Calculate Bounding Box: ###
    x_coords, y_coords = zip(*polygon_points)  # unzip the polygon points into x and y coordinates
    X0, Y0 = min(x_coords), min(y_coords)  # calculate the minimum x and y coordinates
    X1, Y1 = max(x_coords), max(y_coords)  # calculate the maximum x and y coordinates
    bounding_box = (X0, Y0, X1, Y1)  # create the bounding box tuple

    ### Initialize Segmentation Mask: ###
    segmentation_mask = np.zeros(input_shape,
                                 dtype=np.uint8)  # create an empty mask with the same shape as the input

    ### Insert 1s in Polygon Area: ###
    rr, cc = polygon(y_coords, x_coords)  # get the row and column indices of the polygon
    segmentation_mask[rr, cc] = 1  # fill the polygon area with 1s

    return bounding_box, segmentation_mask  # return the bounding box and segmentation mask


def bounding_box_to_polygon_and_mask(bounding_box, input_shape):
    """
    This function takes a bounding box in the format (X0, Y0, X1, Y1) and returns:
    1. A list of polygon points from the bounding box edges.
    2. A binary segmentation mask of the input shape with 1s inside the bounding box.

    Former function name: N/A

    Parameters:
    bounding_box (tuple): A tuple of four integers (X0, Y0, X1, Y1) representing the bounding box coordinates.
                          X0, Y0 are the coordinates of the top-left corner.
                          X1, Y1 are the coordinates of the bottom-right corner.
    input_shape (tuple): A tuple representing the shape of the input (height, width).

    Returns:
    polygon_points (list): A list of tuples representing the four corners of the bounding box.
                           Each tuple contains two integers (x, y).
    segmentation_mask (np.ndarray): A binary mask of the same shape as the input, with the bounding box area filled with ones.
                                    Shape is (height, width).
    """

    ### Extract Bounding Box Coordinates: ###
    X0, Y0, X1, Y1 = bounding_box  # unpack bounding box coordinates

    ### Create List of Polygon Points: ###
    polygon_points = [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)]  # create list of the four corners of the bounding box

    ### Initialize Segmentation Mask: ###
    segmentation_mask = np.zeros(input_shape,
                                 dtype=np.uint8)  # create an empty mask with the same shape as the input

    ### Insert 1s in Bounding Box Area: ###
    segmentation_mask[Y0:Y1, X0:X1] = 1  # fill the bounding box area with 1s using slicing

    return polygon_points, segmentation_mask  # return the polygon points and segmentation mask


def mask_to_bounding_box_and_polygon(segmentation_mask):
    """
    This function takes a segmentation mask and returns:
    1. The minimum containing bounding box in the format (X0, Y0, X1, Y1).
    2. A list of polygon points representing the contour of the mask area.

    Former function name: N/A

    Parameters:
    segmentation_mask (np.ndarray): A binary mask with the shape (height, width), where 1s represent the mask area.

    Returns:
    bounding_box (tuple): A tuple of four integers (X0, Y0, X1, Y1) representing the bounding box coordinates.
                          X0, Y0 are the coordinates of the top-left corner.
                          X1, Y1 are the coordinates of the bottom-right corner.
    polygon_points (list): A list of tuples representing the contour points of the mask area.
                           Each tuple contains two integers (x, y).
    """

    ### Find Contours in the Segmentation Mask: ###
    contours = find_contours(segmentation_mask, level=0.5)  # find contours at the mask boundaries

    ### Check if Contours Were Found: ###
    if len(contours) == 0:
        raise ValueError("No contours found in the segmentation mask.")

    ### Extract the Largest Contour: ###
    largest_contour = max(contours, key=len)  # select the largest contour by length

    ### Convert Contour to Integer Coordinates: ###
    polygon_points = [(int(x), int(y)) for y, x in largest_contour]  # convert contour to integer coordinates

    ### Calculate Bounding Box: ###
    x_coords, y_coords = zip(*polygon_points)  # unzip the polygon points into x and y coordinates
    X0, Y0 = min(x_coords), min(y_coords)  # calculate the minimum x and y coordinates
    X1, Y1 = max(x_coords), max(y_coords)  # calculate the maximum x and y coordinates
    bounding_box = (X0, Y0, X1, Y1)  # create the bounding box tuple

    return bounding_box, polygon_points  # return the bounding box and polygon points


def points_to_bounding_box_and_mask(points, mask_shape):
    """
    This function takes a grid of points and returns:
    1. The minimum containing bounding box in the format (X0, Y0, X1, Y1).
    2. A binary segmentation mask with 1s inside the bounding box.

    Parameters:
    points (np.ndarray): A 2D array of shape [M, 2] where M is the number of points and each point is (x, y).
    mask_shape (tuple): A tuple representing the shape of the output mask (height, width).

    Returns:
    bounding_box (tuple): A tuple of four integers (X0, Y0, X1, Y1) representing the bounding box coordinates.
                          X0, Y0 are the coordinates of the top-left corner.
                          X1, Y1 are the coordinates of the bottom-right corner.
    segmentation_mask (np.ndarray): A binary mask of shape (height, width) with 1s inside the bounding box.
    """

    ### Extract X and Y Coordinates: ###
    x_coords, y_coords = points[:, 0], points[:, 1]  # extract x and y coordinates from the points

    ### Calculate Bounding Box: ###
    X0, Y0 = int(np.min(x_coords)), int(np.min(y_coords))  # calculate the minimum x and y coordinates
    X1, Y1 = int(np.max(x_coords)), int(np.max(y_coords))  # calculate the maximum x and y coordinates
    bounding_box = (X0, Y0, X1, Y1)  # create the bounding box tuple

    ### Create Segmentation Mask: ###
    segmentation_mask = np.zeros(mask_shape, dtype=np.uint8)  # initialize a mask with zeros
    segmentation_mask[Y0:Y1 + 1, X0:X1 + 1] = 1  # set the area inside the bounding box to 1s

    return bounding_box, segmentation_mask  # return the bounding box and segmentation mask


def generate_points_in_BB(bbox, grid_size=5):
    """
    Generate a grid of points within the bounding box.

    Args:
        bbox (tuple or np.ndarray): The bounding box coordinates (x0, y0, x1, y1).
        grid_size (int): The number of points along each dimension.

    Returns:
        list or np.ndarray: The grid of points with shape [grid_size*grid_size, 2]. If bbox is an array of shape [N, 4],
                            returns a list of grids for each bounding box.
    """

    def generate_grid_points(x0, y0, x1, y1, grid_size):
        """
        Helper function to generate grid points within a single bounding box.
        """
        x = np.linspace(x0, x1, grid_size)  # Generate x-coordinates
        y = np.linspace(y0, y1, grid_size)  # Generate y-coordinates
        points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)  # Generate grid of points
        return points  # Return grid of points

    ### Handling Different BBox Formats: ###
    if isinstance(bbox, tuple) or isinstance(bbox, list):  # If bbox is a tuple
        x0, y0, x1, y1 = bbox  # Extract bounding box coordinates
        points = generate_grid_points(x0, y0, x1, y1, grid_size)  # Generate grid points
        return points  # Return grid of points

    elif isinstance(bbox, np.ndarray) and bbox.shape == (4,):  # If bbox is a 1D array of size 4
        x0, y0, x1, y1 = bbox  # Extract bounding box coordinates
        points = generate_grid_points(x0, y0, x1, y1, grid_size)  # Generate grid points
        return points  # Return grid of points

    elif isinstance(bbox, np.ndarray) and bbox.shape[1] == 4:  # If bbox is a 2D array of shape [N, 4]
        points_list = []
        ### Looping Over Bounding Boxes: ###
        for single_bbox in bbox:  # Loop through each bounding box
            x0, y0, x1, y1 = single_bbox  # Extract bounding box coordinates
            points = generate_grid_points(x0, y0, x1, y1, grid_size)  # Generate grid points
            points_list.append(points)  # Append points to list
        return points_list  # Return list of grids for each bounding box

    else:
        raise ValueError("Invalid bbox format. Must be a tuple or an array of shape [N, 4] or [4,].")

def generate_points_in_polygon(polygon, grid_size=5):
    """
    Generate a grid of points within a closed polygon.

    Args:
        polygon (list or np.ndarray): The polygon vertices as a list of tuples or an array of shape [N, 2].
        grid_size (int): The number of points along the longer dimension of the bounding box.

    Returns:
        np.ndarray: The grid of points within the polygon with shape [M, 2], where M is the number of points inside the polygon.
    """
    ### This Is The Code Block: ###
    polygon = np.array(polygon)  # Convert polygon to numpy array if it is not already
    x_min, y_min = np.min(polygon, axis=0)  # Get the minimum x and y coordinates
    x_max, y_max = np.max(polygon, axis=0)  # Get the maximum x and y coordinates

    ### Create a grid of points within the bounding box of the polygon ###
    x_points = np.linspace(x_min, x_max, grid_size)  # Generate x-coordinates
    y_points = np.linspace(y_min, y_max, grid_size)  # Generate y-coordinates
    x_grid, y_grid = np.meshgrid(x_points, y_points)  # Create meshgrid of x and y points
    grid_points = np.vstack((x_grid.flatten(), y_grid.flatten())).T  # Flatten the grid to a list of points

    ### Check which points are inside the polygon ###
    path = mpath.Path(polygon)  # Create a path object from the polygon
    inside_mask = path.contains_points(grid_points)  # Check which grid points are inside the polygon
    points_in_polygon = grid_points[inside_mask]  # Select only the points inside the polygon

    return points_in_polygon  # Return the grid of points within the polygon

def generate_points_in_segmentation_mask(segmentation_mask, grid_size=5):
    """
    This function takes a segmentation mask with one area of 1s and returns:
    1. The minimum containing bounding box in the format (X0, Y0, X1, Y1).
    2. A list of polygon points representing the contour of the mask area.
    3. A grid of points within the polygon area inside the mask.

    Former function name: N/A

    Parameters:
    segmentation_mask (np.ndarray): A binary mask with the shape (height, width), where 1s represent the mask area.
    grid_size (int): The number of points along the longer dimension of the bounding box.

    Returns:
    bounding_box (tuple): A tuple of four integers (X0, Y0, X1, Y1) representing the bounding box coordinates.
                          X0, Y0 are the coordinates of the top-left corner.
                          X1, Y1 are the coordinates of the bottom-right corner.
    polygon_points (list): A list of tuples representing the contour points of the mask area.
                           Each tuple contains two integers (x, y).
    points_in_polygon (np.ndarray): A grid of points within the polygon area inside the mask.
                                    Shape is [M, 2], where M is the number of points inside the polygon.
    """

    ### Find Contours in the Segmentation Mask: ###
    contours = find_contours(segmentation_mask, level=0.5)  # find contours at the mask boundaries

    ### Check if Contours Were Found: ###
    if len(contours) == 0:
        ### Return Default Output: ###
        height, width = segmentation_mask.shape  # extract height and width of the input mask

        ### Define Bounding Box as Entire Frame: ###
        bounding_box = (0, 0, width, height)  # set bounding box to cover the entire frame

        ### Create Segmentation Mask of All 1s: ###
        segmentation_mask = np.ones((height, width), dtype=np.uint8)  # create a mask filled with 1s

        ### Define Polygon as Entire Image Vertices: ###
        polygon_points = [(0, 0), (width - 1, 0), (width - 1, height - 1),
                          (0, height - 1)]  # vertices of the entire image

        ### Generate Points within the Bounding Box: ###
        points_in_polygon = generate_points_in_BB(bounding_box, grid_size)  # generate points within the bounding box

        ### Return the Default Values: ###
        return bounding_box, polygon_points, points_in_polygon  # return the default output

    ### Extract the Largest Contour: ###
    largest_contour = max(contours, key=len)  # select the largest contour by length

    ### Convert Contour to Integer Coordinates: ###
    polygon_points = [(int(x), int(y)) for y, x in largest_contour]  # convert contour to integer coordinates

    ### Calculate Bounding Box: ###
    x_coords, y_coords = zip(*polygon_points)  # unzip the polygon points into x and y coordinates
    X0, Y0 = min(x_coords), min(y_coords)  # calculate the minimum x and y coordinates
    X1, Y1 = max(x_coords), max(y_coords)  # calculate the maximum x and y coordinates
    bounding_box = (X0, Y0, X1, Y1)  # create the bounding box tuple

    ### Generate Points within the Polygon: ###
    points_in_polygon = generate_points_in_polygon(polygon_points, grid_size)  # generate points within the polygon

    return bounding_box, polygon_points, points_in_polygon  # return the bounding box, polygon points, and points in polygon


def user_input_to_all_input_types(user_input, input_method='BB', input_shape=None):
    H, W = input_shape  # unpack input shape
    if user_input is not None:
        if input_method == 'BB':
            BB_XYXY = user_input
            polygon_points, segmentation_mask = bounding_box_to_polygon_and_mask(BB_XYXY, (H, W))
            grid_points = generate_points_in_BB(BB_XYXY, grid_size=5)
        elif input_method == 'polygon':
            polygon_points = user_input
            BB_XYXY, segmentation_mask = polygon_to_bounding_box_and_mask(polygon_points, (H, W))
            grid_points = generate_points_in_polygon(polygon_points, grid_size=5)
        elif input_method == 'segmentation':
            segmentation_mask = user_input
            BB_XYXY, polygon_points = mask_to_bounding_box_and_polygon(segmentation_mask)
            grid_points = generate_points_in_segmentation_mask(segmentation_mask, grid_size=5)
        flag_no_input = False
    else:
        BB_XYXY = [0, 0, W, H]  # default bounding box is full frame
        segmentation_mask = np.ones((H, W), dtype=np.uint8)
        polygon_points = [(0, 0), (W, 0), (W, H), (0, H)]  # default polygon is full frame
        grid_points = generate_points_in_BB(BB_XYXY, grid_size=5)  # default grid points are evenly spaced in the bounding box
        flag_no_input = True
    return BB_XYXY, polygon_points, segmentation_mask, grid_points, flag_no_input

def torch_get_4D(input_tensor, input_dims=None, flag_stack_BT=False):
    #[T,C,H,W]
    #(1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'C':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'T':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #(2).
    if len(input_tensor.shape) == 2:
        if input_dims is None:
            input_dims = 'HW'

        if input_dims == 'HW':
            return input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dims == 'CW':
            return input_tensor.unsqueeze(0).unsqueeze(2)
        elif input_dims == 'TW':
            return input_tensor.unsqueeze(1).unsqueeze(1)
        elif input_dims == 'CH':
            return input_tensor.unsqueeze(0).unsqueeze(-1)
        elif input_dims == 'TH':
            return input_tensor.unsqueeze(1).unsqueeze(-1)
        elif input_dims == 'TC':
            return input_tensor.unsqueeze(-1).unsqueeze(-1)

    #(3).
    if len(input_tensor.shape) == 3:
        if input_dims is None:
            input_dims = 'CHW'

        if input_dims == 'CHW':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'TCH':
            return input_tensor.unsqueeze(-1)
        elif input_dims == 'THW':
            return input_tensor.unsqueeze(1)

    #(4).
    if len(input_tensor.shape) == 4:
        return input_tensor

    #(5).
    if len(input_tensor.shape) == 5:
        if flag_stack_BT:
            B,T,C,H,W = input_tensor.shape
            return input_tensor.view(B*T,C,H,W)
        else:
            return input_tensor[0]


# def torch_get_3D(input_tensor):
#     if len(input_tensor.shape) == 1:
#         return input_tensor.unsqueeze(0).unsqueeze(0)
#     elif len(input_tensor.shape) == 2:
#         return input_tensor.unsqueeze(0)
#     elif len(input_tensor.shape) == 3:
#         return input_tensor
#     elif len(input_tensor.shape) == 4:
#         return input_tensor[0]
#     elif len(input_tensor.shape) == 5:
#         return input_tensor[0,0]

def torch_get_3D(input_tensor, input_dims=None):
    #(1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'C':
            return input_tensor.unsqueeze(-1).unsqueeze(-1)

    #(2).
    if len(input_tensor.shape) == 2:
        if input_dims is None:
            input_dims = 'HW'

        if input_dims == 'HW':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'CW':
            return input_tensor.unsqueeze(1)
        elif input_dims == 'CH':
            return input_tensor.unsqueeze(-1)

    #(3).
    if len(input_tensor.shape) == 3:
        return input_tensor

    #(4).
    if len(input_tensor.shape) == 4:
        return input_tensor[0]

    #(5).
    if len(input_tensor.shape) == 5:
        return input_tensor[0,0]  #TODO: maybe stack on top of each other?


# def torch_get_2D(input_tensor):
#     if len(input_tensor.shape) == 1:
#         return input_tensor.unsqueeze(0)
#     elif len(input_tensor.shape) == 2:
#         return input_tensor
#     elif len(input_tensor.shape) == 3:
#         return input_tensor[0]
#     elif len(input_tensor.shape) == 4:
#         return input_tensor[0,0]
#     elif len(input_tensor.shape) == 5:
#         return input_tensor[0,0,0]


def torch_get_2D(input_tensor, input_dims=None):
    # (1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1)

    # (2).
    if len(input_tensor.shape) == 2:
        return input_tensor

    # (3).
    if len(input_tensor.shape) == 3:
        return input_tensor[0]

    # (4).
    if len(input_tensor.shape) == 4:
        return input_tensor[0,0]

    # (5).
    if len(input_tensor.shape) == 5:
        return input_tensor[0, 0, 0]  # TODO: maybe stack on top of each other?


def torch_at_least_2D(input_tensor):
    return torch_get_2D(input_tensor)
def torch_at_least_3D(input_tensor):
    return torch_get_3D(input_tensor)
def torch_at_least_4D(input_tensor):
    return torch_get_4D(input_tensor)
def torch_at_least_5D(input_tensor):
    return torch_get_5D(input_tensor)


def torch_get_5D(input_tensor, input_dims=None):
    # (1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'C':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'T':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'B':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # (2).
    if len(input_tensor.shape) == 2:
        if input_dims is None:
            input_dims = 'HW'

        if input_dims == 'HW':
            return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'CW':
            return input_tensor.unsqueeze(1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'TW':
            return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        elif input_dims == 'BW':
            return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        elif input_dims == 'CH':
            return input_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'TH':
            return input_tensor.unsqueeze(1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'BH':
            return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        elif input_dims == 'TC':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'BC':
            return input_tensor.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        elif input_dims == 'BT':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # (3).
    if len(input_tensor.shape) == 3:
        if input_dims is None:
            input_dims = 'CHW'

        if input_dims == 'CHW':
            return input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dims == 'THW':
            return input_tensor.unsqueeze(1).unsqueeze(0)
        elif input_dims == 'BHW':
            return input_tensor.unsqueeze(1).unsqueeze(1)
        elif input_dims == 'TCH':
            return input_tensor.unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'BCH':
            return input_tensor.unsqueeze(1).unsqueeze(-1)
        elif input_dims == 'BTC':
            return input_tensor.unsqueeze(-1).unsqueeze(-1)

    # (4).
    if len(input_tensor.shape) == 4:
        if input_dims is None:
            input_dims = 'TCHW'

        if input_dims == 'TCHW':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'BTCH':
            return input_tensor.unsqueeze(-1)
        elif input_dims == 'BCHW':
            return input_tensor.unsqueeze(1)
        elif input_dims == 'BTHW':
            return input_tensor.unsqueeze(2)

    # (5).
    if len(input_tensor.shape) == 5:
        return input_tensor


def crop_torch_batch(images, crop_size_tuple_or_scalar, crop_style='center', start_H=-1, start_W=-1, flag_pad_if_needed=True):
    ### Initial: ###
    if len(images.shape) == 2:
        H,W = images.shape
        C = 1
    elif len(images.shape) == 3:
        C,H,W = images.shape
    elif len(images.shape) == 4:
        T, C, H, W = images.shape  # No Batch Dimension
    else:
        B, T, C, H, W = images.shape  # "Full" with Batch Dimension

    ### Assign cropW and cropH: ###
    if type(crop_size_tuple_or_scalar) == list or type(crop_size_tuple_or_scalar) == tuple:
        cropH = crop_size_tuple_or_scalar[0]
        cropW = crop_size_tuple_or_scalar[1]
    else:
        cropH = crop_size_tuple_or_scalar
        cropW = crop_size_tuple_or_scalar

    ### Decide On Crop Size According To Input: ###
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropW = cropW
    else:
        cropW = min(cropW, W)
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropH = cropH
    else:
        cropH = min(cropH, W)

    ### Get Start-Stop Indices To Crop: ###
    if crop_style == 'random':
        if cropW < W:
            start_W = np.random.randint(0, W - cropW)
        else:
            start_W = 0
        if cropH < H:
            start_H = np.random.randint(0, H - cropH)
        else:
            start_H = 0
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    elif crop_style == 'predetermined':
        start_H = start_H
        start_W = start_W
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    else:  #center
        mat_in_rows_excess = H - cropH
        mat_in_cols_excess = W - cropW
        start = 0
        end_x = W
        end_y = H
        start_W = int(start + mat_in_cols_excess / 2)
        start_H = int(start + mat_in_rows_excess / 2)
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)

    ### Crop Images (consistently between images/frames): ###
    if len(images.shape) == 2:
        return pad_torch_batch(images[start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    elif len(images.shape) == 3:
        return pad_torch_batch(images[:, start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    elif len(images.shape) == 4:
        return pad_torch_batch(images[:, :, start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    else:
        return pad_torch_batch(images[:, :, :, start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)


def crop_numpy_batch(images, crop_size_tuple_or_scalar, crop_style='center', start_H=-1, start_W=-1, flag_pad_if_needed=True):
    ### Initial: ###
    if len(images.shape) == 2:
        H, W = images.shape
    elif len(images.shape) == 3:
        H, W, C = images.shape #BW batch
    else:
        T, H, W, C = images.shape #RGB/NOT-BW batch

    ### Assign cropW and cropH: ###
    if type(crop_size_tuple_or_scalar) == list or type(crop_size_tuple_or_scalar) == tuple:
        cropW = crop_size_tuple_or_scalar[1]
        cropH = crop_size_tuple_or_scalar[0]
    else:
        cropW = crop_size_tuple_or_scalar
        cropH = crop_size_tuple_or_scalar
    cropW = min(cropW, W)
    cropH = min(cropH, H)

    ### Decide On Crop Size According To Input: ###
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropW = cropW
    else:
        cropW = min(cropW, W)
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropH = cropH
    else:
        cropH = min(cropH, W)

    ### Get Start-Stop Indices To Crop: ###
    if crop_style == 'random':
        if cropW < W:
            start_W = np.random.randint(0, W - cropW)
        else:
            start_W = 0
        if cropH < H:
            start_H = np.random.randint(0, H - cropH)
        else:
            start_H = 0
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    elif crop_style == 'predetermined':
        start_H = start_H
        start_W = start_W
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    else:
        mat_in_rows_excess = H - cropH
        mat_in_cols_excess = W - cropW
        start = 0
        end_x = W
        end_y = H
        start_W = int(start + mat_in_cols_excess / 2)
        start_H = int(start + mat_in_rows_excess / 2)
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)

    ### Crop Images (consistently between images/frames): ###    imshow_torch(tensor1_crop)
    if len(images.shape) == 2:
        return pad_numpy_batch(images[start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    elif len(images.shape) == 3:
        return pad_numpy_batch(images[start_H:stop_H, start_W:stop_W, :], crop_size_tuple_or_scalar)
    else:
        return pad_numpy_batch(images[:, start_H:stop_H, start_W:stop_W, :], crop_size_tuple_or_scalar)


def crop_tensor(images, crop_size_tuple_or_scalar, crop_style='center', start_H=-1, start_W=-1, flag_pad_if_needed=True):
    ### crop_style = 'center', 'random', 'predetermined'
    if type(images) == torch.Tensor:
        return crop_torch_batch(images, crop_size_tuple_or_scalar, crop_style, start_H, start_W, flag_pad_if_needed)
    else:
        return crop_numpy_batch(images, crop_size_tuple_or_scalar, crop_style, start_H, start_W, flag_pad_if_needed)


def pad_numpy_batch(input_arr: np.array, pad_size: Tuple[int, int], pad_style='center'):
    # TODO sometime support other pad styles.
    # expected BHWC array
    if pad_style == 'center':
        def pad_shape_HWC(a, pad_size):  # create a rigid shape for padding in dims HWC
            pad_start = np.floor(np.subtract(pad_size, a.shape[-3: -1]) / 2).astype(int)
            pad_end = np.ceil(np.subtract(pad_size, a.shape[-3: -1]) / 2).astype(int)
            return (pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (0, 0)
        def pad_shape_HW(a, pad_size):  # create a rigid shape for padding in dims HWC
            pad_start = np.floor(np.subtract(pad_size, a.shape) / 2).astype(int)
            pad_end = np.ceil(np.subtract(pad_size, a.shape) / 2).astype(int)
            return (pad_start[0], pad_end[0]), (pad_start[1], pad_end[1])

        if len(input_arr.shape) == 4:
            return np.array([np.pad(a, pad_shape_HWC(a, pad_size), 'constant', constant_values=0) for a in input_arr])
        elif len(input_arr.shape) == 3:
            return np.pad(input_arr, pad_shape_HWC(input_arr, pad_size), 'constant', constant_values=0)
        elif len(input_arr.shape) == 2:
            return np.pad(input_arr, pad_shape_HW(input_arr, pad_size), 'constant', constant_values=0)

    else:
        return None


def pad_torch_batch(input_tensor: Tensor, pad_size: Tuple[int, int], pad_style='center'):
    # TODO sometime support other pad styles.
    # expected BHWC array
    if pad_style == 'center':
        def pad_shape(t, pad_size):  # create a rigid shape for padding in dims CHW
            pad_start = np.floor(np.subtract(pad_size, torch.tensor(t.shape[-2:]).cpu().numpy()) / 2)
            pad_end = np.ceil(np.subtract(pad_size, torch.tensor(t.shape[-2:]).cpu().numpy()) / 2)
            return int(pad_start[1]), int(pad_end[1]), int(pad_start[0]), int(pad_end[0])

        return torch.nn.functional.pad(input_tensor, pad_shape(input_tensor, pad_size), mode='constant')
    else:
        return None

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


def list_to_tensor(input_list, device='cpu'):
    input_list = list_to_torch(input_list, device)
    input_list = torch.cat([torch_get_4D(input_list[i]) for i in np.arange(len(input_list))])
    return input_list


def list_to_numpy(input_list):
    input_list = np.concatenate([numpy_unsqueeze(input_list[i], 0) for i in np.arange(len(input_list))])
    return input_list


def numpy_to_list(input_array):
    T, H, W, C = input_array.shape
    output_list = []
    for i in np.arange(T):
        output_list.append(input_array[i])
    return output_list

def imshow_np(image: np.ndarray, title: str = '', flag_new_figure=True, maximize_window=False):
    """
    Display an image using matplotlib.

    Args:
        image (np.ndarray): Image to display.
        title (str): Title of the plot.
        flag_new_figure (bool): Whether to create a new figure.
        maximize_window (bool): Whether to maximize the plot window.
    """

    if flag_new_figure:
        plt.figure()

    plt.imshow(image)
    plt.title(title)

    if maximize_window:
        fig_manager = plt.get_current_fig_manager()
        backend = plt.get_backend()

        if backend == 'TkAgg':
            fig_manager.resize(*fig_manager.window.maxsize())
        elif backend == 'Qt5Agg':
            fig_manager.window.showMaximized()
        else:
            print(f"Maximizing not supported for the {backend} backend.")

    plt.pause(0.5)

def draw_bounding_box(image):
    """
    Allow user to draw a bounding box on the given image.

    Args:
        image (np.ndarray): The image to draw the bounding box on.

    Returns:
        tuple: The bounding box coordinates (x0, y0, x1, y1).
    """

    ### Allow User to Draw Bounding Box: ###
    bbox = cv2.selectROI("Select Bounding Box", image, fromCenter=False, showCrosshair=True)  # Draw bounding box
    cv2.destroyAllWindows()  # Close the window
    return bbox  # Return bounding box coordinates


def select_points_and_build_polygon(image):
    """
    Allows the user to select points within an image and build a closed polygon from them.

    Inputs:
    - image: numpy array of shape [H, W, C], the image in which points will be selected.

    Outputs:
    - polygon: list of tuples, each tuple contains (x, y) coordinates of the points forming the closed polygon.
    """
    ### This Is The Code Block: ###
    fig, ax = plt.subplots()  # Create a figure and axis
    ax.imshow(image)  # Display the image
    ax.set_title('Press ESC to finish')  # Set the title of the figure

    selected_points = []  # Initialize a list to store the selected points
    line, = ax.plot([], [], 'r-')  # Initialize a line object for interactive drawing

        ### This Is The Code Block: ###
    def onclick(event):
        """
        Handles mouse click events to record the coordinates of the selected points.

        Inputs:
        - event: MouseEvent, the event triggered by a mouse click.

        Outputs:
        - None
        """
        if event.button == 1:  # If the left mouse button is clicked
            x, y = event.xdata, event.ydata  # Get the x and y coordinates of the click
            selected_points.append((x, y))  # Append the coordinates to the selected points list
            ax.plot(x, y, 'ro')  # Plot a red dot at the selected point
            update_line()  # Update the line object for interactive drawing
            fig.canvas.draw()  # Update the figure to show the new point

    ### This Is The Code Block: ###
    def on_key(event):
        """
        Handles key press events to close the polygon and finish point selection.

        Inputs:
        - event: KeyEvent, the event triggered by a key press.

        Outputs:
        - None
        """
        if event.key == 'escape':  # If the ESC key is pressed
            if len(selected_points) > 2:  # Ensure there are enough points to form a polygon
                selected_points.append(selected_points[0])  # Close the polygon by adding the first point at the end
                ax.plot([p[0] for p in selected_points], [p[1] for p in selected_points], 'r-')  # Draw the polygon
                fig.canvas.draw()  # Update the figure to show the polygon
                plt.close(fig)  # Close the figure window

    ### This Is The Code Block: ###
    def update_line():
        """
        Updates the line object to draw a line between the last and current points.

        Outputs:
        - None
        """
        if len(selected_points) > 1:
            line.set_data([p[0] for p in selected_points], [p[1] for p in selected_points])  # Update line data

    ### Connect the event handlers ###
    cid = fig.canvas.mpl_connect('button_press_event',
                                 onclick)  # Connect the mouse click event to the onclick function
    kid = fig.canvas.mpl_connect('key_press_event', on_key)  # Connect the key press event to the on_key function

    plt.show()  # Display the figure and start the event loop

    return selected_points  # Return the list of selected points forming the closed polygon



def select_points_and_build_polygon_opencv(image, scale_factor=4):
    """
    Allows the user to select points within an image and build a closed polygon from them using OpenCV.
    Displays a scaled-up version of the image for easier selection but saves the polygon coordinates in the original image coordinates.

    Inputs:
    - image: numpy array of shape [H, W, C], the image in which points will be selected.
    - scale_factor: int, the factor by which to scale up the image for display.

    Outputs:
    - polygon: list of tuples, each tuple contains (x, y) coordinates of the points forming the closed polygon.
    """
    ### This Is The Code Block: ###
    original_height, original_width = image.shape[:2]  # Get the original dimensions of the image
    scaled_image = cv2.resize(image, (original_width * scale_factor, original_height * scale_factor))  # Scale up the image

    selected_points = []  # Initialize a list to store the selected points

    ### This Is The Code Block: ###
    def click_event(event, x, y, flags, param):
        """
        Handles mouse click events to record the coordinates of the selected points and draw the polygon interactively.

        Inputs:
        - event: MouseEvent, the event triggered by a mouse click.
        - x: int, the x-coordinate of the mouse click.
        - y: int, the y-coordinate of the mouse click.
        - flags: int, any relevant flags passed by OpenCV.
        - param: any extra parameters supplied by OpenCV.

        Outputs:
        - None
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # If the left mouse button is clicked
            orig_x, orig_y = x // scale_factor, y // scale_factor  # Scale down the coordinates to the original size
            selected_points.append((orig_x, orig_y))  # Append the coordinates to the selected points list
            if len(selected_points) > 1:
                cv2.line(scaled_image, (selected_points[-2][0] * scale_factor, selected_points[-2][1] * scale_factor), (selected_points[-1][0] * scale_factor, selected_points[-1][1] * scale_factor), (0, 0, 255), 2)  # Draw a line between the last two points
            cv2.circle(scaled_image, (x, y), 5, (0, 0, 255), -1)  # Draw a red circle at the selected point
            cv2.imshow('Image', scaled_image)  # Update the displayed image

    ### This Is The Code Block: ###
    def finalize_polygon():
        """
        Finalizes the polygon by closing it and displaying the final result.

        Outputs:
        - None
        """
        if len(selected_points) > 2:
            cv2.line(scaled_image, (selected_points[-1][0] * scale_factor, selected_points[-1][1] * scale_factor), (selected_points[0][0] * scale_factor, selected_points[0][1] * scale_factor), (0, 0, 255), 2)  # Close the polygon
            cv2.imshow('Image', scaled_image)  # Update the displayed image

    ### Display the image and set the mouse callback ###
    cv2.imshow('Image', scaled_image)  # Display the scaled-up image
    cv2.setMouseCallback('Image', click_event)  # Set the mouse callback function to handle click events

    ### Wait for the user to press ESC to finish ###
    while True:
        key = cv2.waitKey(1) & 0xFF  # Wait for a key press
        if key == 27:  # If the ESC key is pressed
            finalize_polygon()  # Finalize the polygon
            break  # Exit the loop

    cv2.destroyAllWindows()  # Close all OpenCV windows

    return selected_points  # Return the list of selected points forming the closed polygon






def torch_get_4D(input_tensor, input_dims=None, flag_stack_BT=False):
    #[T,C,H,W]
    #(1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'C':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'T':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #(2).
    if len(input_tensor.shape) == 2:
        if input_dims is None:
            input_dims = 'HW'

        if input_dims == 'HW':
            return input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dims == 'CW':
            return input_tensor.unsqueeze(0).unsqueeze(2)
        elif input_dims == 'TW':
            return input_tensor.unsqueeze(1).unsqueeze(1)
        elif input_dims == 'CH':
            return input_tensor.unsqueeze(0).unsqueeze(-1)
        elif input_dims == 'TH':
            return input_tensor.unsqueeze(1).unsqueeze(-1)
        elif input_dims == 'TC':
            return input_tensor.unsqueeze(-1).unsqueeze(-1)

    #(3).
    if len(input_tensor.shape) == 3:
        if input_dims is None:
            input_dims = 'CHW'

        if input_dims == 'CHW':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'TCH':
            return input_tensor.unsqueeze(-1)
        elif input_dims == 'THW':
            return input_tensor.unsqueeze(1)

    #(4).
    if len(input_tensor.shape) == 4:
        return input_tensor

    #(5).
    if len(input_tensor.shape) == 5:
        if flag_stack_BT:
            B,T,C,H,W = input_tensor.shape
            return input_tensor.view(B*T,C,H,W)
        else:
            return input_tensor[0]


# def torch_get_3D(input_tensor):
#     if len(input_tensor.shape) == 1:
#         return input_tensor.unsqueeze(0).unsqueeze(0)
#     elif len(input_tensor.shape) == 2:
#         return input_tensor.unsqueeze(0)
#     elif len(input_tensor.shape) == 3:
#         return input_tensor
#     elif len(input_tensor.shape) == 4:
#         return input_tensor[0]
#     elif len(input_tensor.shape) == 5:
#         return input_tensor[0,0]


def load_state_dict_into_module(model, state_dict, strict=True, loading_prefix=None):
    """
    Load a state dictionary into a model with optional prefix handling.
    Args:
        model (torch.nn.Module): The model to load the state dictionary into.
        state_dict (dict): The state dictionary containing model parameters.
        strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by model's state_dict.
        loading_prefix (str, optional): Prefix to remove from the keys in the state_dict.
    Returns:
        own_state (dict): The state dictionary of the model after loading.
    """
    own_state = model.state_dict()  # Get the state dictionary of the model
    count1, count2, count3 = 0, 0, 0  # Counters for copied, not copied, and resized parameters

    ### Iterating Over State Dictionary Items: ###
    for name, param in state_dict.items():
        if loading_prefix is not None:
            name = name.partition(loading_prefix)[-1]  # Remove prefix if specified

        if name in own_state:  # Check if parameter exists in model's state dictionary
            if isinstance(param, torch.nn.Parameter):
                param = param.data  # Get data from parameter for compatibility
            try:
                own_state[name].resize_as_(param)  # Resize the model's parameter to match the state_dict parameter
                own_state[name].copy_(param)  # Copy parameter from state_dict to model
                count1 += 1  # Increment copied counter
            except Exception:
                if strict:
                    raise RuntimeError(f'Error copying parameter {name}. Model dimensions: {own_state[name].size()}, Checkpoint dimensions: {param.size()}')
                else:
                    print(f'Note: parameter {name} was not copied, you might have a wrong checkpoint')
                    count2 += 1  # Increment not copied counter
        elif strict:
            raise KeyError(f'Unexpected key "{name}" in state_dict')

    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())  # Find missing keys
        if len(missing) > 0:
            raise KeyError(f'Missing keys in state_dict: "{missing}"')

    print(f'Out of {len(state_dict.items())} params: {count1} params were strictly copied/resized, {count2} params were not copied')
    return own_state  # Return the updated state dictionary of the model


### Bicubic Interpolation Function: ###
def bicubic_interpolate(input_image, X, Y):
    """
    Perform bicubic interpolation on an input image.
    Args:
        input_image (torch.Tensor): Input image tensor of shape [B, C, H, W].
        X (torch.Tensor): X-coordinate delta map of shape [B, H, W, 1].
        Y (torch.Tensor): Y-coordinate delta map of shape [B, H, W, 1].
    Returns:
        output (torch.Tensor): Interpolated image tensor.
    """
    B, C, H, W = input_image.shape  # Get input image dimensions
    BXC = B * C  # Compute the total number of channels

    input_image = input_image.contiguous().view(-1, H, W)  # Reshape input image to [B*C, H, W]

    ### Handle Delta Maps: ###
    if X.shape[0] != input_image.shape[0]:
        X = X.repeat([C, 1, 1, 1])
        Y = Y.repeat([C, 1, 1, 1])
    delta_x_flat = X.contiguous().view(BXC, H * W, 1)
    delta_y_flat = Y.contiguous().view(BXC, H * W, 1)

    x_map = delta_x_flat.contiguous().view(-1).float()
    y_map = delta_y_flat.contiguous().view(-1).float()

    height_f = float(H)
    width_f = float(W)

    zero = 0
    max_y = int(H - 1)
    max_x = int(W - 1)

    x_map = (x_map + 1) * (width_f - 1) / 2.0
    y_map = (y_map + 1) * (height_f - 1) / 2.0

    x0 = x_map.floor().int()
    y0 = y_map.floor().int()

    xm1 = x0 - 1
    ym1 = y0 - 1

    x1 = x0 + 1
    y1 = y0 + 1

    x2 = x0 + 2
    y2 = y0 + 2

    tx = x_map - x0.float()
    ty = y_map - y0.float()

    c_xm1 = ((-tx ** 3 + 2 * tx ** 2 - tx) / 2.0)
    c_x0 = ((3 * tx ** 3 - 5 * tx ** 2 + 2) / 2.0)
    c_x1 = ((-3 * tx ** 3 + 4 * tx ** 2 + tx) / 2.0)
    c_x2 = (1.0 - (c_xm1 + c_x0 + c_x1))

    c_ym1 = ((-ty ** 3 + 2 * ty ** 2 - ty) / 2.0)
    c_y0 = ((3 * ty ** 3 - 5 * ty ** 2 + 2) / 2.0)
    c_y1 = ((-3 * ty ** 3 + 4 * ty ** 2 + ty) / 2.0)
    c_y2 = (1.0 - (c_ym1 + c_y0 + c_y1))

    xm1 = xm1.clamp(zero, max_x)
    x0 = x0.clamp(zero, max_x)
    x1 = x1.clamp(zero, max_x)
    x2 = x2.clamp(zero, max_x)

    ym1 = ym1.clamp(zero, max_y)
    y0 = y0.clamp(zero, max_y)
    y1 = y1.clamp(zero, max_y)
    y2 = y2.clamp(zero, max_y)

    dim2 = W
    dim1 = W * H

    base = torch.zeros(dim1 * BXC).int().to(input_image.device)
    for i in np.arange(BXC):
        base[(i + 1) * H * W: (i + 2) * H * W] = torch.Tensor([(i + 1) * H * W]).to(input_image.device).int()

    base_ym1 = base + ym1 * dim2
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    base_y2 = base + y2 * dim2

    idx_ym1_xm1 = base_ym1 + xm1
    idx_ym1_x0 = base_ym1 + x0
    idx_ym1_x1 = base_ym1 + x1
    idx_ym1_x2 = base_ym1 + x2

    idx_y0_xm1 = base_y0 + xm1
    idx_y0_x0 = base_y0 + x0
    idx_y0_x1 = base_y0 + x1
    idx_y0_x2 = base_y0 + x2

    idx_y1_xm1 = base_y1 + xm1
    idx_y1_x0 = base_y1 + x0
    idx_y1_x1 = base_y1 + x1
    idx_y1_x2 = base_y1 + x2

    idx_y2_xm1 = base_y2 + xm1
    idx_y2_x0 = base_y2 + x0
    idx_y2_x1 = base_y2 + x1
    idx_y2_x2 = base_y2 + x2

    input_image_flat = input_image.contiguous().view(-1).float()

    I_ym1_xm1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_ym1_xm1.long()))
    I_ym1_x0 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_ym1_x0.long()))
    I_ym1_x1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_ym1_x1.long()))
    I_ym1_x2 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_ym1_x2.long()))

    I_y0_xm1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y0_xm1.long()))
    I_y0_x0 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y0_x0.long()))
    I_y0_x1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y0_x1.long()))
    I_y0_x2 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y0_x2.long()))

    I_y1_xm1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y1_xm1.long()))
    I_y1_x0 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y1_x0.long()))
    I_y1_x1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y1_x1.long()))
    I_y1_x2 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y1_x2.long()))

    I_y2_xm1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y2_xm1.long()))
    I_y2_x0 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y2_x0.long()))
    I_y2_x1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y2_x1.long()))
    I_y2_x2 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y2_x2.long()))

    output_ym1 = c_xm1 * I_ym1_xm1 + c_x0 * I_ym1_x0 + c_x1 * I_ym1_x1 + c_x2 * I_ym1_x2
    output_y0 = c_xm1 * I_y0_xm1 + c_x0 * I_y0_x0 + c_x1 * I_y0_x1 + c_x2 * I_y0_x2
    output_y1 = c_xm1 * I_y1_xm1 + c_x0 * I_y1_x0 + c_x1 * I_y1_x1 + c_x2 * I_y1_x2
    output_y2 = c_xm1 * I_y2_xm1 + c_x0 * I_y2_x0 + c_x1 * I_y2_x1 + c_x2 * I_y2_x2

    output = c_ym1.view(BXC, H, W) * output_ym1.view(BXC, H, W) + \
             c_y0.view(BXC, H, W) * output_y0.view(BXC, H, W) + \
             c_y1.view(BXC, H, W) * output_y1.view(BXC, H, W) + \
             c_y2.view(BXC, H, W) * output_y2.view(BXC, H, W)

    output = output.contiguous().view(-1, C, H, W)  # Reshape output to original shape
    return output  # Return interpolated image


### Warp Object Class: ###
class Warp_Object(torch.nn.Module):
    """
    Warp object for applying spatial transformations to images.
    """
    def __init__(self):
        super(Warp_Object, self).__init__()
        self.X = None  # X-coordinate meshgrid
        self.Y = None  # Y-coordinate meshgrid

    def forward(self, input_image, delta_x, delta_y, flag_bicubic_or_bilinear='bilinear'):
        """
        Forward pass for warping an image.
        Args:
            input_image (torch.Tensor): Input image tensor of shape [B, C, H, W].
            delta_x (torch.Tensor): X-coordinate delta map.
            delta_y (torch.Tensor): Y-coordinate delta map.
            flag_bicubic_or_bilinear (str): Interpolation method ('bicubic' or 'bilinear').
        Returns:
            warped_image (torch.Tensor): Warped image tensor.
        """
        B, C, H, W = input_image.shape  # Get input image dimensions
        BXC = B * C  # Compute the total number of channels

        ### ReOrder delta_x, delta_y: ###
        if len(delta_x.shape) == 3:  # [B, H, W] -> [B, H, W, 1]
            delta_x = delta_x.unsqueeze(-1)
            delta_y = delta_y.unsqueeze(-1)
            flag_same_on_all_channels = True
        elif len(delta_x.shape) == 4 and delta_x.shape[1] == C:  # [B, C, H, W] -> [BXC, H, W, 1]
            delta_x = delta_x.view(B * C, H, W).unsqueeze(-1)
            delta_y = delta_y.view(B * C, H, W).unsqueeze(-1)
            flag_same_on_all_channels = False
        elif len(delta_x.shape) == 4 and delta_x.shape[1] == 1:  # [B, 1, H, W] -> [B, H, W, 1]
            delta_x = delta_x.permute([0, 2, 3, 1])
            delta_y = delta_y.permute([0, 2, 3, 1])
            flag_same_on_all_channels = True
        elif len(delta_x.shape) == 4 and delta_x.shape[3] == 1:  # [B, H, W, 1]
            flag_same_on_all_channels = True

        flag_same_on_all_channels = True

        ### Create Baseline Meshgrid: ###
        flag_input_changed_from_last_time = (self.X is None) or (self.X.shape[0] != BXC and not flag_same_on_all_channels) or (self.X.shape[0] != B and flag_same_on_all_channels) or (self.X.shape[1] != H) or (self.X.shape[2] != W)
        if flag_input_changed_from_last_time:
            print('new meshgrid')
            [X, Y] = np.meshgrid(np.arange(W), np.arange(H))  # Create meshgrid
            if flag_same_on_all_channels:
                X = torch.Tensor(np.array([X] * B)).unsqueeze(-1)  # [B, H, W, 1]
                Y = torch.Tensor(np.array([Y] * B)).unsqueeze(-1)
            else:
                X = torch.Tensor(np.array([X] * BXC)).unsqueeze(-1)  # [BXC, H, W, 1]
                Y = torch.Tensor(np.array([Y] * BXC)).unsqueeze(-1)
            X = X.to(input_image.device)
            Y = Y.to(input_image.device)
            self.X = X
            self.Y = Y

        ### Add Delta Maps to Meshgrid: ###
        new_X = 2 * ((self.X + delta_x)) / max(W - 1, 1) - 1
        new_Y = 2 * ((self.Y + delta_y)) / max(H - 1, 1) - 1

        if flag_bicubic_or_bilinear == 'bicubic':
            warped_image = bicubic_interpolate(input_image, new_X, new_Y)  # Perform bicubic interpolation
            return warped_image
        else:
            bilinear_grid = torch.cat([new_X, new_Y], dim=3)
            if flag_same_on_all_channels:
                warped_image = torch.nn.functional.grid_sample(input_image, bilinear_grid)  # Perform bilinear interpolation
                return warped_image
            else:
                input_image_to_bilinear = input_image.view(BXC, H, W).unsqueeze(1)  # [B*C, 1, H, W]
                warped_image = torch.nn.functional.grid_sample(input_image_to_bilinear, bilinear_grid)
                warped_image = warped_image.view(B, C, H, W)
                return warped_image



### Optical Flow and Occlusion on Video: ###
# def get_optical_flow_and_occlusion_on_video(input, flow_model=None, occ_model=None):
#     """
#     Get optical flow and occlusion maps for a video.
#     Args:
#         input (torch.Tensor): Input video tensor of shape [B, T, C, H, W].
#         flow_model (torch.nn.Module, optional): Optical flow model.
#         occ_model (torch.nn.Module, optional): Occlusion model.
#     Returns:
#         optical_flow (torch.Tensor): Optical flow tensor.
#         occlusion (torch.Tensor): Occlusion map tensor.
#     """
#     if flow_model is None:
#         flow_model = load_flow_former().eval()  # Load and evaluate flow model
#     if occ_model is None:
#         occ_model = load_pwc().eval()  # Load and evaluate occlusion model
#
#     flow_model_output = []
#     occlusion_model_output = []
#     num_frames = input.shape[1]
#
#     ### Looping Over Frames: ###
#     with torch.no_grad():
#         for i in range(num_frames - 1):
#             flowformer_optical_flow, _, _, _ = flow_model(torch.cat([input[:, i:i + 1], input[:, i + 1:i + 2]], 1))  # Get optical flow
#             flow_model_output.append(flowformer_optical_flow)
#
#             _, occlusion_model_occlusion, _, _ = occ_model.forward(torch.cat([input[:, i:i + 1], input[:, i + 1:i + 2]], 1))  # Get occlusion map
#             occlusion_model_output.append(occlusion_model_occlusion)
#
#     torch.cuda.empty_cache()
#
#     optical_flow = torch.cat([f.unsqueeze(0) for f in flow_model_output])  # Concatenate optical flow outputs
#     occlusion = torch.cat([f.unsqueeze(0) for f in occlusion_model_output])  # Concatenate occlusion outputs
#
#     optical_flow = torchvision.utils.flow_to_image(optical_flow.squeeze()).unsqueeze(1)
#     return optical_flow, occlusion


def closest_multiple(input_number, multiple, direction='down'):
    if direction == 'down':
        return math.floor(input_number / multiple) * multiple
    elif direction == 'up':
        return math.ceil(input_number / multiple) * multiple

def get_optical_flow_and_occlusion_on_video(input_tensor, reference_tensor, flow_model=None, occ_model=None):
    """
    Get optical flow and occlusion maps for a video.

    Args:
        input (torch.Tensor): Input video tensor of shape [B, T, C, H, W].
        flow_model (torch.nn.Module, optional): Optical flow model.
        occ_model (torch.nn.Module, optional): Occlusion model.

    Returns:
        optical_flow (torch.Tensor): Optical flow tensor.
        occlusion (torch.Tensor): Occlusion map tensor.
    """
    ### Load Default Models if None Provided: ###
    if flow_model is None:
        flow_model = load_flow_former().eval()  # Load and evaluate flow model
    if occ_model is None:
        occ_model = load_pwc().eval()  # Load and evaluate occlusion model

    ### Initialize Model Outputs: ###
    flow_model_output = []  # List to store flow model outputs
    occlusion_model_output = []  # List to store occlusion model outputs

    ### Get Number of Frames: ###
    num_frames = input_tensor.shape[1]  # Get number of frames

    ## Get crop size to multiple of 8: ###


    ### Looping Over Frames: ###
    with torch.no_grad():  # Disable gradient calculation
        for i in range(num_frames):  # Loop through frames
            print(i)
            ### Get Optical Flow: ###
            flowformer_optical_flow, _, _, _ = flow_model(
                torch.cat([input_tensor[:, i:i+1], reference_tensor.unsqueeze(0)], 1)
            )  # Get optical flow
            flow_model_output.append(flowformer_optical_flow)  # Append to flow model output list

            ### Get Occlusions: ###
            _, occlusion_model_occlusion, _, _ = occ_model.forward(
                torch.cat([input_tensor[:, i:i+1], reference_tensor.unsqueeze(0)], 1)
            )  # Get occlusion map
            occlusion_model_output.append(occlusion_model_occlusion)  # Append to occlusion model output list

            torch.cuda.empty_cache()  # Clear CUDA memory cache

    ### Free CUDA Memory: ###
    torch.cuda.empty_cache()  # Clear CUDA memory cache

    ### Concatenate Model Outputs: ###
    optical_flow = torch.cat([f.unsqueeze(0) for f in flow_model_output])  # Concatenate flow model outputs
    occlusion = torch.cat([f.unsqueeze(0) for f in occlusion_model_output])  # Concatenate occlusion model outputs

    ### Convert Optical Flow to Image: ###
    optical_flow_pretty_flow = torchvision.utils.flow_to_image(optical_flow.squeeze()).unsqueeze(1)  # Convert flow to image

    return optical_flow, occlusion  # Return optical flow and occlusion



# def denoise_using_optical_flow_and_occlusion_maps(model_output, input_frames):
#     """
#     Denoise model outputs using optical flow and occlusion maps.
#     Args:
#         model_output (list): List containing optical flow, occlusion maps, and other outputs.
#         input_frames (torch.Tensor): Input video frames tensor.
#     Returns:
#         final_estimate (torch.Tensor): Denoised image tensor.
#         warp_list (list): List of warped frames.
#     """
#     warp_object = Warp_Object()
#     flows, occlusion_maps, _, _ = model_output
#
#     N_Frames, B_flow_original, T_flow_original, H_flow_original, W_flow_original = flows.shape
#     middle_frame_index = (flows.shape[0] - 1) // 2
#     warp_list = []
#     warp_list_unscaled = []
#
#     ### Looping Over Frames: ###
#     for idx, flow in enumerate(flows):
#         if idx != middle_frame_index:
#             image_to_warp = input_frames[:, idx].float()
#             warped_frames_towrds_model = warp_object.forward(input_image=image_to_warp, delta_x=flow[0:1, 0:1], delta_y=flow[0:1, 1:2])
#             warp_list.append(warped_frames_towrds_model)
#
#     if occlusion_maps is not None:
#         if occlusion_maps.shape[1] == 1:
#             occlusion_maps = torch.cat([occlusion_maps[:middle_frame_index], occlusion_maps[middle_frame_index + 1:]])
#             Weights = 1 - occlusion_maps.squeeze(1)
#             warped_tensor = torch.cat(warp_list, 0)
#             final_weighted_estimate_nominator = (warped_tensor * Weights).sum(0)
#             final_weighted_estimate_denominator = Weights.sum(0) + 1e-3
#             final_weighted_estimate = final_weighted_estimate_nominator / final_weighted_estimate_denominator
#             final_unweighted_estimate = warped_tensor.mean(0)
#             final_estimate = final_weighted_estimate.unsqueeze(0)
#         elif occlusion_maps.shape[1] == 2:
#             occlusion_maps = torch.cat([occlusion_maps[:middle_frame_index], occlusion_maps[middle_frame_index + 1:]])
#             Weights = 1 - occlusion_maps.squeeze(1)
#             warped_tensor = torch.cat(warp_list, 0)
#             final_weighted_estimate_nominator = (warped_tensor * Weights).sum(0)
#             final_weighted_estimate_denominator = Weights.sum(0) + 1e-3
#             final_weighted_estimate = final_weighted_estimate_nominator / final_weighted_estimate_denominator
#             final_unweighted_estimate = warped_tensor.mean(0)
#             final_estimate = final_weighted_estimate.unsqueeze(0)
#     else:
#         warped_tensor = torch.cat([warp_list[i].unsqueeze(0) for i in range(len(warp_list))], 0)
#         final_estimate = warped_tensor.mean(0)
#         warped_tensor_unscaled = torch.cat([warp_list[i].unsqueeze(0) for i in range(len(warp_list_unscaled))], 0)
#         final_estimate_unscaled = warped_tensor_unscaled.mean(0)
#
#     return final_estimate, warp_list


def denoise_using_optical_flow_and_occlusion_maps(model_output, input_frames):
    """
    Denoise model outputs using optical flow and occlusion maps.

    Args:
        model_output (list): List containing optical flow, occlusion maps, and other outputs.
        input_frames (torch.Tensor): Input video frames tensor.

    Returns:
        final_estimate (torch.Tensor): Denoised image tensor.
        warp_list (list): List of warped frames.
    """
    ### Initialize Warp Object: ###
    warp_object = Warp_Object()  # Initialize warp object

    ### Extract Model Outputs: ###
    flows, occlusion_maps = model_output  # Unpack model output

    ### Get Shape Parameters: ###
    N_Frames, B_flow_original, T_flow_original, H_flow_original, W_flow_original = flows.shape  # Get shape of flows
    middle_frame_index = (flows.shape[0] - 1) // 2  # Get index of middle frame

    ### Initialize Warp Lists: ###
    warp_list = []  # List to store warped frames
    warp_list_unscaled = []  # List to store unscaled warped frames

    ### Looping Over Frames: ###
    for idx, flow in enumerate(flows):  # Loop through each flow
        image_to_warp = input_frames[:, idx].float()  # Get the current frame to warp
        warped_frames_towrds_model = warp_object.forward(
            input_image=image_to_warp,
            delta_x=-flow[0:1, 0:1],
            delta_y=-flow[0:1, 1:2]
        )  # Warp the current frame
        warp_list.append(warped_frames_towrds_model)  # Append the warped frame to the list

    ### Handle Occlusion Maps if Available: ###
    if occlusion_maps is not None:  # If occlusion maps are provided
        if occlusion_maps.shape[1] == 1:  # If occlusion maps have a single channel
            # occlusion_maps = torch.cat(
            #     [occlusion_maps[:middle_frame_index], occlusion_maps[middle_frame_index + 1:]]
            # )  # Concatenate occlusion maps excluding the middle frame
            Weights = 1 - occlusion_maps.squeeze(1)  # Compute weights from occlusion maps
            warped_tensor = torch.cat(warp_list, 0)  # Concatenate all warped frames
            final_weighted_estimate_nominator = (warped_tensor * Weights).sum(0)  # Compute numerator for final estimate
            final_weighted_estimate_denominator = Weights.sum(0) + 1e-3  # Compute denominator for final estimate
            final_weighted_estimate = final_weighted_estimate_nominator / final_weighted_estimate_denominator  # Compute weighted estimate
            final_unweighted_estimate = warped_tensor.mean(0)  # Compute unweighted estimate
            final_estimate = final_weighted_estimate.unsqueeze(0)  # Unsqueeze final estimate

        elif occlusion_maps.shape[1] == 2:  # If occlusion maps have two channels
            # occlusion_maps = torch.cat(
            #     [occlusion_maps[:middle_frame_index], occlusion_maps[middle_frame_index + 1:]]
            # )  # Concatenate occlusion maps excluding the middle frame
            Weights = 1 - occlusion_maps.squeeze(1)  # Compute weights from occlusion maps
            warped_tensor = torch.cat(warp_list, 0)  # Concatenate all warped frames
            final_weighted_estimate_nominator = (warped_tensor * Weights).sum(0)  # Compute numerator for final estimate
            final_weighted_estimate_denominator = Weights.sum(0) + 1e-3  # Compute denominator for final estimate
            final_weighted_estimate = final_weighted_estimate_nominator / final_weighted_estimate_denominator  # Compute weighted estimate
            final_unweighted_estimate = warped_tensor.mean(0)  # Compute unweighted estimate
            final_estimate = final_weighted_estimate.unsqueeze(0)  # Unsqueeze final estimate

    ### Handle Cases Without Occlusion Maps: ###
    else:  # If occlusion maps are not provided
        warped_tensor = torch.cat([warp_list[i].unsqueeze(0) for i in range(len(warp_list))], 0)  # Concatenate warped frames
        final_estimate = warped_tensor.mean(0)  # Compute mean of warped frames
        warped_tensor_unscaled = torch.cat([warp_list[i].unsqueeze(0) for i in range(len(warp_list_unscaled))], 0)  # Concatenate unscaled warped frames
        final_estimate_unscaled = warped_tensor_unscaled.mean(0)  # Compute mean of unscaled warped frames

    return final_estimate, warp_list  # Return final estimate and list of warped frames


# from skvideo import io



def load_pwc(checkpoint=path_checkpoint_latest_things, base_name_restore=BASE_NAME_RESTORE):
    """

    Args:
        checkpoint: model checkpoint
        train_devices: gpus
        base_name_restore: base name to make a valid path

    Returns:

    """
    args = EasyDict()
    # args.restore_ckpt = os.path.join(base_name_restore, checkpoint)
    args.restore_ckpt = checkpoint

    model = PWCNet(args).to(DEVICE)
    loading_prefix = '_model.'
    pretrained_dict = torch.load(path_checkpoint_latest_things, map_location=torch.device(DEVICE))['state_dict']
    state_dict = load_state_dict_into_module(model, pretrained_dict, strict=False, loading_prefix=loading_prefix)
    model.load_state_dict(state_dict)
    return model



def load_flow_former(checkpoint=path_restore_ckpt_denoise_flow_former, base_name_restore=BASE_NAME_RESTORE):
    """

    Args:
        checkpoint: model checkpoint
        train_devices: gpus
        base_name_restore: base name to make a valid path

    Returns:

    """
    args = EasyDict()
    # args.restore_ckpt = os.path.join(base_name_restore, checkpoint)
    args.restore_ckpt = checkpoint
    args.name = 'flowformer'
    args.stage = 'sintel'
    args.validation = 'true'
    args.mixed_precision = True
    cfg = get_cfg()
    cfg.update(vars(args))

    # model = torch.nn.DataParallel(FlowFormer(cfg.latentcostformer), device_ids=[DEVICE]).to(DEVICE)
    # model.load_state_dict(torch.load(args.restore_ckpt))
    # return model
    ##ToDo i just paste a const path, it should be like abouve
    model = torch.nn.DataParallel(FlowFormer(cfg.latentcostformer), device_ids=[DEVICE]).to(DEVICE)
    model.load_state_dict(torch.load(path_restore_ckpt_denoise_flow_former))
    return model


### Denoise Main Function: ###
# def align_and_average_frames_using_FlowFormer_and_PWC(input_frames, flow_model=None, occ_model=None):
#     """
#     Main function for denoising video frames.
#     Args:
#         input_frames (torch.Tensor): Input video frames tensor of shape [B, T, C, H, W].
#         flow_model (torch.nn.Module, optional): Optical flow model.
#         occ_model (torch.nn.Module, optional): Occlusion model.
#     Returns:
#         denoised_reference (torch.Tensor): Denoised reference frame tensor.
#         warp_list (list): List of warped frames.
#     """
#     if flow_model is None:
#         flow_model = load_flow_former().eval()  # Load and evaluate flow model
#     if occ_model is None:
#         occ_model = load_pwc().eval()  # Load and evaluate occlusion model
#
#     num_frames = input_frames.shape[1]
#     ref_frame = input_frames[:, ((num_frames - 1) // 2)].unsqueeze(0)  # Reference frame
#
#     outputs = get_optical_flow_and_occlusion_on_video(flow_model, occ_model, input_frames)  # Get optical flow and occlusions
#
#     denoised_reference, warp_list = denoise_using_optical_flow_and_occlusion_maps(outputs, input_frames)  # Denoise model outputs
#
#     return denoised_reference, warp_list


def torch_get_3D(input_tensor, input_dims=None):
    #(1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'C':
            return input_tensor.unsqueeze(-1).unsqueeze(-1)

    #(2).
    if len(input_tensor.shape) == 2:
        if input_dims is None:
            input_dims = 'HW'

        if input_dims == 'HW':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'CW':
            return input_tensor.unsqueeze(1)
        elif input_dims == 'CH':
            return input_tensor.unsqueeze(-1)

    #(3).
    if len(input_tensor.shape) == 3:
        return input_tensor

    #(4).
    if len(input_tensor.shape) == 4:
        return input_tensor[0]

    #(5).
    if len(input_tensor.shape) == 5:
        return input_tensor[0,0]  #TODO: maybe stack on top of each other?


# def torch_get_2D(input_tensor):
#     if len(input_tensor.shape) == 1:
#         return input_tensor.unsqueeze(0)
#     elif len(input_tensor.shape) == 2:
#         return input_tensor
#     elif len(input_tensor.shape) == 3:
#         return input_tensor[0]
#     elif len(input_tensor.shape) == 4:
#         return input_tensor[0,0]
#     elif len(input_tensor.shape) == 5:
#         return input_tensor[0,0,0]


def torch_get_2D(input_tensor, input_dims=None):
    # (1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1)

    # (2).
    if len(input_tensor.shape) == 2:
        return input_tensor

    # (3).
    if len(input_tensor.shape) == 3:
        return input_tensor[0]

    # (4).
    if len(input_tensor.shape) == 4:
        return input_tensor[0,0]

    # (5).
    if len(input_tensor.shape) == 5:
        return input_tensor[0, 0, 0]  # TODO: maybe stack on top of each other?


def torch_at_least_2D(input_tensor):
    return torch_get_2D(input_tensor)
def torch_at_least_3D(input_tensor):
    return torch_get_3D(input_tensor)
def torch_at_least_4D(input_tensor):
    return torch_get_4D(input_tensor)
def torch_at_least_5D(input_tensor):
    return torch_get_5D(input_tensor)


def torch_get_5D(input_tensor, input_dims=None):
    # (1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'C':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'T':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'B':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # (2).
    if len(input_tensor.shape) == 2:
        if input_dims is None:
            input_dims = 'HW'

        if input_dims == 'HW':
            return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'CW':
            return input_tensor.unsqueeze(1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'TW':
            return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        elif input_dims == 'BW':
            return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        elif input_dims == 'CH':
            return input_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'TH':
            return input_tensor.unsqueeze(1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'BH':
            return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        elif input_dims == 'TC':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'BC':
            return input_tensor.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        elif input_dims == 'BT':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # (3).
    if len(input_tensor.shape) == 3:
        if input_dims is None:
            input_dims = 'CHW'

        if input_dims == 'CHW':
            return input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dims == 'THW':
            return input_tensor.unsqueeze(1).unsqueeze(0)
        elif input_dims == 'BHW':
            return input_tensor.unsqueeze(1).unsqueeze(1)
        elif input_dims == 'TCH':
            return input_tensor.unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'BCH':
            return input_tensor.unsqueeze(1).unsqueeze(-1)
        elif input_dims == 'BTC':
            return input_tensor.unsqueeze(-1).unsqueeze(-1)

    # (4).
    if len(input_tensor.shape) == 4:
        if input_dims is None:
            input_dims = 'TCHW'

        if input_dims == 'TCHW':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'BTCH':
            return input_tensor.unsqueeze(-1)
        elif input_dims == 'BCHW':
            return input_tensor.unsqueeze(1)
        elif input_dims == 'BTHW':
            return input_tensor.unsqueeze(2)

    # (5).
    if len(input_tensor.shape) == 5:
        return input_tensor
