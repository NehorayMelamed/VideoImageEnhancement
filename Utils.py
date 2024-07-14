


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

start = -1
end = -1


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





