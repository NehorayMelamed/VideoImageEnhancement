### Standard Library Imports ###
import math  # Importing math for mathematical functions
import sys  # Importing sys for system-specific parameters and functions
import os  # Importing os for interacting with the operating system
import copy  # Importing copy for deep copying objects

### OpenCV and PIL Imports ###
import cv2  # Importing OpenCV for computer vision tasks
from PIL import Image  # Importing Image from PIL for image processing

### NumPy and Torch Imports ###
import numpy as np  # Importing NumPy for numerical operations
import torch  # Importing PyTorch for deep learning

### Matplotlib Imports ###
from matplotlib import pyplot as plt  # Importing pyplot from matplotlib for plotting

### Skimage Imports ###
from skimage.io import imread, imsave  # Importing imread and imsave from skimage.io for image I/O
from skimage.color import rgb2gray  # Importing rgb2gray from skimage.color for color space conversion
from skimage.util import img_as_ubyte  # Importing img_as_ubyte from skimage.util for converting images to 8-bit unsigned integers

### Torchvision Imports ###
import torchvision  # Importing torchvision for computer vision tasks
from torchvision.utils import make_grid  # Importing make_grid from torchvision.utils for creating a grid of images

### Parameter Import ###
import PARAMETER  # Importing custom PARAMETER module for accessing parameters

### Append Paths for Custom Modules ###
sys.path.append('../../../')  # Appending relative path for custom modules
sys.path.append('../..')  # Appending relative path for custom modules
sys.path.append('../')  # Appending relative path for custom modules
sys.path.append(f"{PARAMETER.CHECKPOINTS}/NonUniformBlurKernelEstimation/NonUniformBlurKernelEstimation/utils_NUBKE")  # Appending path for custom utility modules
sys.path.append(f"{PARAMETER.CHECKPOINTS}/NonUniformBlurKernelEstimation/NonUniformBlurKernelEstimation")  # Appending path for custom modules
sys.path.append(f"{PARAMETER.CHECKPOINTS}/dwdn/dwdn")  # Appending path for custom modules


sys.path.append(PARAMETER.CHECKPOINTS)
### Custom Module Imports ###
from NonUniformBlurKernelEstimation.NonUniformBlurKernelEstimation.models.TwoHeadsNetwork import TwoHeadsNetwork  # Importing TwoHeadsNetwork model from custom module
from NonUniformBlurKernelEstimation.NonUniformBlurKernelEstimation.utils_NUBKE.visualization import save_kernels_grid, get_kernels_grid  # Importing visualization utilities from custom module
from NonUniformBlurKernelEstimation.NonUniformBlurKernelEstimation.utils_NUBKE.restoration import RL_restore, combined_RL_restore

from video_editor.imshow_pyqt import *

### Pause to Avoid Conflicts with OpenCV ###
plt.pause(2)  # Pause to avoid conflicts with OpenCV
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array

### Model: ###
DEVICE = 0
# model_file = "RDND_proper/models/NonUniformBlurKernelEstimation/NonUniformBlurKernelEstimation/models/TwoHeads.pkl"
# model_file = "TwoHeads.pkl"
model_file = PARAMETER.Blur_kernel_NubKe_model

def tensor2im(image_tensor, imtype=np.uint8):
	image_numpy = image_tensor.cpu().float().numpy()
	image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) * 255.0
	return image_numpy.astype(imtype)



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


def save_image(image_numpy, image_path):
	image_pil = None
	if image_numpy.shape[2] == 1:
		image_numpy = np.reshape(image_numpy, (image_numpy.shape[0],image_numpy.shape[1]))
		image_pil = Image.fromarray(image_numpy, 'L')
	else:
		image_pil = Image.fromarray(image_numpy)
	image_pil.save(image_path)


def save_kernels_grid_(blurry_image, kernels, masks, image_name):
    '''
     Draw and save CONVOLUTION kernels in the blurry image.
     Notice that computed kernels are CORRELATION kernels, therefore are flipped.
    :param blurry_image: Tensor (channels,M,N)
    :param kernels: Tensor (K,kernel_size,kernel_size)
    :param masks: Tensor (K,M,N)
    :return:
    '''
    K = masks.size(0)
    M = masks.size(1)
    N = masks.size(2)
    kernel_size = kernels.size(1)

    blurry_image = blurry_image.cpu().numpy()
    kernels = kernels.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()

    grid_to_draw = blurry_image.copy()
    for i in range(kernel_size, M - kernel_size // 2, 2 * kernel_size):
        for j in range(kernel_size, N - kernel_size // 2, 2 * kernel_size):
            kernel_ij = np.zeros((kernel_size, kernel_size))
            for k in range(K):
                kernel_ij += masks[k, i, j] * kernels[k]
            kernel_ij_norm = (kernel_ij - kernel_ij.min()) / (kernel_ij.max() - kernel_ij.min())
            grid_to_draw[:, i - kernel_size // 2:i + kernel_size // 2 + 1,
            j - kernel_size // 2:j + kernel_size // 2 + 1] = kernel_ij_norm[None, ::-1, ::-1]

    #self.add_image(image_name, grid_to_draw, step)

    imsave(image_name, img_as_ubyte(grid_to_draw.transpose((1, 2, 0))))


def save_kernels_grid_green(blurry_image, kernels, masks, image_name):
    '''
     Draw and save CONVOLUTION kernels in the blurry image.
     Notice that computed kernels are CORRELATION kernels, therefore are flipped.
    :param blurry_image: Tensor (channels,M,N)
    :param kernels: Tensor (K,kernel_size,kernel_size)
    :param masks: Tensor (K,M,N)
    :return:
    '''
    K = masks.size(0)
    M = masks.size(1)
    N = masks.size(2)
    kernel_size = kernels.size(1)

    blurry_image = blurry_image.cpu().numpy()
    kernels = kernels.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    grid_to_draw = blurry_image.copy()
    for i in range(kernel_size, M - kernel_size // 2, kernel_size):
        for j in range(kernel_size, N - kernel_size // 2, kernel_size):
            kernel_ij = np.zeros((3, kernel_size, kernel_size))
            for k in range(K):
                kernel_ij[1, :, :] += masks[k, i, j] * kernels[k]
            kernel_ij_norm = (kernel_ij - kernel_ij.min()) / (kernel_ij.max() - kernel_ij.min())
            grid_to_draw[:, i - kernel_size // 2:i + kernel_size // 2 + 1,
            j - kernel_size // 2:j + kernel_size // 2 + 1] = \
                0.5 * grid_to_draw[:, i - kernel_size // 2:i + kernel_size // 2 + 1,
                      j - kernel_size // 2:j + kernel_size // 2 + 1] + 0.5 * kernel_ij_norm[:, ::-1, ::-1]

    imsave(image_name, img_as_ubyte(grid_to_draw.transpose((1, 2, 0))))



from scipy.stats import linregress


# def fit_straight_line_to_kernel(blur_kernel):
#     """
#     Fits a straight line to the input blur kernel estimation array and returns the new blur kernel straight line fit.
#
#     Parameters:
#     -----------
#     blur_kernel : np.ndarray
#         2D array representing the estimated blur kernel.
#
#     Returns:
#     --------
#     straight_line_fit : np.ndarray
#         2D array representing the blur kernel with the straight line fit.
#     angle: float
#         Angle of the fitted line.
#     """
#     # Convert to binary image
#     # display_media(cv2.resize((blur_kernel * 255).astype(np.uint8), (33 * 5, 33 * 5)))
#     _, binary = cv2.threshold(blur_kernel, 0.5, 1.0, cv2.THRESH_BINARY)
#
#     # Find contours
#     contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Get the largest contour
#     cnt = contours[0]
#
#     # Calculate the moments of the binary image
#     M = cv2.moments(cnt)
#     if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#     else:
#         cX, cY = 0, 0
#
#     # Perform PCA to find the main axis
#     data_pts = cnt[:, 0, :]
#     mean, eigenvectors = cv2.PCACompute(data_pts.astype(np.float32), mean=np.array([]))
#
#     # Get the angle of the principal component
#     angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
#
#     # Fit a line using the points along the main axis
#     vx, vy, x, y = cv2.fitLine(data_pts, cv2.DIST_L2, 0, 0.01, 0.01)
#
#     # Generate points along the fitted line
#     line_points = []
#     for i in range(-100, 100):
#         line_points.append((x + vx * i, y + vy * i))
#     line_points = np.array(line_points)
#
#     # Create an empty image to draw the line
#     straight_line_fit = np.zeros_like(blur_kernel)
#
#     # Draw the fitted line
#     for point in line_points:
#         if 0 <= int(point[1]) < straight_line_fit.shape[0] and 0 <= int(point[0]) < straight_line_fit.shape[1]:
#             straight_line_fit[int(point[1]), int(point[0])] = 1.0
#     # display_media(cv2.resize((blur_kernel * 255).astype(np.uint8), (33 * 5, 33 * 5)))
#     # display_media(cv2.resize((straight_line_fit * 255).astype(np.uint8), (33 * 5, 33 * 5)))
#
#     return straight_line_fit, np.degrees(angle)


def fit_straight_line_to_kernel(blur_kernel):
    """
    Fits a straight line to the input blur kernel estimation array and returns the new blur kernel straight line fit.

    Parameters:
    -----------
    blur_kernel : np.ndarray
        2D array representing the estimated blur kernel.

    Returns:
    --------
    straight_line_fit : np.ndarray
        2D array representing the blur kernel with the straight line fit.
    angle: float
        Angle of the fitted line.
    """
    # Convert to binary image
    _, binary = cv2.threshold(blur_kernel, 0.5, 1.0, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    cnt = contours[0]

    # Calculate the moments of the binary image
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Perform PCA to find the main axis
    data_pts = cnt[:, 0, :]
    mean, eigenvectors = cv2.PCACompute(data_pts.astype(np.float32), mean=np.array([]))

    # Get the angle of the principal component
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

    # Fit a line using the points along the main axis
    vx, vy, x, y = cv2.fitLine(data_pts, cv2.DIST_L2, 0, 0.01, 0.01)

    # Calculate the endpoints of the line segment within the blur kernel bounds
    try:
        left_y = int((-x * vy / (vx+1e-3)) + y)
    except Exception as e:
        left_y = blur_kernel.shape[0]-1
    try:
        right_y = int(((blur_kernel.shape[1] - x) * vy / vx) + y)
    except Exception as e:
        right_y = int(0)
    try:
        top_x = int((-y * vx / (vy+1e-3)) + x)
    except Exception as e:
        top_x = blur_kernel.shape[1]-1
    try:
        bottom_x = int(((blur_kernel.shape[0] - y) * vx / vy) + x)
    except Exception as e:
        bottom_x = int(0)
    # Clip the coordinates to be within the image bounds
    left_y = np.clip(left_y, 0, blur_kernel.shape[0] - 1)
    right_y = np.clip(right_y, 0, blur_kernel.shape[0] - 1)
    top_x = np.clip(top_x, 0, blur_kernel.shape[1] - 1)
    bottom_x = np.clip(bottom_x, 0, blur_kernel.shape[1] - 1)

    # Create an empty image to draw the line
    straight_line_fit = np.zeros_like(blur_kernel)

    # Determine the two endpoints of the line within the bounds
    if np.abs(vx) > np.abs(vy):
        pt1 = (0, left_y)
        pt2 = (blur_kernel.shape[1] - 1, right_y)
    else:
        pt1 = (top_x, 0)
        pt2 = (bottom_x, blur_kernel.shape[0] - 1)

    # Draw the fitted line
    cv2.line(straight_line_fit, pt1, pt2, 1, 1)

    ### multiply the straight line fit by the thresholded blur kernel to put it within bounds of the blur kernel: ###
    straight_line_fit *= blur_kernel
    # display_media(cv2.resize((blur_kernel * 255).astype(np.uint8), (33 * 5, 33 * 5)))
    # display_media(cv2.resize((straight_line_fit * 255).astype(np.uint8), (33 * 5, 33 * 5)))

    return straight_line_fit, np.degrees(angle)


def save_kernels_grid(blurry_image, kernels, masks, flag_threshold_blur_kernel=False, quantile_to_threshold=0.9):
    '''
     Draw and save CONVOLUTION kernels in the blurry image.
     Notice that computed kernels are CORRELATION kernels, therefore are flipped.
    :param blurry_image: Tensor (channels,M,N)
    :param kernels: Tensor (K,kernel_size,kernel_size)
    :param masks: Tensor (K,M,N)
    :return:
    '''
    K = masks.size(0)
    M = masks.size(1)
    N = masks.size(2)
    kernel_size = kernels.size(1)

    blurry_image = blurry_image.cpu().numpy()
    kernels = kernels.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()

    ### Initialize image with blur kernels on it: ###
    grid_to_draw = 0.4*1 + 0.6*rgb2gray(blurry_image.transpose(1,2,0)).copy()
    grid_to_draw = np.repeat(grid_to_draw[None,:,:], 3, axis=0)

    ### Initialize image with thresholded blur kernels on it: ###
    grid_to_draw_thresholded = 0.4 * 1 + 0.6 * rgb2gray(blurry_image.transpose(1, 2, 0)).copy()
    grid_to_draw_thresholded = np.repeat(grid_to_draw_thresholded[None, :, :], 3, axis=0)

    ### Initialize image with straight line fit blur kernels on it: ###
    grid_to_draw_straight_line = 0.4 * 1 + 0.6 * rgb2gray(blurry_image.transpose(1, 2, 0)).copy()
    grid_to_draw_straight_line = np.repeat(grid_to_draw_straight_line[None, :, :], 3, axis=0)


    ### Loop over the grid cells: ###
    for i in range(kernel_size, M - kernel_size // 2, kernel_size):
        for j in range(kernel_size, N - kernel_size // 2, kernel_size):

            ### Get the average blur kernel over a region: ###
            kernel_ij = np.zeros((3, kernel_size, kernel_size))
            for k in range(K):
                kernel_ij[None, :, :] += masks[k, i, j] * kernels[k]
                # display_media(cv2.resize((kernel_ij[0]*255).astype(np.uint8), (33 * 5, 33 * 5)))

            ### Threshold blur kernel if wanted: ###
            if flag_threshold_blur_kernel:
                threshold = np.quantile(kernel_ij, quantile_to_threshold)
                kernel_ij_thresholded = (kernel_ij > threshold).astype(float)
                # display_media(cv2.resize((kernel_ij[0]*255).astype(np.uint8), (33*5,33*5)))
                # display_media(cv2.resize((kernel_ij_thresholded[0]*255).astype(np.uint8), (33*5,33*5)))

            ### Perform straight line fit to the blur kernel: ###
            kernel_ij_straight_line_fit, degree = fit_straight_line_to_kernel(kernel_ij_thresholded[0])
            kernel_ij_straight_line_fit = np.concatenate([kernel_ij_straight_line_fit[None, :, :]]*3)

            ### Get the normalized intensity of the blur kernel in current grid cell: ###
            kernel_ij_norm = (kernel_ij - kernel_ij.min()) / (kernel_ij.max() - kernel_ij.min())

            ### Get slice indices: ###
            h_start = i - kernel_size // 2
            h_end = i + kernel_size // 2 + 1
            w_start = j - kernel_size // 2
            w_end = j + kernel_size // 2 + 1

            ### Plot blur kernel on top of image: ###
            grid_to_draw[0, h_start:h_end, w_start:w_end] = 0.5 * kernel_ij_norm[0, ::-1, ::-1] + \
                                                            (1 - kernel_ij_norm[0, ::-1, ::-1]) * grid_to_draw[0,
                                                                                                  h_start:h_end,
                                                                                                  w_start:w_end]
            grid_to_draw[1:, h_start:h_end, w_start:w_end] = (1 - kernel_ij_norm[1:, ::-1, ::-1]) * \
                                                             grid_to_draw[1:, h_start:h_end, w_start:w_end]

            ### Plot blur kernel thresholded on top of image: ###
            grid_to_draw_thresholded[0, h_start:h_end, w_start:w_end] = 0.5 * kernel_ij_thresholded[0, ::-1, ::-1] + \
                                                            (1 - kernel_ij_thresholded[0, ::-1, ::-1]) * grid_to_draw_thresholded[0,
                                                                                                  h_start:h_end,
                                                                                                  w_start:w_end]
            grid_to_draw_thresholded[1:, h_start:h_end, w_start:w_end] = (1 - kernel_ij_thresholded[1:, ::-1, ::-1]) * \
                                                             grid_to_draw_thresholded[1:, h_start:h_end, w_start:w_end]

            ### Plot blur kernel straight line fit on top of image: ###
            grid_to_draw_straight_line[0, h_start:h_end, w_start:w_end] = 0.5 * kernel_ij_straight_line_fit[0, ::-1, ::-1] + \
                                                                        (1 - kernel_ij_straight_line_fit[0, ::-1, ::-1]) * grid_to_draw_straight_line[0,
                                                                                      h_start:h_end,
                                                                                      w_start:w_end]
            grid_to_draw_straight_line[1:, h_start:h_end, w_start:w_end] = (1 - kernel_ij_straight_line_fit[1:, ::-1, ::-1]) * \
                                                                         grid_to_draw_straight_line[1:, h_start:h_end, w_start:w_end]
            # display_media(grid_to_draw[0])
            # display_media(grid_to_draw_thresholded[0])
            # display_media(grid_to_draw_straight_line[0])
            # display_media(cv2.resize((kernel_ij_straight_line_fit[0]*255).astype(np.uint8), (33*5,33*5)))

    # for i in range(kernel_size, M - kernel_size // 2, kernel_size):
    #     for j in range(kernel_size, N - kernel_size // 2, kernel_size):
    #         kernel_ij = np.zeros((3, kernel_size, kernel_size))
    #         for k in range(K):
    #             kernel_ij[None, :, :] += masks[k, i, j] * kernels[k]
    #
    #         kernel_ij_norm = (kernel_ij - kernel_ij.min()) / (kernel_ij.max() - kernel_ij.min())
    #         # h_slice = i - kernel_size // 2 : i + kernel_size // 2 + 1
    #         # w_slice = j - kernel_size // 2 : j + kernel_size // 2 + 1
    #         grid_to_draw[0, i - kernel_size // 2:i + kernel_size // 2 + 1,
    #         j - kernel_size // 2:j + kernel_size // 2 + 1] = 0.5 * kernel_ij_norm[0, ::-1, ::-1] + (1-kernel_ij_norm[0, ::-1, ::-1]) * grid_to_draw[0, i - kernel_size // 2:i + kernel_size // 2 + 1,
    #                   j - kernel_size // 2:j + kernel_size // 2 + 1]
    #
    #         grid_to_draw[1:, i - kernel_size // 2:i + kernel_size // 2 + 1,
    #         j - kernel_size // 2:j + kernel_size // 2 + 1] = (1- kernel_ij_norm[1:, ::-1, ::-1]) * grid_to_draw[1:, i - kernel_size // 2:i + kernel_size // 2 + 1,
    #                   j - kernel_size // 2:j + kernel_size // 2 + 1]


    grid_to_draw = np.clip(grid_to_draw, 0, 1)
    grid_to_draw_thresholded = np.clip(grid_to_draw_thresholded, 0, 1)
    grid_to_draw_straight_line = np.clip(grid_to_draw_straight_line, 0, 1)
    return grid_to_draw, grid_to_draw_thresholded, grid_to_draw_straight_line
    # imsave(image_name, img_as_ubyte(grid_to_draw.transpose((1, 2, 0))))



def get_kernels_grid(kernels, masks):
    '''
     Draw and save CONVOLUTION kernels in the blurry image.
     Notice that computed kernels are CORRELATION kernels, therefore are flipped.
    :param blurry_image: Tensor (channels,M,N)
    :param kernels: Tensor (K,kernel_size,kernel_size)
    :param masks: Tensor (K,M,N)
    :return:
    '''
    ### This function basically gives a grid of kernels at stride kernel_size for the entire image
    # with the kernel not a weighted sum of each region, but multiplies by the center of the segmentation mask at each center block position...seems fishy

    K = masks.size(0)
    M = masks.size(1)
    N = masks.size(2)
    kernel_size = kernels.size(1)

    corrected_kernels = []
    corrected_kernels_inner = []
    for i in np.arange(kernel_size, M - kernel_size // 2, kernel_size):
        corrected_kernels_inner = []
        for j in np.arange(kernel_size, N - kernel_size // 2, kernel_size):
            kernel_ij = masks[:, i:i+1, j:j+1] * kernels
            corrected_kernels_inner.append(kernel_ij.sum(0))
        corrected_kernels.append(corrected_kernels_inner)

    reshaped_kernels = torch.zeros((len(corrected_kernels), len(corrected_kernels[0]), kernel_size, kernel_size))
    for i in np.arange(reshaped_kernels.shape[0]):
        for j in np.arange(reshaped_kernels.shape[1]):
            reshaped_kernels[i, j] = corrected_kernels[i][j]

    return reshaped_kernels


def read_image_torch(path, flag_convert_to_rgb=1, flag_normalize_to_float=0):
    image = cv2.imread(path, cv2.IMREAD_COLOR);
    if flag_convert_to_rgb == 1:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB);
    if flag_normalize_to_float == 1:
        image = image.astype(np.float32) / 255; #normalize to range [0,1] instead of [0,255]
    if image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        image = np.expand_dims(image, axis=2);
    if image.shape[2] > 4:
        # In the weird case where the image has more then 3 channels only take the first 3 channels to make it conform with 3-channels image format:
        image = image[:,:,:3];

    image = np.transpose(image,[2,0,1])
    image = torch.Tensor(image);
    image = image.unsqueeze(0)
    return image


def numpy_to_torch(input_image, device='cpu', flag_unsqueeze=False):
    #Assumes "Natural" RGB form to there's no BGR->RGB conversion, can also accept BW image!:
    if input_image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        input_image = np.expand_dims(input_image, axis=2) #[H,W]->[H,W,1]
    if input_image.ndim == 3:
        input_image = np.transpose(input_image, (2, 0, 1))  # [H,W,C]->[C,H,W]
    elif input_image.ndim == 4:
        input_image = np.transpose(input_image, (0, 3, 1, 2)) #[T,H,W,C] -> [T,C,H,W]
    input_image = torch.from_numpy(input_image.astype(float)).float().to(device) # to float32

    if flag_unsqueeze:
        input_image = input_image.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
    return input_image


### Function to Create a Grid of Kernels ###
def create_kernel_grid(kernels):
    """
    Creates a grid image of kernels.

    Inputs:
    - kernels: torch.Tensor of shape [num_kernels, kernel_size, kernel_size], the input kernels.

    Outputs:
    - kernel_image: torch.Tensor of shape [grid_size*kernel_size, grid_size*kernel_size], the grid image of kernels.
    """
    ### This Is The Code Block: ###
    num_kernels = kernels.shape[0]  # Number of kernels
    kernel_size = kernels.shape[1]  # Size of each kernel
    grid_size = int(math.sqrt(num_kernels))  # Calculate grid size
    kernel_image = torch.zeros((grid_size * kernel_size, grid_size * kernel_size))  # Initialize the grid image

    ### Looping Over Indices: ###
    for kernel_id, kernel in enumerate(kernels):  ### Looping Over Indices: ###
        i = kernel_id // grid_size  # Calculate row index in the grid
        j = kernel_id % grid_size  # Calculate column index in the grid
        kernel_image[i * kernel_size:(i + 1) * kernel_size, j * kernel_size:(j + 1) * kernel_size] = kernel  # Place kernel in the grid

    return kernel_image  # Return the grid image

### Function to Create a Grid of Kernel Indices ###
def create_kernel_grid_indices(kernels, kernel_id):
    """
    Creates a grid image of kernel indices.

    Inputs:
    - kernels: torch.Tensor of shape [num_kernels, kernel_size, kernel_size], the input kernels.
    - kernel_id: int, the index of the kernel to highlight.

    Outputs:
    - kernel_image: torch.Tensor of shape [grid_size*kernel_size, grid_size*kernel_size], the grid image of kernel indices.
    """
    ### This Is The Code Block: ###
    num_kernels = kernels.shape[0]  # Number of kernels
    kernel_size = kernels.shape[1]  # Size of each kernel
    grid_size = int(math.sqrt(num_kernels))  # Calculate grid size
    kernel_image = torch.zeros((grid_size * kernel_size, grid_size * kernel_size))  # Initialize the grid image

    i = kernel_id // grid_size  # Calculate row index in the grid
    j = kernel_id % grid_size  # Calculate column index in the grid
    kernel_image[i * kernel_size:(i + 1) * kernel_size, j * kernel_size:(j + 1) * kernel_size] = 1  # Highlight the kernel

    return kernel_image  # Return the grid image

### Function to Get Average Kernel from Segmentation Masks and Kernels ###
def get_avg_kernel_from_segmentation_masks_and_kernels(basis_kernels, masks, seg_mask=None, flag_plot=False):
    """
    Computes the average kernel inside the segmentation mask.

    Inputs:
    - blurry_image: torch.Tensor of shape [C, H, W], the input blurry image.
    - basis_kernels: torch.Tensor of shape [K, kernel_size, kernel_size], the basis kernels.
    - masks: torch.Tensor of shape [K, H, W], the masks for each kernel.
    - seg_mask: torch.Tensor of shape [H, W], the segmentation mask (optional).

    Outputs:
    - avg_kernel: torch.Tensor of shape [kernel_size, kernel_size], the average kernel inside the segmentation mask.
    """
    ### This Is The Code Block: ###
    M, N = masks.shape[-2:]  # Height and width of the masks

    ### Initialize Segmentation Mask if Not Provided ###
    if seg_mask is None:
        seg_mask = torch.zeros((1, 1, M, N))  # Initialize segmentation mask with zeros

    if len(seg_mask.shape) == 3:  # Ensure seg_mask has 4 dimensions
        seg_mask = seg_mask.unsqueeze(0)
    if len(seg_mask.shape) != 4:
        raise ValueError("Segmentation mask has invalid shape")  # Raise error for invalid shape

    if flag_plot:
        imshow_torch_temp(seg_mask[0], name="seg_mask_before_scale")  # Display the segmentation mask before scaling

    ### Get Corrected Kernels ###
    corrected_kernels = get_kernels_grid(basis_kernels.squeeze(), masks.squeeze())  # Get corrected kernels
    corrected_kernels = corrected_kernels.flatten(0, 1)  # Flatten the kernels

    ### Create Kernel Grids Logical Mask (according to position of the above corrected kernels) ###
    kernel_grids = torch.cat([create_kernel_grid_indices(corrected_kernels.squeeze(), i).unsqueeze(0) for i in range(corrected_kernels.shape[0])])  # Create kernel grids

    ### Upsample Segmentation Mask ###
    seg_mask = torch.nn.functional.upsample(seg_mask, size=kernel_grids.shape[1:]).squeeze()  # Upsample the segmentation mask
    if flag_plot:
        imshow_torch_temp(seg_mask.unsqueeze(0), name="seg_mask_after_scale")  # Display the segmentation mask after scaling

    ### Compute Kernel Weights ###
    kernel_weights = kernel_grids * seg_mask.to(kernel_grids.device)  # Compute kernel weights
    kernel_weights = kernel_weights.sum(-1).sum(-1)  # Sum kernel weights

    ### Compute Average Kernel ###
    avg_kernel = (corrected_kernels.squeeze() * kernel_weights.unsqueeze(-1).unsqueeze(-1).to(corrected_kernels.device)).sum(0)  # Compute the average kernel
    if flag_plot:
        imshow_torch_temp(avg_kernel.unsqueeze(0), "avg_kernel")  # Display the average kernel

    return avg_kernel  # Return the average kernel


def get_avg_kernel_from_segmentation_masks_and_kernels2(basis_kernels, masks, seg_mask=None, flag_plot=False, threshold_quantile=0.95):
    """
    Computes the average kernel inside the segmentation mask.

    Inputs:
    - blurry_image: torch.Tensor of shape [C, H, W], the input blurry image.
    - basis_kernels: torch.Tensor of shape [K, kernel_size, kernel_size], the basis kernels.
    - masks: torch.Tensor of shape [K, H, W], the masks for each kernel.
    - seg_mask: torch.Tensor of shape [H, W], the segmentation mask (optional).

    Outputs:
    - avg_kernel: torch.Tensor of shape [kernel_size, kernel_size], the average kernel inside the segmentation mask.
    """
    ### This Is The Code Block: ###
    M, N = masks.shape[-2:]  # Height and width of the masks

    ### Initialize Segmentation Mask if Not Provided ###
    if seg_mask is None:
        seg_mask = torch.zeros((1, 1, M, N))  # Initialize segmentation mask with zeros

    if len(seg_mask.shape) == 3:  # Ensure seg_mask has 4 dimensions
        seg_mask = seg_mask.unsqueeze(0)
    if len(seg_mask.shape) != 4:
        raise ValueError("Segmentation mask has invalid shape")  # Raise error for invalid shape

    if flag_plot:
        imshow_torch_temp(seg_mask[0], name="seg_mask_before_scale")  # Display the segmentation mask before scaling

    ### Average Kernel: ###
    kernel_masks_weighted_by_segmentation = (masks * seg_mask.to(masks.device)).mean([-1,-2], True)  # Weighted masks by segmentation mask
    average_kernel_in_segmentation_torch = (basis_kernels * kernel_masks_weighted_by_segmentation).sum(1,True).squeeze()  # Compute the average kernel

    ### Average Kernel Thresholded and Straight Line Fit: ###
    threshold = average_kernel_in_segmentation_torch.quantile(threshold_quantile)
    threshold = max(threshold, 5e-5)
    average_kernel_in_segmentation_thresholded = (average_kernel_in_segmentation_torch > threshold).float()
    average_kernel_in_segmentation_thresholded_RGB = BW2RGB(average_kernel_in_segmentation_thresholded).cpu().numpy()
    average_kernel_in_segmentation_straight_line, angle = fit_straight_line_to_kernel(average_kernel_in_segmentation_thresholded.cpu().numpy())
    average_kernel_in_segmentation_thresholded_torch = torch.tensor(average_kernel_in_segmentation_thresholded).cuda()
    average_kernel_in_segmentation_straight_line_torch = torch.tensor(average_kernel_in_segmentation_straight_line).cuda()

    ### Noramlize: ###
    average_kernel_in_segmentation_torch = average_kernel_in_segmentation_torch / average_kernel_in_segmentation_torch.sum()
    average_kernel_in_segmentation_thresholded_torch = average_kernel_in_segmentation_thresholded_torch / average_kernel_in_segmentation_thresholded_torch.sum()
    average_kernel_in_segmentation_straight_line_torch = average_kernel_in_segmentation_straight_line_torch / average_kernel_in_segmentation_straight_line_torch.sum()
    # display_media(cv2.resize((average_kernel_in_segmentation.cpu().numpy()*255).astype(np.uint8), (33*5,33*5)))
    # display_media(cv2.resize((average_kernel_in_segmentation_thresholded.cpu().numpy()*255).astype(np.uint8), (33*5,33*5)))
    # display_media(cv2.resize((average_kernel_in_segmentation_straight_line*255).astype(np.uint8), (33*5,33*5)))
    # average_kernel_per_pixel = (masks * seg_mask.to(masks.device))  #TODO: i left it alone because it would require too much memory

    if flag_plot:
        imshow_torch_temp(average_kernel_in_segmentation_torch.unsqueeze(0), "avg_kernel")  # Display the average kernel

    return average_kernel_in_segmentation_torch, average_kernel_in_segmentation_thresholded_torch, average_kernel_in_segmentation_straight_line_torch  # Return the average kernel


# def find_kernels_NUBKE(blurred_image, output_dir):
#     K = 25  # number of elements en the base
#     #Todo check with Yoav the correct pkl file !!!
#     gamma_factor = 2.2
#
#
#     if os.path.exists(output_dir) is False:
#         os.mkdir(output_dir)
#
#     two_heads = TwoHeadsNetwork(K).to(DEVICE)
#     print('loading weight\'s model')
#     two_heads.load_state_dict(torch.load(model_file, map_location='cuda:%d' % DEVICE))
#
#     two_heads.eval()
#     # todo: blurred_image to range [0, 1]
#     blurred_image = blurred_image.squeeze()
#     # Kernels and masks are estimated
#     blurry_tensor_to_compute_kernels = blurred_image ** gamma_factor - 0.5
#     kernels_estimated, masks_estimated = two_heads(blurry_tensor_to_compute_kernels[None, :, :, :])
#
#     # kernels_grid = my_own_make_grid(kernels_estimated.squeeze())
#     # imshow_torch_temp(255 * kernels_grid)
#
#
#     kernels_val_n = kernels_estimated[0, :, :, :]
#     kernels_val_n_ext = kernels_val_n[:, np.newaxis, :, :]
#
#     blur_kernel_val_grid = make_grid(kernels_val_n_ext, nrow=K,
#                                      normalize=True, scale_each=True, pad_value=1)
#     mask_val_n = masks_estimated[0, :, :, :]
#     mask_val_n_ext = mask_val_n[:, np.newaxis, :, :]
#     blur_mask_val_grid = make_grid(mask_val_n_ext, nrow=K, pad_value=1)
#
#     imsave(os.path.join(output_dir, '_kernels.png'),
#            img_as_ubyte(blur_kernel_val_grid.detach().cpu().numpy().transpose((1, 2, 0))))
#
#     imsave(os.path.join(output_dir, '_masks.png'),
#            img_as_ubyte(blur_mask_val_grid.detach().cpu().numpy().transpose((1, 2, 0))))
#
#     win_kernels_grid = save_kernels_grid(blurred_image, torch.flip(kernels_estimated[0], dims=(1, 2)),
#                                          masks_estimated[0],
#                                          os.path.join(output_dir, '_kernels_grid.png'))
#
#     # for i in range(K):
#     #     fig = imshow_torch(kernels_val_n[i])
#     #     plt.savefig(f"/raid/yoav/temp_garbage/base_kernel_{str(i).zfill(2)}.png")
#
#     return kernels_val_n

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



def find_kernels_NUBKE(blurred_image, output_dir=None, device='cuda:0', model=None, model_full_filename='', K_number_of_base_elements=25, gamma_factor=2.2, flag_save=False):
    """
    Finds and saves kernels and masks estimated by the NUBKE model for a given blurred image.

    Inputs:
    - blurred_image: torch.Tensor of shape [C, H, W], the input blurred image.
    - output_dir: str, the directory to save the output images.
    - model_file: str, the path to the pre-trained model file.
    - device: str, the device to run the model on ('cpu' or 'cuda:X').

    Outputs:
    - kernels_val_n: torch.Tensor of shape [K, H, W], the estimated kernels.
    """

    ### Create Output Directory if It Doesn't Exist ###
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)  # Create the output directory

    ### Load Pre-trained Model ###
    if model is not None:
        two_heads = model.eval() # Set model to evaluation mode
        two_heads = two_heads.cuda()  # Set model to evaluation mode
    else:
        two_heads = TwoHeadsNetwork(K_number_of_base_elements).to(device)  # Initialize the model and move it to the specified device
        print("Loading model weights")
        two_heads.load_state_dict(torch.load(model_full_filename, map_location=device))  # Load model weights
        two_heads.eval()  # Set model to evaluation mode
        two_heads = two_heads.cuda()  # Set model to evaluation mode

    ### Preprocess Blurred Image ###
    blurred_image = blurred_image.squeeze()  # Remove singleton dimensions
    blurry_tensor_to_compute_kernels = blurred_image ** gamma_factor - 0.5  # Apply gamma correction

    ### Estimate Kernels and Masks ###
    with torch.no_grad():  # Disable gradient computation for inference
        kernels_estimated, masks_estimated = two_heads(BW2RGB(torch_get_4D(blurry_tensor_to_compute_kernels.cuda())))  # Estimate kernels and masks

    ### Prepare and Save Kernel Grid Image ###
    output_dir = r'C:\Users\orior\PycharmProjects\VideoImageEnhancement'
    kernels_val_n = kernels_estimated[0]  # Extract the first set of estimated kernels
    kernels_val_n_ext = kernels_val_n[:, np.newaxis, :, :]  # Add a new axis for the channel dimension
    blur_kernel_val_grid = make_grid(kernels_val_n_ext, nrow=K_number_of_base_elements, normalize=True, scale_each=True, pad_value=1)  # Create a grid of kernel images
    if flag_save and output_dir is not None:
        imsave(os.path.join(output_dir, '_kernels.png'), img_as_ubyte(blur_kernel_val_grid.detach().cpu().numpy().transpose((1, 2, 0))))  # Save the kernel grid image

    ### Prepare and Save Mask Grid Image ###
    mask_val_n = masks_estimated[0]  # Extract the first set of estimated masks
    mask_val_n_ext = mask_val_n[:, np.newaxis, :, :]  # Add a new axis for the channel dimension
    blur_mask_val_grid = make_grid(mask_val_n_ext, nrow=K_number_of_base_elements, pad_value=1)  # Create a grid of mask images
    if flag_save and output_dir is not None:
        imsave(os.path.join(output_dir, '_masks.png'), img_as_ubyte(blur_mask_val_grid.detach().cpu().numpy().transpose((1, 2, 0))))  # Save the mask grid image

    ### Save Kernels Grid Image ###
    (image_with_blur_kernels_on_it,
     image_with_blur_kernels_thresholded_on_it,
     image_with_blur_kernels_straight_line_on_it) = save_kernels_grid(blurred_image,
                                                                      torch.flip(kernels_estimated[0], dims=(1, 2)),
                                                                      masks_estimated[0],
                                                                      flag_threshold_blur_kernel=True,
                                                                      quantile_to_threshold=0.95)  # Save the kernels grid image
    # imsave(os.path.join(output_dir, '_kernels_grid.png'), img_as_ubyte(image_with_blur_kernels_on_it.transpose((1, 2, 0))))
    # imsave(os.path.join(output_dir, '_kernels_grid_thresholded.png'), img_as_ubyte(image_with_blur_kernels_thresholded_on_it.transpose((1, 2, 0))))
    # imsave(os.path.join(output_dir, '_kernels_grid_straight_line.png'), img_as_ubyte(image_with_blur_kernels_straight_line_on_it.transpose((1, 2, 0))))
    # display_media((image_with_blur_kernels_on_it*255).astype(np.uint8).transpose([1,2,0]))
    # display_media((image_with_blur_kernels_thresholded_on_it*255).astype(np.uint8).transpose([1,2,0]))
    # display_media((image_with_blur_kernels_straight_line_on_it*255).astype(np.uint8).transpose([1,2,0]))

    # if flag_save and output_dir is not None:
    #     image_with_blur_kernels_on_it = save_kernels_grid(blurred_image, torch.flip(kernels_estimated[0], dims=(1, 2)), masks_estimated[0], os.path.join(output_dir, '_kernels_grid.png'))  # Save the kernels grid image
    #     # imsave(os.path.join(output_dir, '_kernels_grid.png'), img_as_ubyte(image_with_blur_kernels_on_it.transpose((1, 2, 0))))
    return kernels_val_n, image_with_blur_kernels_on_it, image_with_blur_kernels_thresholded_on_it, image_with_blur_kernels_straight_line_on_it  # Return the estimated kernels


def find_kernel_NUBKE(blurred_image, seg_mask=None):
    K = 25  # number of elements en the base
    gamma_factor = 2.2

    two_heads = TwoHeadsNetwork(K).to(DEVICE)
    print('loading weight\'s model')
    two_heads.load_state_dict(torch.load(model_file, map_location='cuda:%d' % DEVICE))

    with torch.no_grad():
        blurry_tensor_to_compute_kernels = blurred_image ** gamma_factor - 0.5
        kernels, masks = two_heads(blurry_tensor_to_compute_kernels.unsqueeze(0))

    if seg_mask is None:
        seg_mask = torch.ones_like(blurred_image)

    avg_kernel = get_avg_kernel_from_segmentation_masks_and_kernels(kernels, masks, seg_mask)
    return avg_kernel


# def deblur_image_pipline_NUBKE(blurred_image, seg_mask=None):
#     K = 25  # number of elements en the base
#     gamma_factor = 2.2
#
#     two_heads = TwoHeadsNetwork(K).to(DEVICE)
#     print('loading weight\'s model')
#     two_heads.load_state_dict(torch.load(model_file, map_location='cuda:%d' % DEVICE))
#
#     initial_restoration_tensor = blurred_image.clone()
#
#     with torch.no_grad():
#         blurry_tensor_to_compute_kernels = blurred_image ** gamma_factor - 0.5
#         kernels, masks = two_heads(blurry_tensor_to_compute_kernels.unsqueeze(0))
#
#
#     avg_kernel = get_avg_kernel_from_segmentation_masks_and_kernels(blurry_tensor_to_compute_kernels, kernels, masks, seg_mask)
#     # imshow_torch_temp(avg_kernel, name="weighted_kernel_according_to_mask")
#
#     # imshow_torch_temp(masks[0, 1], name="mask_0")
#     # for i in np.arange(25):
#     #     imshow_torch_temp(kernels[0, i], name="kernel_" + str(i).zfill(2))
#
#     output = initial_restoration_tensor
#
#     with torch.no_grad():
#
#         if True:
#             blurred_image = blurred_image.unsqueeze(0)
#             output = output.unsqueeze(0)
#             # print(blurred_image.shape)
#             # print(tweets_per_accounts.shape)
#             # print(kernels.shape)
#             # print(masks.shape)
#             # print(blurred_image.max())
#             # print(tweets_per_accounts.max())
#             # print(kernels.max())
#             # print(masks.max())
#             output = combined_RL_restore(blurred_image,
#                                          output,
#                                          kernels,
#                                          masks,
#                                          30,
#                                          blurred_image.device,
#                                          SAVE_INTERMIDIATE=True,
#                                          saturation_threshold=0.99,
#                                          reg_factor=1e-3,
#                                          optim_iters=1e-6,
#                                          gamma_correction_factor=2.2,
#                                          apply_dilation=False,
#                                          apply_smoothing=True,
#                                          apply_erosion=True)
#         else:
#             output = RL_restore(blurred_image.unsqueeze(0), output, kernels, masks, 30, blurred_image.device)
#
#     imshow_torch_temp(output[0].clip(0, 1), name="NUBKE_estimation")
#
#     return output

def get_KernelBasis_Masks_AvgKernel_From_NUBKE(blurred_image, seg_mask, NUBKE_model, gamma_factor):
    with torch.no_grad():  # Disable gradient computation for inference
        ### Preprocess Blurred Image ###
        blurry_tensor_to_compute_kernels = blurred_image ** gamma_factor - 0.5  # Apply gamma correction

        ### Estimate Kernels and Masks ###
        kernels, masks = NUBKE_model(blurry_tensor_to_compute_kernels.unsqueeze(0).cuda()) # Estimate kernels and masks

    ### Compute Average Kernel ###
    (average_kernel_in_segmentation,
     average_kernel_in_segmentation_thresholded_torch,
     average_kernel_in_segmentation_straight_line_torch) = get_avg_kernel_from_segmentation_masks_and_kernels2(kernels,
                                                                    masks,
                                                                    seg_mask)  # Compute the average kernel using the segmentation mask
    return kernels, masks, average_kernel_in_segmentation, average_kernel_in_segmentation_thresholded_torch, average_kernel_in_segmentation_straight_line_torch



def get_deblurred_image_from_kernels_and_masks(blurred_image,
                                               kernels_basis_tensor,
                                               masks,
                                               avg_kernel,
                                               K=25,
                                               n_iters=30,
                                               device='cuda',
                                               SAVE_INTERMIDIATE=True,
                                               saturation_threshold=0.99,
                                               reg_factor=1e-3,
                                               optim_iters=1e-6,
                                               gamma_correction_factor=2.2,
                                               apply_dilation=False,
                                               apply_smoothing=True,
                                               apply_erosion=True,
                                               flag_use_avg_kernel_on_everything=False):
    ### Initialize: ###
    initial_restoration_tensor = blurred_image.clone()  # Clone the blurred image for initial restoration
    output = initial_restoration_tensor  # Initialize output with the initial restoration tensor

    ### Switch to average kernel on everything if wanted: ###
    #TODO: doesn't work for this version of the code
    if flag_use_avg_kernel_on_everything:
        kernels_basis_tensor = avg_kernel.unsqueeze(0).unsqueeze(0).repeat(1,K,1,1)
        masks = torch.ones_like(masks)/K  # Set all masks to

    ### Perform RL Restoration ###
    with torch.no_grad():  # Disable gradient computation for inference
        ### Deblur the Image ###
        blurred_image = blurred_image.unsqueeze(0)  # Add batch dimension to the blurred image
        output = output.unsqueeze(0)  # Add batch dimension to the output

        ### Apply RL Restoration Algorithm ###
        output = combined_RL_restore(blurred_image,
                                     output,
                                     kernels_basis_tensor,
                                     masks,
                                     n_iters=n_iters,
                                     GPU=device,
                                     SAVE_INTERMIDIATE=SAVE_INTERMIDIATE,
                                     saturation_threshold=saturation_threshold,
                                     reg_factor=reg_factor,
                                     optim_iters=optim_iters,
                                     gamma_correction_factor=gamma_correction_factor,
                                     apply_dilation=apply_dilation,
                                     apply_smoothing=apply_smoothing,
                                     apply_erosion=apply_erosion)  # Apply the combined RL restoration algorithm


    return output


def crop_image_around_polygon(image, polygon_points):
    """
    Crops the image to the minimum bounding box surrounding the polygon points.

    Inputs:
    - image: numpy array of shape [H, W, C], the input image.
    - polygon_points: list of tuples, each tuple contains (x, y) coordinates of the points forming the polygon.

    Outputs:
    - cropped_image: numpy array of shape [H', W', C], the cropped image surrounding the polygon.
    """
    ### This Is The Code Block: ###
    polygon_points = np.array(polygon_points, dtype=np.int32)  # Convert polygon points to numpy array of type int32

    ### Get Bounding Box Coordinates ###
    x_min = np.min(polygon_points[:, 0])  # Minimum x-coordinate
    x_max = np.max(polygon_points[:, 0])  # Maximum x-coordinate
    y_min = np.min(polygon_points[:, 1])  # Minimum y-coordinate
    y_max = np.max(polygon_points[:, 1])  # Maximum y-coordinate

    ### Crop the Image Using Bounding Box Coordinates ###
    cropped_image = image[y_min:y_max, x_min:x_max]  # Crop the image to the bounding box

    return cropped_image  # Return the cropped image

def crop_image_around_segmentation_mask(image, mask):
    """
    Crops the image to the minimum bounding box surrounding the binary mask.

    Inputs:
    - image: numpy array of shape [H, W, C], the input image.
    - mask: numpy array of shape [H, W], the binary segmentation mask with values 0 and 1.

    Outputs:
    - cropped_image: numpy array of shape [H', W', C], the cropped image surrounding the mask.
    """
    ### This Is The Code Block: ###
    assert image.shape[:2] == mask.shape, "Image and mask must have the same height and width"  # Ensure image and mask dimensions match

    ### Get Non-Zero Mask Coordinates ###
    y_indices, x_indices = np.where(mask == 1)  # Get the indices of the mask where the value is 1

    ### Get Bounding Box Coordinates ###
    x_min = np.min(x_indices)  # Minimum x-coordinate
    x_max = np.max(x_indices)  # Maximum x-coordinate
    y_min = np.min(y_indices)  # Minimum y-coordinate
    y_max = np.max(y_indices)  # Maximum y-coordinate

    ### Crop the Image Using Bounding Box Coordinates ###
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]  # Crop the image to the bounding box

    return cropped_image  # Return the cropped image


def deblur_image_pipeline_NUBKE(blurred_image,
                                seg_mask=None,
                                device='cuda:0',
                                NUBKE_model=None,
                                n_iters=30,
                                SAVE_INTERMIDIATE=True,
                                saturation_threshold=0.99,
                                K=25,
                                reg_factor=1e-3,
                                optim_iters=1e-6,
                                gamma_correction_factor=2.2,
                                apply_dilation=False,
                                apply_smoothing=True,
                                apply_erosion=True,
                                flag_use_avg_kernel_on_everything=True,
                                flag_plot=False):
    """
    Deblurs an image using the NUBKE pipeline with an optional segmentation mask.

    Inputs:
    - blurred_image: torch.Tensor of shape [C, H, W], the input blurred image.
    - model_file: str, the path to the pre-trained model file.
    - seg_mask: torch.Tensor of shape [H, W], the optional segmentation mask.
    - device: str, the device to run the model on ('cpu' or 'cuda:X').

    Outputs:
    - output: torch.Tensor of shape [C, H, W], the deblurred image.
    """
    ### Initial Restoration Tensor ###
    (kernels_basis_tensor, masks,
     avg_kernel, avg_kernel_thresholded, avg_kernel_straight_line) = get_KernelBasis_Masks_AvgKernel_From_NUBKE(
        blurred_image,
        seg_mask,
        NUBKE_model,
        gamma_factor=2.2)

    ### Deblur the Image ###
    deblurred_entire_image = get_deblurred_image_from_kernels_and_masks(
        blurred_image.cuda(),
        kernels_basis_tensor,
        masks,
        avg_kernel,  #TODO: maybe use the thresholded or straight line fit average blur kernel
        K=K,
        n_iters=n_iters,
        device=device,
        SAVE_INTERMIDIATE=SAVE_INTERMIDIATE,
        saturation_threshold=saturation_threshold,
        reg_factor=reg_factor,
        optim_iters=optim_iters,
        gamma_correction_factor=gamma_correction_factor,
        apply_dilation=apply_dilation,
        apply_smoothing=apply_smoothing,
        apply_erosion=apply_erosion,
        flag_use_avg_kernel_on_everything=flag_use_avg_kernel_on_everything)

    ### Get Image Crops After Deblur: ###
    deblurred_crop = crop_image_around_segmentation_mask(torch_to_numpy(deblurred_entire_image.squeeze()), torch_to_numpy(seg_mask.squeeze()))

    ### Show Output: ###
    if flag_plot:
        imshow_torch_temp(deblurred_entire_image[0].clip(0, 1), name="NUBKE_estimation")  # Display the deblurred image

    return avg_kernel, avg_kernel_thresholded, avg_kernel_straight_line, deblurred_entire_image, deblurred_crop   # Return the deblurred image



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


def imshow_torch_temp(image, name="temp"):
    #ToDo rename it
    plt.imshow(image.permute(1, 2, 0).cpu().detach().numpy())
    plt.savefig(f"{output_dir}/{name}.png")

def get_full_shape_torch(input_tensor):
    if len(input_tensor.shape) == 1:
        W = input_tensor.shape
        H = 1
        C = 1
        T = 1
        B = 1
        shape_len = 1
        shape_vec = (W)
    elif len(input_tensor.shape) == 2:
        H, W = input_tensor.shape
        C = 1
        T = 1
        B = 1
        shape_len = 2
        shape_vec = (H,W)
    elif len(input_tensor.shape) == 3:
        C, H, W = input_tensor.shape
        T = 1
        B = 1
        shape_len = 3
        shape_vec = (C,H,W)
    elif len(input_tensor.shape) == 4:
        T, C, H, W = input_tensor.shape
        B = 1
        shape_len = 4
        shape_vec = (T,C,H,W)
    elif len(input_tensor.shape) == 5:
        B, T, C, H, W = input_tensor.shape
        shape_len = 5
        shape_vec = (B,T,C,H,W)
    shape_vec = np.array(shape_vec)
    return (B,T,C,H,W), shape_len, shape_vec

def torch_to_numpy(input_tensor):
    if type(input_tensor) == torch.Tensor:
        (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
        input_tensor = input_tensor.cpu().data.numpy()
        if shape_len == 2:
            #[H,W]
            return input_tensor
        elif shape_len == 3:
            #[C,H,W] -> [H,W,C]
            return np.transpose(input_tensor, [1,2,0])
        elif shape_len == 4:
            #[T,C,H,W] -> [T,H,W,C]
            return np.transpose(input_tensor, [0,2,3,1])
        elif shape_len == 5:
            #[B,T,C,H,W] -> [B,T,H,W,C]
            return np.transpose(input_tensor, [0,1,3,4,2])
    return input_tensor

def create_segmentation_mask(image_shape, polygon_points):
    """
    Creates a segmentation mask with ones inside the polygon areas.

    Inputs:
    - image_shape: tuple, the shape of the image (height, width).
    - polygon_points: list of tuples, each tuple contains (x, y) coordinates of the points forming the polygon.

    Outputs:
    - mask: numpy array of shape [height, width], the segmentation mask with ones inside the polygon areas.
    """
    ### This Is The Code Block: ###
    height, width = image_shape  # Extract height and width from the image shape
    mask = np.zeros((height, width), dtype=np.uint8)  # Initialize the mask with zeros

    ### Convert polygon points to numpy array ###
    polygon = np.array(polygon_points, dtype=np.int32)  # Convert polygon points to numpy array of type int32

    ### Fill the polygon area with ones ###
    cv2.fillPoly(mask, [polygon], 1)  # Fill the polygon area with ones in the mask

    return mask  # Return the segmentation mask

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

def imshow_torch(image, flag_colorbar=False, title_str='', flag_maximize=False):
    fig = plt.figure()
    # plt.cla()
    plt.clf()
    # plt.close()

    if len(image.shape) == 4:
        plt.imshow(np.transpose(image[0].detach().cpu().numpy(), (1, 2, 0)).squeeze())  # transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 3:
        plt.imshow(np.transpose(image.detach().cpu().numpy(),(1,2,0)).squeeze()) #transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 2:
        plt.imshow(image.detach().cpu().numpy())

    if flag_colorbar:
        plt.colorbar()  #TODO: fix the bug of having multiple colorbars when calling this multiple times
    plt.title(title_str)
    plt.show()

    if flag_maximize:
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

    return fig


def main_interface(image_path, output_dir_for_global):
    # global output_dir
    # output_dir = output_dir_for_global
    # if os.path.exists(output_dir_for_global) is False:
    #     os.mkdir(output_dir_for_global)

    #### load_from_segmentation file
    # seg_path = "/home/dudy/Nehoray/segment_anything_base_dir/Grounded-Segment-Anything/outputs/output_minivan_1/27/masks/27_mask_car_3.pt"
    # seg_mask = torch.load(seg_path)

    # ###Load image and let user to draw segmentation mask
    # img_segmenter = ImageSegmenter(image_path)
    # seg_mask, image_base_segment = img_segmenter.get_mask()

    ### Paths: ###
    image_path = r"C:\Users\dudyk\Documents\RDND_dudy\SHABAK\blurred_image.png"
    output_dir = r"C:\Users\dudyk\Documents\RDND_dudy\SHABAK"

    ### Dudy working shit: ###
    blurred_image_torch = read_image_torch(image_path).to(DEVICE)
    input_image_torch = read_image_torch(image_path)
    input_image_np = torch_to_numpy(input_image_torch[0])

    ### Select points using OpenCV and build polygon: ###
    selected_points = select_points_and_build_polygon_opencv(input_image_np/255, scale_factor=1)

    ### Get Segmentaiton Mask: ###
    seg_mask = create_segmentation_mask(input_image_np.shape[0:2], selected_points)
    seg_mask_torch = numpy_to_torch(seg_mask)
    # imshow_np(seg_mask, "Segmented Mask")

    ### Load Model: ###
    K_number_of_base_elements = 25  # Number of elements in the base
    gamma_factor = 2.2  # Gamma correction factor
    device = 'cuda'
    NUBKE_model = TwoHeadsNetwork(K_number_of_base_elements).to('cuda')  # Initialize the model and move it to the specified device
    print("Loading model weights")
    NUBKE_model.load_state_dict(torch.load(model_file, map_location=device))  # Load model weights
    NUBKE_model.eval()  # Set model to evaluation mode

    ### Get Blur Kernel: ###
    # [K_kernels_basis_tensor] = [K_number_of_base_elements, K_size, K_size]
    K_kernels_basis_tensor = find_kernels_NUBKE(blurred_image_torch.squeeze() / 255,
                                                output_dir,
                                                model=NUBKE_model,
                                                model_full_filename='',
                                                K_number_of_base_elements=K_number_of_base_elements,
                                                gamma_factor=gamma_factor,
                                                flag_save=False)  # Gamma correction factor)

    ### Deblur The Image: ###
    avg_kernel, deblurred_image, deblurred_crop = deblur_image_pipeline_NUBKE(blurred_image_torch.squeeze() / 255,
                                                              seg_mask=seg_mask_torch,
                                                              NUBKE_model=NUBKE_model,
                                                              n_iters=30,
                                                              SAVE_INTERMIDIATE=True,
                                                              saturation_threshold=0.99,
                                                              K=K_number_of_base_elements, # TODO: probably doesn't need this because it is only necessary for model initialization and visualizations
                                                              reg_factor=1e-3,
                                                              optim_iters=1e-6,
                                                              gamma_correction_factor=gamma_factor,
                                                              apply_dilation=False,
                                                              apply_smoothing=True,
                                                              apply_erosion=True,
                                                              flag_use_avg_kernel_on_everything=False,
                                                              flag_plot=False)
    # imshow_torch(blurred_image_torch/255)
    # imshow_torch(deblurred_image)
    # imshow_torch(deblurred_image, title_str="Deblurred Image")
    # imshow_torch(avg_kernel, title_str="Average Kernel")
    return avg_kernel, K_kernels_basis_tensor, deblurred_image, deblurred_crop


def get_blur_kernel_and_deblurred_image_using_NUBKE(blurry_image_torch,
                                                    segmentation_mask_torch=None,
                                                    NUBKE_model=None,
                                                    device='cuda',
                                                    n_iters=30,
                                                    SAVE_INTERMIDIATE=True,
                                                    saturation_threshold=0.99,
                                                    K_number_of_base_elements=25, # TODO: probably doesn't need this because it is only necessary for model initialization and visualizations
                                                    reg_factor=1e-3,
                                                    optim_iters=1e-6,
                                                    gamma_correction_factor=2.2,
                                                    apply_dilation=False,
                                                    apply_smoothing=True,
                                                    apply_erosion=True,
                                                    flag_use_avg_kernel_on_everything=True):
    ### [blurry_image_torch] is [3, H, W]
    ### [segmentation_mask_torch] is [1, H, W]

    ### Load Model: ###
    if NUBKE_model is None:
        NUBKE_model = TwoHeadsNetwork(K_number_of_base_elements).to('cuda')  # Initialize the model and move it to the specified device
        print("Loading model weights")
        NUBKE_model.load_state_dict(torch.load(model_file, map_location=device))  # Load model weights
        NUBKE_model.eval()  # Set model to evaluation modw
        NUBKE_model= NUBKE_model.cuda()  # Set model to evaluation modw

    ### Segmentation Mask: ###
    if segmentation_mask_torch is None:
        H,W = blurry_image_torch.shape[-2:]
        segmentation_mask_torch = torch.ones((1,H,W)).to(device) > 0  # Assume

    ### Get Blur Kernel: ###
    # [K_kernels_basis_tensor] = [K_number_of_base_elements, K_size, K_size]
    (K_kernels_basis_tensor,
     image_with_blur_kernels_on_it,
     image_with_blur_kernels_thresholded_on_it,
     image_with_blur_kernels_straight_line_on_it) = find_kernels_NUBKE(BW2RGB(blurry_image_torch) / 255,
                                                                       # TODO: change back
                                                                       output_dir=None,
                                                                       model=NUBKE_model,
                                                                       model_full_filename='',
                                                                       K_number_of_base_elements=K_number_of_base_elements,
                                                                       gamma_factor=gamma_correction_factor,
                                                                       flag_save=False)  # Gamma correction factor)

    ### Deblur The Image: ###
    (avg_kernel, avg_kernel_thresholded,
     avg_kernel_straight_line, deblurred_image, deblurred_crop) = deblur_image_pipeline_NUBKE(blurry_image_torch / 255,
                                                              seg_mask=segmentation_mask_torch,
                                                              NUBKE_model=NUBKE_model,
                                                              n_iters=n_iters,
                                                              SAVE_INTERMIDIATE=SAVE_INTERMIDIATE,
                                                              saturation_threshold=saturation_threshold,
                                                              K=K_number_of_base_elements, #TODO: probably doesn't need this because it is only necessary for model initialization and visualizations
                                                              reg_factor=reg_factor,
                                                              optim_iters=optim_iters,
                                                              gamma_correction_factor=gamma_correction_factor,
                                                              apply_dilation=apply_dilation,
                                                              apply_smoothing=apply_smoothing,
                                                              apply_erosion=apply_erosion,
                                                              flag_use_avg_kernel_on_everything=flag_use_avg_kernel_on_everything,
                                                              flag_plot=False)

    return (avg_kernel, avg_kernel_thresholded, avg_kernel_straight_line,
            K_kernels_basis_tensor, deblurred_image, deblurred_crop,
            image_with_blur_kernels_on_it, image_with_blur_kernels_thresholded_on_it, image_with_blur_kernels_straight_line_on_it)



#
# image_path = r"C:\Users\dudyk\Documents\RDND_dudy\SHABAK\blurred_image.png"
# output_dir = r"C:\Users\dudyk\Documents\RDND_dudy\SHABAK"
# main_interface(image_path=image_path, output_dir_for_global="output4")


