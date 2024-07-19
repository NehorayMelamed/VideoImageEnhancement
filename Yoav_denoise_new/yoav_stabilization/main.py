import math
import shutil
from glob import glob
from typing import Tuple, Union, Callable, Dict, List

import cv2
import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor


def raise_if(condition: bool, error: Exception = RuntimeError, message: str = "RuntimeError") -> None:
    if condition:
        raise error(message)


def raise_if_not(condition: bool, error: Exception = RuntimeError, message: str = "RuntimeError") -> None:
    if not condition:
        raise error(message)


def raise_if_not_close(a: Union[int, float], b: Union[float, int], error: Exception = RuntimeError,
                       message: str = "RuntimeError", closeness_distance: float = 1e-7) -> None:
    if closeness_distance < abs(a-b):
        raise error(message)

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



def type_check_parameters(arguments: List[Tuple]) -> None:
    """
    :param arguments: List of tuples of params and what their types can be
    """
    for num_arg, arg in enumerate(arguments):
        if type(arg[1]) in [list, tuple]:
            raise_if_not(type(arg[0]) in arg[1], message=f"Type {type(arg[0])} not valid for argument {num_arg}. Valid types: {arg[1]}")
        else:
            raise_if_not(type(arg[0]) == arg[1], message=f"Type {type(arg[0])} not valid for argument {num_arg}. Valid type: {arg[1]}")

def construct_tensor(elements: Union[Tensor, list, tuple, float, int, np.array], device: Union[str, torch.device] = "cpu"):
    if type(elements) in [float, int]:
        return torch.tensor([elements], device=device)
    # we have a device issue here - if tensor, leave the original device or change?
    elif type(elements) == Tensor:
        if len(elements.shape) == 0:  # is tensor of form torch.tensor(float), as opposed to torch.tensor([float]). Problem since former type is not iterable
            return torch.tensor([float(elements)]).to(elements.device)
        return elements.to(elements.device)
    elif type(elements) in [list, tuple]:
        return convert_iterable_tensor(elements).to(device)
    else:
        return torch.tensor(elements, device=device)

def convert_iterable_tensor(elements: Union[list, tuple]) -> Tensor:
    # recursively converts a list of any form into Tensor.
    if type(elements[0]) == Tensor:
        return torch.stack(elements)
    elif type(elements[0]) in [float, int, np.float, np.float64, np.float32, np.int, np.uint8, np.int16]:
        return Tensor(elements)
    else:
        return torch.stack([convert_iterable_tensor(iterable) for iterable in elements])

def true_dimensionality(matrix: Tensor) -> int:
    first_dim = len(matrix.shape)
    for num_dim, dim in enumerate(matrix.shape):
        if dim != 1:
            first_dim = num_dim
            break
    return len(matrix.shape)-first_dim


def compare_unequal_dimensionality_tensors(greater_dim_vector: Tensor, lesser_dim_vector: Tensor) -> bool:
    # returns if the tensors are the same size, consdiering dimensions of size 1 to be irrelevant
    # can also be used to compare equal sized vectors
    for i in range(-1, -len(greater_dim_vector.shape)-1, -1):
        try:
            if greater_dim_vector.shape[i] != lesser_dim_vector.shape[i]:
                return False
        except IndexError:
            if greater_dim_vector.shape[i] != 1:
                return False
    return True

def compare_unequal_outer_dimensions(greater_dim_vector: Tensor, lesser_dim_vector: Tensor) -> bool:
    # A general test for cases where given tensors of different B, T dimensions
    # return true if lesser_dim_vector is a legitimate reference matrix for greater_dim_vector
    # Phase 1: CHW dimensions. must both exist and equal
    for i in range(-1, max(-len(greater_dim_vector.shape) - 1, -4), -1):
        try:
            if greater_dim_vector.shape[i] != lesser_dim_vector.shape[i]:
                return False
        except IndexError:
            return False
    # Phase 2: BT dimensions. False if lesser_dim_vector is not "private case" of greater_dim_vector.
    for i in range(-4, -len(greater_dim_vector.shape) - 1, -1):
        try:
            if greater_dim_vector.shape[i] != lesser_dim_vector.shape[i]:
                if lesser_dim_vector.shape[i] > 1:
                    return False
        except IndexError:
            return True
    return True



def format_parameters_classic_circular_cc(matrix: Union[Tensor, np.array, tuple, list],
                                          reference_matrix: Union[Tensor, np.array, tuple, list],
                                          matrix_fft: torch.Tensor = None, reference_fft: torch.Tensor = None,
                                          normalize: bool = False,
                                          fftshift: bool = False) -> Tuple[
    Tensor, Tensor, Union[Tensor, None], Union[Tensor, None], bool, bool]:
    type_check_parameters(
        [(matrix, (Tensor, np.array, tuple, list)), (reference_matrix, (Tensor, np.array, tuple, list)),
         (matrix_fft, (Tensor, type(None))), (reference_fft, (Tensor, type(None))), (fftshift, bool),
         (normalize, bool)])
    matrix = construct_tensor(matrix)
    reference_matrix = construct_tensor(reference_matrix)
    raise_if(true_dimensionality(matrix) == 0, message="Matrix is empty")
    raise_if(true_dimensionality(reference_matrix) == 0, message="Reference Matrix is empty")
    raise_if_not(compare_unequal_outer_dimensions(matrix, reference_matrix),
                 message="Matrix and Reference matrix are not same size")
    if matrix_fft is not None:  # is not None
        raise_if(compare_unequal_dimensionality_tensors(matrix, matrix_fft),
                 message="Matrix and Matrix FFT are same size")
    if reference_fft is not None:
        raise_if(compare_unequal_dimensionality_tensors(reference_matrix, reference_fft),
                 message="Reference matrix and Reference matrix FFT are not same size")
    if len(matrix.shape) > 2:
        raise_if_not(matrix.shape[-3] != 1 or matrix.shape[-3] != 3, message="Matrix must be grayscale or RGB")
    if len(reference_matrix.shape) > 2:
        raise_if_not(matrix.shape[-3] != 1 or matrix.shape[-3] != 3, message="Matrix must be grayscale or RGB")
    gray_matrix = RGB2BW(matrix)
    gray_reference_matrix = RGB2BW(reference_matrix)
    return gray_matrix, gray_reference_matrix, matrix_fft, reference_fft, normalize, fftshift



def get_color_formula_numpy(color_formula_triplet):
    formula_triplet = []
    if 0 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 1 in color_formula_triplet:
        formula_triplet.append(lambda x: 0.5)
    if 2 in color_formula_triplet:
        formula_triplet.append(lambda x: 1)
    if 3 in color_formula_triplet:
        formula_triplet.append(lambda x: x)
    if 4 in color_formula_triplet:
        formula_triplet.append(lambda x: x**2)
    if 5 in color_formula_triplet:
        formula_triplet.append(lambda x: x**3)
    if 6 in color_formula_triplet:
        formula_triplet.append(lambda x: x**4)
    if 7 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.5)
    if 8 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.25)
    if 9 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.sin(np.pi/2*x))
    if 10 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.cos(np.pi/2*x))
    if 11 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.abs(x-0.5))
    if 12 in color_formula_triplet:
        formula_triplet.append(lambda x: (2*x-1)**2)
    if 13 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.sin(np.pi*x))
    if 14 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(numpy.cos(np.pi*x)))
    if 15 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.sin(2*np.pi*x))
    if 16 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.cos(2*np.pi*x))
    if 17 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(numpy.sin(2*np.pi*x)))
    if 18 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(numpy.cos(2*np.pi*x)))
    if 19 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(numpy.sin(4*np.pi*x)))
    if 20 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(4*np.pi*x)))
    if 21 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x)
    if 22 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-1)
    if 23 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-2)
    if 24 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-1))
    if 25 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-2))
    if 26 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-1)/2)
    if 27 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-2)/2)
    if 28 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-1)/2))
    if 29 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-2)/2))
    if 30 in color_formula_triplet:
        formula_triplet.append(lambda x: x/0.32-0.78125)
    if 31 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.84)
    if 32 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 33 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(2*x-0.5))
    if 34 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x)
    if 35 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.5)
    if 36 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-1)
    return formula_triplet


def to_range(input_array, low, high):
    new_range_delta = high-low
    old_range_delta = input_array.max() - input_array.min()
    new_min = low
    old_min = input_array.min()
    input_array = ((input_array-old_min)*new_range_delta/old_range_delta) + new_min
    return input_array

def validate_warp_method(method: str, valid_methods=['bilinear', 'bicubic', 'nearest', 'fft']) -> None:
    raise_if_not(method in valid_methods, message="Invalid method")

def format_align_to_reference_frame_circular_cc_params(matrix: Union[Tensor, np.array, tuple, list],
                                                       reference_matrix: Union[Tensor, np.array, tuple, list],
                                                       matrix_fft: torch.Tensor=None,
                                                       reference_fft: torch.Tensor=None,
                                                       normalize: bool=False,
                                                       warp_method: str='bilinear',
                                                       crop_warped_matrix: bool=False,
                                                       fftshift: bool=False) -> Tuple[
    Tensor, Tensor, Union[Tensor, None], Union[Tensor, None], bool, str, bool, bool]:
    matrix, reference_matrix, matrix_fft, reference_fft, normalize, fftshift = format_parameters_classic_circular_cc(matrix, reference_matrix, matrix_fft,
                                                                                                                     reference_fft, normalize, fftshift)
    type_check_parameters([(warp_method, str), (crop_warped_matrix, bool)])
    validate_warp_method(warp_method, valid_methods=['bilinear', 'bicubic', 'nearest', 'fft'])

    return matrix, reference_matrix, matrix_fft, reference_fft, normalize, warp_method, crop_warped_matrix, fftshift

def gray2color_numpy(input_array, type_id=0):
    if type_id == 0:
        formula_id_triplet = [7,5,15]
        # formula_id_triplet = [3,4,5]
    elif type_id == 1:
        formula_id_triplet = [3,11,6]
    elif type_id == 2:
        formula_id_triplet = [23,28,3]
    elif type_id == 3:
        formula_id_triplet = [21,22,23]
    elif type_id == 4:
        formula_id_triplet = [30,31,32]
    elif type_id == 5:
        formula_id_triplet = [31,13,10]
    elif type_id == 6:
        formula_id_triplet = [34,35,36]

    formula_triplet = get_color_formula_numpy(formula_id_triplet);

    input_min = input_array.min()
    input_max = input_array.max()
    input_array = to_range(input_array,0,1)
    # input_array = input_array/256

    R = formula_triplet[0](input_array)
    G = formula_triplet[1](input_array)
    B = formula_triplet[2](input_array)
    R = R.clip(0,1)
    G = G.clip(0,1)
    B = B.clip(0,1)
    if len(R.shape)==4:
        color_array = numpy.concatenate([R,G,B], 3)
    else:
        color_array = numpy.concatenate([R,G,B], 2)

    # input_array = input_array*256
    color_array = to_range(color_array, input_min, input_max)




    return color_array

def path_make_path_if_none_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image_torch(folder_path=None, filename=None, torch_tensor=None, flag_convert_bgr2rgb=True,
                     flag_scale_by_255=False, flag_array_to_uint8=True, flag_imagesc=False, flag_convert_grayscale_to_heatmap=False,
                     flag_save_figure=False, flag_colorbar=False, flag_print=False):
    if flag_scale_by_255:
        scale_factor = 255
    else:
        scale_factor = 1

    if len(torch_tensor.shape) == 4:
        if flag_convert_bgr2rgb:
            saved_array = cv2.cvtColor(torch_tensor[0].cpu().data.numpy().transpose([1, 2, 0]) * scale_factor, cv2.COLOR_BGR2RGB)
        else:
            saved_array = torch_tensor[0].cpu().data.numpy().transpose([1, 2, 0]) * scale_factor
    else:
        if flag_convert_bgr2rgb:
            saved_array = cv2.cvtColor(torch_tensor.cpu().data.numpy().transpose([1, 2, 0]) * scale_factor, cv2.COLOR_BGR2RGB)
        else:
            saved_array = torch_tensor.cpu().data.numpy().transpose([1, 2, 0]) * scale_factor

    if flag_convert_grayscale_to_heatmap:
        if torch_tensor.shape[0]==1:
            #(1). Direct Formula ColorMap:
            saved_array = gray2color_numpy(saved_array,0)
            # #(2). Matplotlib Jet:
            # cmap = plt.cm.jet
            # norm = plt.Normalize(vmin=0, vmax=150)
            # gt_disparity = saved_array
            # gt_disparity2 = norm(gt_disparity)
            # gt_disparity3 = cmap(gt_disparity2)
            # saved_array = 255 * gt_disparity3[:,:,0:3]


    path_make_path_if_none_exists(folder_path)

    if flag_imagesc:
        new_range = (0, 255)
        new_range_delta = new_range[1]-new_range[0]
        old_range_delta = saved_array.max() - saved_array.min()
        new_min = new_range[0]
        old_min = saved_array.min()
        saved_array = ((saved_array-old_min)*new_range_delta/old_range_delta) + new_min

    if flag_array_to_uint8:
        saved_array = np.uint8(saved_array)

    if flag_convert_grayscale_to_heatmap:
        saved_array = cv2.cvtColor(saved_array, cv2.COLOR_BGR2RGB)

    if flag_save_figure:
        if len(saved_array.shape)==3:
            if saved_array.shape[2] == 1:
                plt.imshow(saved_array.squeeze())
            else:
                plt.imshow(saved_array)
        else:
            plt.imshow(saved_array)
        if flag_colorbar:
            plt.colorbar()
        plt.savefig(os.path.join(folder_path, filename))
        plt.close()
    else:
        cv2.imwrite(os.path.join(folder_path, filename), saved_array)

    if flag_print:
        # print(os.path.join(folder_path, filename))
        print(filename)


def calculate_meshgrids(input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
    ndims = len(input_tensor.shape)
    H = input_tensor.shape[-2]
    W = input_tensor.shape[-1]
    # Get tilt phases k-space:
    y = torch.arange(-math.floor(H / 2), math.ceil(H / 2), 1)
    x = torch.arange(-math.floor(W / 2), math.ceil(W / 2), 1)
    delta_f1 = 1 / H
    delta_f2 = 1 / W
    f_y = y * delta_f1
    f_x = x * delta_f2
    # Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
    f_y = torch.fft.fftshift(f_y)
    f_x = torch.fft.fftshift(f_x)

    # Build k-space meshgrid:
    [kx, ky] = torch.meshgrid(f_y, f_x, indexing='ij')
    # Frequency vec to tensor:
    for i in range(ndims - 2):
        kx = kx.unsqueeze(0)
        ky = ky.unsqueeze(0)
    if kx.device != input_tensor.device:
        kx = kx.to(input_tensor.device)
        ky = ky.to(input_tensor.device)
    return kx, ky

def fix_dimensionality(matrix: Tensor, num_dims: int) -> Tensor:
    # makes matrix requisite number of dimensions
    current_dims = len(matrix.shape)
    if current_dims <= num_dims:
        for i in range(num_dims - len(matrix.shape)):
            matrix = matrix.unsqueeze(0)
        return matrix
    else:
        raise RuntimeError("Tried to expand a Tensor to a size smaller than itself")



def fit_polynomial(x: Union[torch.Tensor, list], y: Union[torch.Tensor, list]) -> List[float]:
    # solve for 2nd degree polynomial deterministically using three points seperated by distance of 1
    a = (y[..., 2] + y[..., 0] - 2 * y[..., 1]) / 2
    b = -(y[..., 0] + 2 * a * x[1] - y[..., 1] - a)
    c = y[..., 1] - b * x[1] - a * x[1] ** 2
    return [c, b, a]



def read_image_torch(path, flag_convert_to_rgb=1, flag_normalize_to_float=0):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.imread(path)
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

class NoMemoryMatrixOrigami:
    # expands matrix to 5 dimensions, then folds it back to original size at the end
    def __init__(self, matrix: Tensor):
        self.original_dims = matrix.shape

    def expand_matrix(self, matrix: Tensor, num_dims: int = 5) -> Tensor:
        # makes matrix requisite number of dimensions
        return fix_dimensionality(matrix=matrix, num_dims=num_dims)

    def squeeze_to_original_dims(self, matrix: Tensor) -> Tensor:
        for dim in matrix.shape:
            if dim != 1:
                break
            matrix = matrix.squeeze(0)
        for i in range(len(self.original_dims) - len(matrix.shape)):
            matrix = matrix.unsqueeze(0)
        return matrix

def shifts_from_circular_cc(batch_cc: Tensor, midpoints: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
    # dudi wrote this function, I had nothing to do with it
    # Elisheva added the B dim to enable parallelism :)
    B, T, _, H, W = batch_cc.shape  # _ is C, but C must be 1
    output_CC_flattened_indices = torch.argmax(batch_cc.contiguous().view(B, T, H * W), dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // W
    i1 = output_CC_flattened_indices - i0 * W
    i0[i0 > midpoints[0]] -= H
    i1[i1 > midpoints[1]] -= W
    i0_original = i0 + 0
    i1_original = i1 + 0
    i0_minus1 = i0 - 1
    i0_plus1 = i0 + 1
    i1_minus1 = i1 - 1
    i1_plus1 = i1 + 1
    ### Correct For Wrap-Arounds: ###
    # (1). Below Zero:
    i0_minus1[i0_minus1 < 0] += H
    i1_minus1[i1_minus1 < 0] += W
    i0_plus1[i0_plus1 < 0] += H
    i1_plus1[i1_plus1 < 0] += W
    i0[i0 < 0] += H
    i1[i1 < 0] += W
    # (2). Above Max:
    i0[i0 > H] -= H
    i1[i1 > W] -= W  # TODO commit this change to Anvil
    i0_plus1[i0_plus1 > H] -= H
    i1_plus1[i1_plus1 > W] -= W
    i0_minus1[i0_minus1 > W] -= H
    i1_minus1[i1_minus1 > W] -= W
    ### Get Flattened Indices From Row/Col Indices: ###
    output_CC_flattened_indices_i1 = i1 + i0 * W
    output_CC_flattened_indices_i1_minus1 = i1_minus1 + i0_minus1 * W
    output_CC_flattened_indices_i1_plus1 = i1_plus1 + i0_plus1 * W
    output_CC_flattened_indices_i0 = i1 + i0 * W
    output_CC_flattened_indices_i0_minus1 = i1 + i0_minus1 * W
    output_CC_flattened_indices_i0_plus1 = i1 + i0_plus1 * W

    ### Get Proper Values For Fit: ###
    output_CC = batch_cc.contiguous().view(B, T, H * W)
    output_CC_flattened_values_i0 = torch.gather(output_CC, -1, output_CC_flattened_indices_i0.long())
    output_CC_flattened_values_i0_minus1 = torch.gather(output_CC, -1, output_CC_flattened_indices_i0_minus1)
    output_CC_flattened_values_i0_plus1 = torch.gather(output_CC, -1, output_CC_flattened_indices_i0_plus1)
    output_CC_flattened_values_i1 = torch.gather(output_CC, -1, output_CC_flattened_indices_i1)
    output_CC_flattened_values_i1_minus1 = torch.gather(output_CC, -1, output_CC_flattened_indices_i1_minus1)
    output_CC_flattened_values_i1_plus1 = torch.gather(output_CC, -1, output_CC_flattened_indices_i1_plus1)

    ### Get Sub Pixel Shifts: ###
    fitting_points_x = torch.cat(
        [output_CC_flattened_values_i1_minus1, output_CC_flattened_values_i1, output_CC_flattened_values_i1_plus1], -1)
    fitting_points_y = torch.cat(
        [output_CC_flattened_values_i0_minus1, output_CC_flattened_values_i0, output_CC_flattened_values_i0_plus1], -1)
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(x_vec, fitting_points_x.squeeze())
    [c_y, b_y, a_y] = fit_polynomial(y_vec, fitting_points_y.squeeze())
    delta_shiftx = -b_x / (2 * a_x)
    delta_shifty = -b_y / (2 * a_y)
    # Add integer shift:
    shiftx = i1_original.squeeze() + delta_shiftx
    shifty = i0_original.squeeze() + delta_shifty
    return shifty, shiftx


def normalize_cc_matrix(cc_matrix: Tensor, matrix: Tensor, reference_matrix: Tensor) -> Tensor:
    H, W = cc_matrix.shape[-2:]
    A_sum = reference_matrix.sum(dim=[-1, -2])
    A_sum2 = (reference_matrix ** 2).sum(dim=[-1, -2])
    sigmaA = (A_sum2 - A_sum ** 2 / (H * W)) ** (1 / 2)
    sigmaB = (matrix).std(dim=[-1, -2]) * (H * W - 1) ** (1 / 2)
    B_mean = (matrix).mean(dim=[-1, -2])
    normalized_cc = (cc_matrix - (A_sum * B_mean).unsqueeze(-1).unsqueeze(-1)) / (sigmaA * sigmaB).unsqueeze(
        -1).unsqueeze(-1)
    return normalized_cc



def circular_cross_correlation_classic(matrix: Tensor, reference_matrix: Tensor, matrix_fft: Union[Tensor, None] = None,
                                       reference_matrix_fft: Union[Tensor, None] = None,
                                       normalize: bool = False, fftshift: bool = False) -> Tensor:
    #TODO: dudy
    # matrix_fft = faster_fft(matrix, matrix_fft)
    # reference_tensor_fft = faster_fft(reference_matrix, reference_matrix_fft)
    # cc = torch.fft.ifftn(matrix_fft * reference_tensor_fft.conj(), dim=[-1, -2]).real
    a=1+1
    cc = torch.fft.ifftn(faster_fft(matrix, matrix_fft) * faster_fft(reference_matrix, reference_matrix_fft).conj(), dim=[-1, -2]).real
    if normalize:
        cc = normalize_cc_matrix(cc, matrix, reference_matrix)
    if fftshift:
        cc = torch.fft.fftshift(cc, dim=[-2, -1])
    return cc


def non_interpolated_center_crop(matrix: Tensor, target_H: int, target_W: int) -> Tensor:
    B,T,C,H,W = matrix.shape
    excess_rows = H - target_H
    excess_columns = W - target_W
    start_y = int(excess_rows/2)
    start_x = int(excess_columns/2)
    stop_y = start_y+target_H
    stop_x = start_x+target_W
    return matrix[:, :, :, start_y:stop_y, start_x:stop_x]

def classic_circular_cc_shifts_calc(matrix: Tensor, reference_matrix: Tensor, matrix_fft: Tensor,
                                    reference_tensor_fft: Tensor,
                                    normalize: bool = False, fftshift: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    ## returns tuple of vertical shifts, horizontal shifts, and cross correlation
    cc = circular_cross_correlation_classic(matrix, reference_matrix, matrix_fft, reference_tensor_fft, normalize,
                                            fftshift)
    B, T, C, H, W = matrix.shape
    midpoints = (H // 2, W // 2)
    shifty, shiftx = shifts_from_circular_cc(cc, midpoints)
    return shifty, shiftx, cc

def transformation_matrix_2D(center: Tuple[int, int], angle: float = 0, scale: float = 1, shifts: Tuple[float, float] = (0, 0)) -> Tensor:
    alpha = scale * math.cos(angle)
    beta = scale * math.sin(angle)
    #Issue
    affine_matrix = Tensor([[alpha, beta, (1-alpha)*center[1] - beta * center[0]],
                   [-beta, alpha, beta*center[1]+(1-alpha)*center[0]]])
    affine_matrix[1, 2] += float(shifts[0])  # shift_y
    affine_matrix[0, 2] += float(shifts[1])  # shift_x
    transformation_matrix = torch.zeros((3, 3))
    transformation_matrix[2, 2] = 1
    transformation_matrix[0:2, :] = affine_matrix
    return transformation_matrix

def param2theta(transformation: Tensor, H: int, W: int):
    param = torch.linalg.inv(transformation)
    theta = torch.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * H / W
    theta[0, 2] = param[0, 2] * 2 / W + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = param[1, 0] * W / H
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / H + theta[1, 0] + theta[1, 1] - 1
    return theta
def affine_transformation_matrix(dims: Tuple[int, int], angle: float = 0, scale: float = 1, shifts: Tuple[float, float] = (0, 0)) -> Tensor:
    H, W = dims
    invertable_transformation = transformation_matrix_2D((H/2, W/2), angle, scale, shifts)  #TODO: dudy changed from H//2,W//2
    transformation = param2theta(invertable_transformation, H, W)
    return transformation


def _shift_matrix_subpixel_fft(matrix: torch.Tensor, shift_H: Tensor, shift_W: Tensor, matrix_fft: Tensor = None, warp_method: str = 'fft') -> torch.Tensor:
    """
    :param matrix: 5D matrix
    :param shift_H: either singleton or of length T
    :param shift_W: either singleton or of length T
    :param matrix_fft: fft of matrix, if precalculated
    :return: subpixel shifted matrix
    """
    ky, kx = calculate_meshgrids(matrix)
    #shift_W, shift_H = expand_shifts(matrix, shift_H, shift_W)
    ### Displace input image: ###
    displacement_matrix = torch.exp(-(1j * 2 * torch.pi * ky * shift_H + 1j * 2 * torch.pi * kx * shift_W)).to(matrix.device)
    fft_image = faster_fft(matrix, matrix_fft, dim=[-1, -2])
    fft_image_displaced = fft_image * displacement_matrix
    original_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1, -2]).abs()
    return original_image_displaced


def identity_transforms(N: int, angles: Tensor, scales: Tensor , shifts: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
    # returns identity angles, scale, shifts when they are not already defined
    if angles is None:
        angles = torch.zeros(N)
    if scales is None:
        scales = torch.ones(N)
    if shifts is None:
        shifts = (Tensor([0 for _ in range(N)]), Tensor([0 for _ in range(N)]))
    return angles, scales, shifts


def batch_affine_matrices(dims: Tuple[int, int], N: int = 1, angles: Tensor = None, scales: Tensor = None, shifts: Tuple[Tensor, Tensor] = None) -> Tensor:
    angles, scales, shifts = identity_transforms(N, angles, scales, shifts)
    affine_matrices = torch.zeros((N, 2, 3))
    for i in range(N):
        affine_matrices[i] = affine_transformation_matrix(dims, angles[i], scales[i], (shifts[0][i], shifts[1][i]))
    return affine_matrices



def extend_vector_length_n(vector: Tensor, N: int) -> Tensor:
    vector_length = len(vector)
    if vector_length == 1:
        return torch.ones(N) * vector
    elif int(N) == vector_length:
        return vector
    elif int(N) % vector_length == 0:
        return vector.repeat(int(N) // vector_length)
    else:
        raise RuntimeError("Cannot extend Tensor not factor length of N")


def _shift_matrix_subpixel_interpolated(matrix: torch.Tensor, shift_H: Tensor, shift_W: Tensor, warp_method='bilinear') -> torch.Tensor:
    B, T, C, H, W = matrix.shape
    ##Question: rounded to int??
    N = B*T
    #TODO: dudy: don't!!! reinstantiate the grid all over every time!!! use a layer!!!!!
    affine_matrices = batch_affine_matrices((H,W), N, shifts=(extend_vector_length_n(shift_H, N), extend_vector_length_n(shift_W, N)))
    output_grid = torch.nn.functional.affine_grid(affine_matrices,
                                                  torch.Size((N, C, H, W))).to(matrix.device)
    matrix = matrix.reshape((B * T, C, H, W))  # grid sample can only handle 4D. #TODO: dudy: use view instead of reshape
    output_tensor = torch.nn.functional.grid_sample(matrix, output_grid, mode=warp_method)
    return output_tensor.reshape((B, T, C, H, W))

class MatrixOrigami:
    # expands matrix to 5 dimensions, then folds it back to original size at the end
    def __init__(self, matrix: Tensor):
        self.matrix = matrix
        self.original_dims = matrix.shape

    def expand_matrix(self, num_dims: int = 5) -> Tensor:
        # makes matrix requisite number of dimensions
        return fix_dimensionality(matrix=self.matrix, num_dims=num_dims)

    def expand_other_matrix(self, matrix: Tensor, num_dims = 5) -> Tensor:
        return fix_dimensionality(matrix, num_dims=num_dims)

    def squeeze_to_original_dims(self, matrix: Tensor) -> Tensor:
        # TODO Elisheva added this loop instead of squeeze()
        for dim in matrix.shape:
            if dim != 1:
                break
            matrix = matrix.squeeze(0)
        for i in range(len(self.original_dims) - len(matrix.shape)):
             matrix = matrix.unsqueeze(0)
        return matrix


def extend_tensor_length_N(vector: Tensor, N: int) -> Tensor:
    """
    :param vector: 1D vector of size 1, or size N
    :param N: length the vector is, or will be extended to
    :return: vector of size N containing the element/s of input vector
    """
    if len(vector) == N:
        return vector
    else:
        return torch.ones(N) * vector


def equal_size_tensors(a: Tensor, b: Tensor) -> bool:
    # returns if tensors are of equal size
    if true_dimensionality(a) != true_dimensionality(b):
        return False
    elif len(a.shape) > len(b.shape):
        return compare_unequal_dimensionality_tensors(a, b)
    else: # need to know which tensor is greater dimensionality before comparing unequal dimensions
        return compare_unequal_dimensionality_tensors(b, a)

def format_shift(matrix: Tensor, shift: torch.Tensor) -> Tensor:
    """
    :param matrix: matrix shift will be applied to
    :param shift: either singleton or length N 1D vector. Exception will be raised if these do not hold
    :return: singleton vector or 1D vector of length time domain of the input matrix
    """
    if true_dimensionality(shift) > 0: # has more than one element. Shifts must be same length as time dimension, matrix must have 4+ dimensions
        raise_if_not(len(torch.nonzero(shift.to(torch.int64) - shift)) == 0, message="Shifts are not integer")
        raise_if_not(true_dimensionality(shift) == 1, message="Shifts must be 1D vector or int") # shouldn't be 2D
        raise_if_not(true_dimensionality(matrix) >= 4, message="Shifts has larger dimensionality than input matrix") # matrix has time dimension
        raise_if_not(matrix.shape[-4] == shift.shape[0], message="Shifts not same length as time dimension") # has to have same length as time
        return shift
    else:
        raise_if(len(shift) == 0)
        return shift



def dimension_N(matrix: Tensor, dimension: int) -> int:
    # returns size of dimension -dimension, or 1 if it doesn't exist
    if len(matrix.shape) >= dimension:
        return matrix.shape[-dimension]
    else:
        return 1



def format_subpixel_shift_params(matrix: Union[torch.Tensor, np.array, tuple, list], shift_H: Union[torch.Tensor, list, Tuple, float, int],
                                 shift_W: Union[torch.Tensor, list, Tuple, float, int], matrix_FFT=None, warp_method='bilinear')\
                                    -> Tuple[Tensor, Tensor, Tensor, Tensor, str]:
    type_check_parameters([(matrix, (torch.Tensor, np.ndarray, tuple, list)), (shift_H, (Tensor, np.ndarray, list, tuple, int, float)),
                           (shift_W, (Tensor, list, tuple, int, float, np.ndarray)), (warp_method, str), (matrix_FFT, (type(None), Tensor))])
    validate_warp_method(warp_method)
    matrix = construct_tensor(matrix)
    if matrix_FFT is not None:
        raise_if_not(warp_method=='fft', message="FFT should only be passed when using the FFT warp method")
        raise_if_not(equal_size_tensors(matrix_FFT, matrix))
    shift_H = construct_tensor(shift_H).to(matrix.device)
    shift_W = construct_tensor(shift_W).to(matrix.device)
    shift_H = format_shift(matrix, shift_H)
    shift_W = format_shift(matrix, shift_W)
    if shift_H.shape[0] > 1 or shift_W.shape[0] > 1:
        time_dimension_length = dimension_N(matrix, 4)
        shift_H = extend_tensor_length_N(shift_H, time_dimension_length)
        shift_W = extend_tensor_length_N(shift_W, time_dimension_length)
    return matrix, shift_H, shift_W, matrix_FFT, warp_method


class InvalidMethodError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)

def pick_method(functions: Dict[str, Callable], method: str, *args, **kwargs): # can return anything
    # pass functions in dictionary, and args in args and kwargs. Saves many lines of code
    # prune args of Null arguments
    """
    for s in args:
        print(type(s))
    for s in kwargs:
        print(f"{s}: {kwargs[s]}")
    print("+++++++++++++++++++++++++")
    """
    args = [s_arg for s_arg in args if s_arg is not None]  # s_arg = single_arg
    kwargs = {arg_key: kwargs[arg_key] for arg_key in kwargs if kwargs[arg_key] is not None}
    """
    for s in args:
        print(type(s))
    for s in kwargs:
        print(f"{s}: {kwargs[s]}")
    print("\n\n\n\n\n********************************************\n\n\n\n\n")
    """

    if method in functions.keys():
        return functions[method](*args, **kwargs)
    else:
        raise InvalidMethodError("Given method not valid")



def shift_matrix_subpixel(matrix: Union[torch.Tensor, np.array, tuple, list],
                          shift_H: Union[torch.Tensor, list, tuple, float, int],
                          shift_W: Union[torch.Tensor, list, tuple, float, int],
                          matrix_FFT=None, warp_method='bilinear') -> torch.Tensor:
    """Performs subpixel shift on given matrix. Consider pixels over the side of frame to have undefined behavior.

    :param matrix: 2-5D matrix of form (B,T,C,H,W) to be shifted
    :param shift_H: singleton or length T dimension vector of vertical shift/s. Shifts downwards
    :param shift_W: singleton or length T dimension vector of horizontal shift/s. Shifts rightwards
    :param matrix_FFT: For use only when using 'fft' warp method. In case the fft of the matrix has already been calculated, it can be passed to the function to improve performance. FFT must be over dimensions -2, -1
    :param warp_method: method used to warp the matrix when shifting. Default: 'bilinear'. Options: 'bilinear', 'bicubic', 'nearest' and 'fft'
    :return: matrix shifted according to given shifts
    """
    # possible_methods/pick methods allows for checking for improper methods as well as a more concise selection code
    matrix, shift_H, shift_W, matrix_FFT, warp_method = format_subpixel_shift_params(matrix, shift_H, shift_W, matrix_FFT, warp_method)
    dimensions_memory = MatrixOrigami(matrix)
    expanded_matrix = dimensions_memory.expand_matrix()
    possible_methods = {'fft': _shift_matrix_subpixel_fft, 'bilinear': _shift_matrix_subpixel_interpolated,
                        'bicubic': _shift_matrix_subpixel_interpolated, 'nearest': _shift_matrix_subpixel_interpolated}
    shifted_matrix = pick_method(possible_methods, warp_method, expanded_matrix, shift_H, shift_W, matrix_FFT, warp_method=warp_method)
    return dimensions_memory.squeeze_to_original_dims(shifted_matrix)



def align_circular_cc(matrix: Tensor, reference_matrix: Tensor, matrix_fft: Tensor,
                      reference_tensor_fft: Tensor,
                      normalize: bool=False,
                      crop_warped_matrix: bool=False,
                      warp_method: str='bilinear',
                      fftshift: bool=False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # calculate shifts
    shifts_h, shifts_w, cc = classic_circular_cc_shifts_calc(matrix, reference_matrix, matrix_fft, reference_tensor_fft, normalize, fftshift)
    # warp matrix
    warped_matrix = shift_matrix_subpixel(matrix, -shifts_h, -shifts_w, matrix_FFT=None, warp_method=warp_method)

    if crop_warped_matrix:
        B, T, C, H, W = warped_matrix.shape
        max_shift_h = shifts_h.max()
        max_shift_w = shifts_w.max()
        new_h = H - (max_shift_h.abs().int().item() * 2 + 5)  # TODO make sure no need to also add .cpu().numpy() instead of .item()
        new_w = W - (max_shift_w.abs().int().item() * 2 + 5)  # TODO also make safety margins not rigid (5 is arbitrary)
        warped_matrix = non_interpolated_center_crop(warped_matrix, new_h, new_w)

    return warped_matrix, shifts_h, shifts_w, cc



def align_to_reference_frame_circular_cc(matrix: Union[torch.Tensor, np.array, tuple, list],
                                         reference_matrix: Union[torch.Tensor, np.array, list, tuple],
                                         matrix_fft: Union[torch.Tensor, np.array, tuple, list] = None,
                                         reference_matrix_fft: Union[torch.Tensor, np.array, list, tuple] = None,
                                         normalize_over_matrix: bool = False,
                                         warp_method: str = 'bilinear',
                                         crop_warped_matrix: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Validate parameters
    matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix, warp_method, crop_warped_matrix, _ = \
        format_align_to_reference_frame_circular_cc_params(matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix,
                                                                     warp_method, crop_warped_matrix, False)
    # Memorize dimensions
    dimensions_memory = NoMemoryMatrixOrigami(matrix)
    matrix = dimensions_memory.expand_matrix(matrix)
    reference_matrix = dimensions_memory.expand_matrix(reference_matrix)
    if matrix_fft is not None:
        matrix_fft = dimensions_memory.expand_matrix(matrix_fft)
    if reference_matrix_fft is not None:
        reference_matrix_fft = dimensions_memory.expand_matrix(reference_matrix_fft)
    # Warp matrix
    warped_matrix, shifts_h, shifts_w, cc = align_circular_cc(
        matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix, crop_warped_matrix, warp_method, False)
    # Return to original dimensionality
    warped_matrix = dimensions_memory.squeeze_to_original_dims(warped_matrix)
    cc = dimensions_memory.squeeze_to_original_dims(cc)

    return warped_matrix, shifts_h, shifts_w, cc


def faster_fft(matrix: Tensor, matrix_fft: Tensor, dim=[-2,-1]) -> Tensor:
    if matrix_fft is None:
        return torch.fft.fftn(matrix, dim=dim)
    else:
        return matrix_fft



def _shift_matrix_subpixel_fft_batch_with_channels(matrix: torch.Tensor, shift_H, shift_W, matrix_fft=None,
                                                   warp_method: str = 'fft') -> torch.Tensor:
    """
    matrix of t, c, h, w, shifts of size t

    :param matrix: 5D matrix
    :param shift_H: either singleton or of length T
    :param shift_W: either singleton or of length T
    :param matrix_fft: fft of matrix, if precalculated
    :return: subpixel shifted matrix
    """
    t, c, h, w = matrix.shape
    if isinstance(shift_W, float):
        shift_W = torch.tensor(shift_W).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(t, 1, 1).to(matrix.device)
    if isinstance(shift_H, float):
        shift_H = torch.tensor(shift_H).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(t, 1, 1).to(matrix.device)
    if shift_W.dim() == 1:
        shift_W = shift_W.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(matrix.device)
    if shift_H.dim() == 1:
        shift_H = shift_H.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(matrix.device)

    # print(shift_W.shape)
    # print(shift_H.shape)

    ky, kx = calculate_meshgrids(matrix)
    # shift_W, shift_H = expand_shifts(matrix, shift_H, shift_W)
    ### Displace input image: ###
    displacement_matrix = torch.exp(-(
                1j * 2 * torch.pi * ky.repeat(t, c, 1, 1).to(matrix.device) * shift_H + 1j * 2 * torch.pi * kx.repeat(t,
                                                                                                                      c,
                                                                                                                      1,
                                                                                                                      1).to(
            matrix.device) * shift_W)).to(matrix.device)
    fft_image = faster_fft(matrix, matrix_fft, dim=[-1, -2])
    fft_image_displaced = fft_image * displacement_matrix
    original_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1, -2]).abs()

    # imshow_torch(original_image_displaced[0])
    # imshow_torch(original_image_displaced[1])
    # imshow_torch(original_image_displaced[2])
    # imshow_torch(original_image_displaced[3])
    # imshow_torch(original_image_displaced[4])

    return original_image_displaced


def extract_frames_from_video(video_path, output_dir):
    """
    Extract frames from a video and save them to a specified directory.

    Args:
    - video_path (str): Path to the video file.
    - output_dir (str): Directory to save the extracted frames.

    Returns:
    - None
    """
    # Create the tweets_per_accounts directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        frame_filename = os.path.join(output_dir, f"frame_{count:04d}.png")
        cv2.imwrite(frame_filename, image)
        success, image = vidcap.read()
        count += 1


import os

video_file_extensions = (
'.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.89', '.aaf', '.aec', '.aep', '.aepx',
'.aet', '.aetx', '.ajp', '.ale', '.am', '.amc', '.amv', '.amx', '.anim', '.aqt', '.arcut', '.arf', '.asf', '.asx', '.avb',
'.avc', '.avd', '.avi', '.avp', '.avs', '.avs', '.avv', '.axm', '.bdm', '.bdmv', '.bdt2', '.bdt3', '.bik', '.bin', '.bix',
'.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj', '.camrec', '.camv', '.ced', '.cel', '.cine', '.cip',
'.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst', '.cvc', '.cx3', '.d2v', '.d3v', '.dat', '.dav', '.dce',
'.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx', '.dmb', '.dmsd', '.dmsd3d', '.dmsm', '.dmsm3d', '.dmss',
'.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi', '.dv4', '.dvdmedia', '.dvr', '.dvr-ms', '.dvx', '.dxr',
'.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p', '.f4v', '.fbr', '.fbr', '.fbz', '.fcp', '.fcproject',
'.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl', '.gom', '.grasp', '.gts', '.gvi', '.gvp', '.h264', '.hdmov',
'.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf', '.ism', '.ismc', '.ismv', '.iva', '.ivf', '.ivr', '.ivs',
'.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn', '.lrec', '.lsf', '.lsx', '.m15', '.m1pg', '.m1v', '.m21',
'.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u', '.m4v', '.m75', '.mani', '.meta', '.mgv', '.mj2', '.mjp',
'.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
'.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2', '.mpgindex', '.mpl',
'.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd', '.mse', '.msh', '.mswmm', '.mts', '.mtv', '.mvb', '.mvc',
'.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv', '.mys', '.ncor', '.nsv', '.nut', '.nuv', '.nvc', '.ogm',
'.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi', '.photoshow', '.piv', '.pjs', '.playlist', '.plproj',
'.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj', '.prtl', '.psb', '.psh', '.pssd', '.pva', '.pvr', '.pxv',
'.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd', '.rcproject', '.rdb', '.rec', '.rm', '.rmd', '.rmd',
'.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts', '.rts', '.rum', '.rv', '.rvid', '.rvl', '.sbk', '.sbt',
'.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj', '.seq', '.sfd', '.sfvidcap', '.siv', '.smi', '.smi',
'.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf', '.ssm', '.stl', '.str', '.stx', '.svi', '.swf', '.swi',
'.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp', '.tp0', '.tpd', '.tpr', '.trp', '.ts', '.tsp', '.ttxt',
'.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo', '.vdr', '.vdx', '.veg','.vem', '.vep', '.vf', '.vft',
'.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv', '.vivo', '.vlab', '.vob', '.vp3', '.vp6', '.vp7', '.vpj',
'.vro', '.vs4', '.vse', '.vsp', '.w32', '.wcp', '.webm', '.wlmp', '.wm', '.wmd', '.wmmp', '.wmv', '.wmx', '.wot', '.wp3',
'.wpl', '.wtv', '.wve', '.wvx', '.xej', '.xel', '.xesc', '.xfl', '.xlmv', '.xmv', '.xvid', '.y4m', '.yog', '.yuv', '.zeg',
'.zm1', '.zm2', '.zm3', '.zmv'  )

def is_video_file(filename):
    return os.path.splitext(filename)[-1] in video_file_extensions




def prepare_input_source(input_path):
    """
    Prepare the input source for processing.

    Args:
    - input_path (str): Path to the video file or folder.

    Returns:
    - str: Path to the directory containing the frames.
    """

    # If it's a video, extract frames to a temp directory
    if is_video_file(input_path):
        temp_dir = "temp_frames_dir"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        extract_frames_from_video(input_path, temp_dir)
        return temp_dir

    # If it's a directory, return the input path as is
    else:
        return input_path


def get_avg_reference_given_shifts(video, shifts):
    sx, sy = shifts
    warped = _shift_matrix_subpixel_fft_batch_with_channels(video, -sy.to(video.device), -sx.to(video.device))
    return warped.mean(0)

def video_numpy_array_to_video(input_tensor, video_name='my_movie.avi', FPS=25.0):
    T, H, W, C = input_tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
    video_writer = cv2.VideoWriter(video_name, fourcc, FPS, (W, H))
    for frame_counter in np.arange(T):
        current_frame = input_tensor[frame_counter]
        video_writer.write(current_frame)
    video_writer.release()

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


def video_torch_array_to_video(input_tensor, video_name='my_movie.avi', FPS=25.0):
    T, C, H, W = input_tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
    video_writer = cv2.VideoWriter(video_name, fourcc, FPS, (W, H))
    for frame_counter in np.arange(T):
        current_frame = input_tensor[frame_counter]
        current_frame = current_frame.permute([1,2,0]).numpy()
        current_frame = BW2RGB(current_frame)
        current_frame = (current_frame * 255).astype(np.uint8)
        video_writer.write(current_frame)
    video_writer.release()



def clean_1_frame(estimator_video, averaged_video, index, max_window=None):
    """
    warp all video towrds 1 frame and average the frame.
    Args:
        estimator_video:
        averaged_video:
        index:

    Returns:

    """
    if max_window is None:
        _, sy, sx, _ = align_to_reference_frame_circular_cc(estimator_video.unsqueeze(0),
                                                            estimator_video[index:index + 1].unsqueeze(0).repeat(1,
                                                                                                                 estimator_video.shape[
                                                                                                                     0],
                                                                                                                 1, 1,
                                                                                                                 1))
        avg_ref = get_avg_reference_given_shifts(averaged_video, [sx, sy])
        return avg_ref, sx, sy
    if index + max_window < estimator_video.shape[0]:
        _, sy, sx, _ = align_to_reference_frame_circular_cc(estimator_video[index:index + max_window].unsqueeze(0),
                                                            estimator_video[index:index + 1].unsqueeze(0).repeat(1,
                                                                                                                 max_window,
                                                                                                                 1, 1,
                                                                                                                 1))
        avg_ref = get_avg_reference_given_shifts(averaged_video[index:index + max_window], [sx, sy])
        return avg_ref, sx, sy
    else:
        _, sy, sx, _ = align_to_reference_frame_circular_cc(estimator_video[index - max_window:index].unsqueeze(0),
                                                            estimator_video[index:index + 1].unsqueeze(0).repeat(1,
                                                                                                                 max_window,
                                                                                                                 1, 1,
                                                                                                                 1))
        avg_ref = get_avg_reference_given_shifts(averaged_video[index - max_window:index], [sx, sy])
        return avg_ref, sx, sy


def shift_2_center(video):
    """
    warp all video towrds center frame
    """
    middle_frame = (video.shape[0] // 2 - 1)
    _, sy, sx, _ = align_to_reference_frame_circular_cc(video.unsqueeze(0),
                                                        video[middle_frame:middle_frame + 1].unsqueeze(0).repeat(1,
                                                                                                                 video.shape[
                                                                                                                     0],
                                                                                                                 1, 1,
                                                                                                                 1))
    video = _shift_matrix_subpixel_fft_batch_with_channels(video, -sy, -sx)
    return video


def stabilize_ccc_video_no_avg(folder_path, max_frames=None, video_path_to_save="output_video_ecc_no_avg.mp4"):
    """

    Args:
        folder_path: path to video frames(png)
        max_window: window size to average upon, larger will be slower
        max_frames: stabilize first max_frames frames
        loop: iteration count of averaging
    Returns:

    """
    #### generate data
    # img = read_image_torch(default_image_filename_to_load1) # get single channel of demo image
    file_names = sorted(glob(os.path.join(folder_path, "*.png")))
    video = torch.cat([read_image_torch(name) for name in file_names])[:max_frames]
    averaged_video = shift_2_center(video)
    # save_video_torch(video/255, "/raid/yoav/temp_garbage/ecc/original_no_avg")
    # save_video_torch(averaged_video/255, "/raid/yoav/temp_garbage/ecc/averaged_no_avg")
    video_torch_array_to_video(averaged_video / 255, video_name=video_path_to_save)
    return averaged_video


def stabilize_ccc_video(folder_path, max_window=5, max_frames=None, loop=1, video_path_to_save="output_video_ecc_avg.mp4"):
    """

    Args:
        folder_path: path to video frames(png)
        max_window: window size to average upon, larger will be slower
        max_frames: stabilize first max_frames frames
        loop: iteration count of averaging

    Returns:

    """
    file_names = sorted(glob(os.path.join(folder_path, "*.png")))
    video = torch.cat([read_image_torch(name) for name in file_names])[:max_frames]

    averaged_video = video.clone()
    for i in range(loop):
        previous_avg_video = averaged_video.clone()
        for j in range(averaged_video.shape[0]):
            print(i, j)
            averaged_video[j], _, _ = clean_1_frame(previous_avg_video, video, j, max_window)

    averaged_video = shift_2_center(averaged_video)
    video_torch_array_to_video(averaged_video / 255, video_name=video_path_to_save)
    return averaged_video


def stabilize_ccc_image(folder_path, image_output_name="ecc.png",output_dir_path="tweets_per_accounts", max_frames=None, loop=1):
    """

    Args:
        folder_path: path to video frames(png)
        max_window: window size to average upon, larger will be slower
        max_frames: stabilize first max_frames frames
        loop: iteration count of averaging

    Returns:

    """
    #### generate data
    # img = read_image_torch(default_image_filename_to_load1) # get single channel of demo image
    file_names = sorted(glob(os.path.join(folder_path, "*.png")))
    video = torch.cat([read_image_torch(name) for name in file_names])[:max_frames]

    averaged_video = video.clone()
    for i in range(loop):
        # imshow_torch(video[i]/255)
        ###### estimate shits on avg video
        # for each index average all tensor
        previous_avg_video = averaged_video.clone()
        for j in range(averaged_video.shape[0]):
            print(i, j)
            averaged_video[j], _, _ = clean_1_frame(previous_avg_video, video, j)

    averaged_video = shift_2_center(video)
    save_image_torch(folder_path=output_dir_path, filename=image_output_name, torch_tensor=averaged_video[averaged_video.shape[0] // 2],
                     flag_convert_bgr2rgb=False, flag_scale_by_255=False, flag_save_figure=True)
    return averaged_video.mean(0)

#
if __name__ == '__main__':
    input_folder_path = "/home/nehoray/PycharmProjects/VideoImageEnhancement/data/dgx_data/blur_cars/white_car"
    input_video_path = "/home/nehoray/PycharmProjects/VideoImageEnhancement/data/dgx_data/shabak_clips/shabak_0/scene_5.mp4"
    output_path_to_save_video = "output"


    # a = stabilize_ccc_image(folder_path=frame_dir, max_frames=20, loop=1,output_dir_path=output_dir_path, image_output_name=output_image_name)
    stabilize_ccc_video_no_avg(folder_path=input_folder_path, video_path_to_save=output_path_to_save_video)

    # stabilize_ccc_image(folder_path=input_source, output_dir_path=full_path_to_output)
    #
    # stabilize_ccc_video(folder_path=input_source, video_path_to_save=full_path_to_output_specific)

