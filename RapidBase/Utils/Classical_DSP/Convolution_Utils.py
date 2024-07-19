from typing import Tuple, List
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import numpy as np
import cv2
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_get_5D, torch_get_3D, torch_get_2D, torch_get_4D
# from RapidBase.import_all import *

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Computes padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) >= 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(

    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = max(computed_tmp - 1, 0)
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding


def filter2D_torch(input: torch.Tensor,
                   kernel: torch.Tensor,
                   border_type: str = 'reflect',
                   normalized: bool = False,
                   flag_return_valid_part_only=True) -> torch.Tensor:
    r"""Convolve a tensor with a 2d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2D_torch(input, kernel)
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input image is not torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel is not torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3 and kernel.shape[0] != 1:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = _compute_padding([height, width])
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-2), input_pad.size(-1))

    # convolve the tensor with the kernel.
    if flag_return_valid_part_only:
        final_padding = 0
    else:
        final_padding = kernel.shape[-1]//2   #TODO: make 2D for generality
    output = F.conv2d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=final_padding, stride=1)
    B, C, H, W = output.shape
    residual_shape_1 = output.shape[-1] - input.shape[-1]
    residual_shape_2 = output.shape[-2] - input.shape[-2]
    if flag_return_valid_part_only:
        output = output[:, :, 0:H - residual_shape_2, 0:W - residual_shape_1]
        return output.view(b, c, h, w)
    else:
        return output.view(b,c,H,W)




def filter3D_torch(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'replicate',
             normalized: bool = False) -> torch.Tensor:
    r"""Convolve a tensor with a 3d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, D, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kD, kH, kW)`  or :math:`(B, kD, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``,
          ``'replicate'`` or ``'circular'``. Default: ``'replicate'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, D, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 5., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]]
        ... ]]])
        >>> kernel = torch.ones(1, 3, 3, 3)
        >>> filter3D_torch(input, kernel)
        tensor([[[[[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]]]]])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input border_type is not torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input border_type is not torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 4 and kernel.shape[0] != 1:
        raise ValueError("Invalid kernel shape, we expect 1xDxHxW. Got: {}"
                         .format(kernel.shape))

    # prepare kernel
    b, c, d, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel = normalize_kernel2d(tmp_kernel.view(
            bk, dk, hk * wk)).view_as(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1, -1)

    # pad the input tensor
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape: List[int] = _compute_padding([depth, height, width])
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-3), input_pad.size(-2), input_pad.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv3d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    B,C,D,H,W = output.shape
    residual_shape_1 = output.shape[-1] - input.shape[-1]
    residual_shape_2 = output.shape[-2] - input.shape[-2]
    residual_shape_3 = output.shape[-3] - input.shape[-3]
    output = output[:,:,0:D-residual_shape_3,0:H-residual_shape_2,0:W-residual_shape_1]

    return output.view(b, c, d, h, w)


# class convn_layer_torch(nn.Module):
#     """
#     Apply gaussian smoothing on a
#     1d, 2d or 3d tensor. Filtering is performed seperately for each channel
#     in the input using a depthwise convolution.
#     Arguments:
#         kernel_size (int, sequence): Size of the gaussian kernel.
#         sigma (float, sequence): Standard deviation of the gaussian kernel.
#         dim (int, optional): The number of dimensions of the data.
#             Default value is 2 (spatial).
#     """
#     def __init__(self, kernel, dim=2):
#         super(convn_layer_torch, self).__init__()
#
#         # Reshape to depthwise convolutional weight
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
#
#         self.register_buffer('weight', kernel)
#         self.groups = channels
#
#         if dim == 1:
#             self.conv = F.conv1d
#         elif dim == 2:
#             self.conv = F.conv2d
#         elif dim == 3:
#             self.conv = F.conv3d
#         else:
#             raise RuntimeError(
#                 'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
#             )
#
#         self.padding = []
#         for i in kernel_size:
#             self.padding.append(i//2)
#             self.padding.append(i//2)
#
#     def forward(self, input):
#         """
#         Apply gaussian filter to input.
#         Arguments:
#             input (torch.Tensor): Input to apply gaussian filter on.
#         Returns:
#             filtered (torch.Tensor): Filtered output.
#         """
#         return self.conv(F.pad(input, self.padding), weight=self.weight, groups=self.groups)

import matplotlib.pyplot as plt
class convn_layer_Filter3D_torch(nn.Module):
    """
    1D convolution of a n-dimensional tensor over wanted dimension
    """
    def __init__(self):
        super(convn_layer_Filter3D_torch, self).__init__()

    def forward(self, input_tensor, kernel, dim):
        """
        Arguments:
            input (torch.Tensor): Input to apply filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        ### Get input tensor to proper size (B,T,C,H,W)/(B,C,H,W,D): ###
        original_shape = input_tensor.shape
        input_tensor = torch_get_5D(input_tensor)

        ### Get explicit dimension index: ###
        if dim >= 0:
            dim = dim - len(original_shape)
        dim = len(input_tensor.shape) - np.abs(dim)

        ### Filter: ###
        if dim>=2 and dim<=4:
            ### Get kernel to be 4D: (1,Kx,Ky,Kz) or (B,Kx,Ky,Kz) ###
            if dim == 4:
                kernel_final = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif dim == 3:
                kernel_final = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            elif dim == 2:
                kernel_final = kernel.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            output_tensor = filter3D_torch(input_tensor, kernel_final)
        else:
            if dim == 1:
                input_tensor = input_tensor.permute([0, 2, 3, 4, 1])
                kernel_final = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                output_tensor = filter3D_torch(input_tensor, kernel_final)
                output_tensor = output_tensor.permute([0, 4, 1, 2, 3])
            elif dim == 0:
                input_tensor = input_tensor.permute([1, 2, 3, 4, 0])
                kernel_final = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                output_tensor = filter3D_torch(input_tensor, kernel_final)
                output_tensor = output_tensor.permute([4, 0, 1, 2, 3])

        if len(original_shape) == 3:
            output_tensor = output_tensor[0,0]
        if len(original_shape) == 4:
            output_tensor = output_tensor[0]

        # plt.figure()
        # plt.imshow(output_tensor[0,0].permute([1,2,0]).cpu().numpy())
        # plt.figure()
        # plt.imshow(input_tensor[0,0].permute([1,2,0]).cpu().numpy())
        # plt.show()
        return output_tensor

class convn_layer_Filter2D_torch(nn.Module):
    """
    1D convolution of a n-dimensional tensor over wanted dimension
    """
    def __init__(self):
        super(convn_layer_Filter2D_torch, self).__init__()

    def forward(self, input_tensor, kernel, dim):
        """
        Arguments:
            input (torch.Tensor): Input to apply filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        ### Get input tensor to proper size: ###
        original_shape = input_tensor.shape
        input_tensor = torch_get_4D(input_tensor)
        B,C,H,W = input_tensor.shape

        ### Get explicit dimension index: ###
        if dim >= 0:
            dim = dim - len(original_shape)
        dim = len(input_tensor.shape) - np.abs(dim)

        ### Filter: ###
        if dim>=2 and dim<=3:
            ### Get kernel to be 4D: (1,Kx,Ky) or (B,Kx,Ky) ###
            if dim == 3:
                kernel_final = kernel.unsqueeze(0).unsqueeze(0)
            elif dim == 2:
                kernel_final = kernel.unsqueeze(0).unsqueeze(-1)
            output_tensor = filter2D_torch(input_tensor, kernel_final)
        else:
            if dim == 1:
                input_tensor = input_tensor.permute([0,2,3,1])
                kernel_final = kernel.unsqueeze(0).unsqueeze(0)
                output_tensor = filter2D_torch(input_tensor, kernel_final)
                output_tensor = output_tensor.permute([0,3,1,2])
            elif dim == 0:
                input_tensor = input_tensor.permute([1, 2, 3, 0])
                kernel_final = kernel.unsqueeze(0).unsqueeze(0)
                output_tensor = filter2D_torch(input_tensor, kernel_final)
                output_tensor = output_tensor.permute([3, 0, 1, 2])


        if len(original_shape) == 2:
            output_tensor = output_tensor[0,0]
        if len(original_shape) == 3:
            output_tensor = output_tensor[0]

        # plt.figure()
        # plt.imshow(output_tensor[0].permute([1,2,0]).cpu().numpy())
        # plt.figure()
        # plt.imshow(input_tensor[0].permute([1,2,0]).cpu().numpy())
        # plt.show()
        return output_tensor


class convn_layer_torch(nn.Module):
    """
    1D convolution of a n-dimensional tensor over wanted dimension
    """

    def __init__(self):
        super(convn_layer_torch, self).__init__()
        self.convn_layer_Filter2D_torch = convn_layer_Filter2D_torch()
        self.convn_layer_Filter3D_torch = convn_layer_Filter3D_torch()

    def forward(self, input_tensor, kernel, dim):
        """
        Arguments:
            input (torch.Tensor): Input to apply filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        #TODO: allow the kernel input to be a list or a tuple!!!!!

        ### Choose which function to use according to input size and dimension
        # (assuming timing(Filter1D)<timing(Filter2D)<timing(Filter3D) in any case): ###
        original_shape = input_tensor.shape
        if dim < 0:
            dim = len(original_shape) - np.abs(dim)

        if len(original_shape) == 1:
            output_tensor = self.convn_layer_Filter2D_torch.forward(input_tensor, kernel=kernel, dim=dim)
        elif len(original_shape) == 2:
            output_tensor = self.convn_layer_Filter2D_torch.forward(input_tensor, kernel=kernel, dim=dim)
        elif len(original_shape) == 3:
            if dim == 1 or dim == 2:
                output_tensor = self.convn_layer_Filter2D_torch.forward(input_tensor, kernel=kernel, dim=dim)
            elif dim == 0:
                output_tensor = self.convn_layer_Filter3D_torch.forward(input_tensor, kernel=kernel, dim=dim)
        elif len(original_shape) == 4:
            if dim == 2 or dim == 3:
                output_tensor = self.convn_layer_Filter2D_torch.forward(input_tensor, kernel=kernel, dim=dim)
            elif dim == 0 or dim == 1:
                output_tensor = self.convn_layer_Filter3D_torch.forward(input_tensor, kernel=kernel, dim=dim)
        elif len(original_shape) == 5:
            output_tensor = self.convn_layer_Filter3D_torch.forward(input_tensor, kernel=kernel, dim=dim)

        # plt.figure()
        # plt.imshow(output_tensor[0].permute([1,2,0]).cpu().numpy())
        # plt.figure()
        # plt.imshow(input_tensor[0].permute([1,2,0]).cpu().numpy())
        # plt.show()
        return output_tensor

def convn_torch(input_tensor, kernel, dim):
    original_shape = input_tensor.shape
    if dim < 0:
        dim = len(original_shape) - np.abs(dim)

    #TODO: turn convn_layer_Filter2D_torch into function!!!!!
    if len(original_shape) == 1:
        output_tensor = convn_layer_Filter2D_torch().forward(input_tensor, kernel=kernel, dim=dim)
    elif len(original_shape) == 2:
        output_tensor = convn_layer_Filter2D_torch().forward(input_tensor, kernel=kernel, dim=dim)
    elif len(original_shape) == 3:
        if dim == 1 or dim == 2:
            output_tensor = convn_layer_Filter2D_torch().forward(input_tensor, kernel=kernel, dim=dim)
        elif dim == 0:
            output_tensor = convn_layer_Filter3D_torch().forward(input_tensor, kernel=kernel, dim=dim)
    elif len(original_shape) == 4:
        if dim == 2 or dim == 3:
            output_tensor = convn_layer_Filter2D_torch().forward(input_tensor, kernel=kernel, dim=dim)
        elif dim == 0 or dim == 1:
            output_tensor = convn_layer_Filter3D_torch().forward(input_tensor, kernel=kernel, dim=dim)
    elif len(original_shape) == 5:
        output_tensor = convn_layer_Filter3D_torch().forward(input_tensor, kernel=kernel, dim=dim)

    return output_tensor


class convn_fft_layer_torch(nn.Module):
    """
    1D convolution of a n-dimensional tensor over wanted dimension
    """
    def __init__(self):
        super(convn_fft_layer_torch, self).__init__()

    def forward(self, input_tensor, kernel, dim, flag_return_real=True):
        """
        Arguments:
            input (torch.Tensor): Input to apply filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        original_shape = input_tensor.shape
        if dim < 0:
            dim = len(input_tensor.shape) - abs(dim)
        kernel_fft = torch.fft.fftn(kernel, s=input_tensor.shape[dim], dim=-1)
        number_of_dimensions_to_add_in_the_end = len(original_shape) - dim - 1
        number_of_dimensions_to_add_in_the_beginning = len(original_shape) - number_of_dimensions_to_add_in_the_end - 1
        for i in np.arange(number_of_dimensions_to_add_in_the_end):
            kernel_fft = kernel_fft.unsqueeze(-1)
        for i in np.arange(number_of_dimensions_to_add_in_the_beginning):
            kernel_fft = kernel_fft.unsqueeze(0)

        output_tensor = torch.fft.ifftn(torch.fft.fftn(input_tensor, dim=dim) * kernel_fft, dim=dim)
        if flag_return_real:
            return output_tensor.real
        else:
            return output_tensor


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




# ### TODO: temp ###
# default_image_filename_to_load1 = r'/home/mafat/DataSets/DIV2K/DIV2K/Flickr2K/000001.png'
# kernel = torch.ones((11))/11
# input_tensor = read_image_torch(default_image_filename_to_load1)/255
# # input_tensor = torch.randn(3,3,3,100,100)
# convn_layer = convn_layer_torch()
# convn_layer_2D = convn_layer_Filter2D_torch()
# convn_layer_3D = convn_layer_Filter3D_torch()
# convn_layer_fft = convn_fft_layer_torch()
# output_tensor_fft_space = convn_layer_fft(input_tensor, kernel, -1)
# output_tensor_real_space = convn_layer(input_tensor, kernel, -1)
#
# plt.figure(); plt.imshow(output_tensor_fft_space[0].permute([1,2,0]).cpu().numpy())
# plt.figure(); plt.imshow(output_tensor_real_space[0].permute([1,2,0]).cpu().numpy())
# plt.figure(); plt.imshow((output_tensor_real_space-output_tensor_fft_space)[0].permute([1,2,0]).cpu().numpy())
# plt.figure(); plt.imshow(input_tensor[0].permute([1,2,0]).cpu().numpy())



from scipy import signal

#TODO: get overlap-add object for 1D signals layer
#TODO: write a real-space filtering layer and not just fft (for short filters)
#TODO: get dudy's FIR builder into pytorch version
# from RapidBase.import_all import *
# from eylon_full_utils import *


from scipy import signal

# from RapidBase.import_all import *
from scipy import signal


def get_filter_1D(filter_name, filter_type, N, f_sampling, f_low_cutoff, f_high_cutoff, filter_parameter=True,
                  attenuation=0, beta=14):
    # Get window type wanted:
    # if filter_name == 'kaiser':
    #     window = signal.windows.kaiser(N + 1, beta, filter_parameter)
    # elif filter_name == 'hann' or filter_name == 'hanning':
    #     window = signal.windows.hann(N + 1, filter_parameter)
    # elif filter_name == 'hamming' or filter_name == "hamm":
    #     window = signal.windows.hamming(N + 1, filter_parameter)
    # elif filter_name == 'blackmanharris':
    #     window = signal.windows.blackmanharris(N + 1, filter_parameter)
    # elif filter_name == 'cheb':
    #     window = signal.windows.chebwin(N + 1, attenuation, filter_parameter)
    if filter_name == 'kaiser':
        filter_name = "kaiser"
    elif filter_name == 'hann' or filter_name == 'hanning':
        filter_name = "hann"
    elif filter_name == 'hamming' or filter_name == "hamm":
        filter_name = "hamming"
    elif filter_name == 'blackmanharris':
        filter_name = "blackmanharris"
    elif filter_name == 'cheb':
        filter_name = "chebwin"
    # Calculate the coefficients using the fir1 function
    if filter_name in ["chebwin", "kaiser"]:
        if filter_type in ['bandpass', 'stop']:
            coefficients = signal.firwin(N, [f_low_cutoff, f_high_cutoff], width=attenuation, window=filter_name,
                                         pass_zero="bandpass")
        elif filter_type in ['low', 'lowpass']:
            coefficients = signal.firwin(N, f_low_cutoff, window=filter_name, width=attenuation, pass_zero="lowpass")
        elif filter_type in ['high', 'highpass']:
            coefficients = signal.firwin(N, f_high_cutoff, window=filter_name, width=attenuation, pass_zero="highpass")
    else:
        if filter_type in ['bandpass', 'stop']:
            coefficients = signal.firwin(N, [f_low_cutoff, f_high_cutoff], fs=f_sampling, window=filter_name,
                                         pass_zero="bandpass")
        elif filter_type in ['low', 'lowpass']:
            coefficients = signal.firwin(N, f_low_cutoff, fs=f_sampling, window=filter_name, pass_zero="lowpass")
        elif filter_type in ['high', 'highpass']:
            coefficients = signal.firwin(N, f_high_cutoff, fs=f_sampling, window=filter_name, pass_zero="highpass")
    actual_filter = signal.dlti(coefficients, [1], dt=1)

    return actual_filter, coefficients


from RapidBase.Utils.Classical_DSP.FFT_utils import torch_fftshift
import torch



def get_filter_2D(filter_coefficients_1D):
    #TODO: still doesn't work like the matlab, need to check and compare where things go wrong
    ### Get the 1D filter Coefficients: ###
    n = int((len(filter_coefficients_1D) - 1) / 2)  # filter_coefficients_1D must be of odd length!!!!
    filter_coefficients_1D_shifted = torch_fftshift(filter_coefficients_1D.flip(-1)).flip(-1)
    a = torch.cat([filter_coefficients_1D_shifted[0:1], 2 * filter_coefficients_1D_shifted[1:n]], -1)

    ### Use Chebyshev polynomials to compute h: ###
    t = torch.ones(3, 3).to(filter_coefficients_1D.device)
    t[0, 0] = 1
    t[0, 1] = 2
    t[0, 2] = 1
    t[1, 0] = 2
    t[1, 1] = -4
    t[1, 2] = 2
    t[2, 0] = 1
    t[2, 1] = 2
    t[2, 2] = 1
    t = t / 8
    P0 = 1
    P1 = t
    h = a[1] * P1
    inset0 = int(np.floor((t.shape[-1] - 1) / 2))
    inset1 = int(np.floor((t.shape[-2] - 1) / 2))
    inset = [inset0, inset1]
    rows = inset[0] + 1
    cols = inset[1] + 1
    h[rows - 1, cols - 1] = h[rows - 1, cols - 1] + a[0] * P0
    for i in np.arange(3, n + 1):
        print(i)
        P1 = torch_get_3D(P1,'HW')
        t = torch_get_4D(t,'HW')
        P2 = 2 * filter2D_torch(t, P1, border_type='constant', flag_return_valid_part_only=False)
        rows = rows + inset[0]
        cols = cols + inset[1]
        if np.isscalar(rows):
            P2[..., rows-1, cols - 1] = P2[..., rows-1, cols - 1:cols] - P0
        else:
            P2[..., rows[0]-1:rows[-1], cols[0]-1:cols[-1]] = P2[..., rows[0]-1:rows[-1], cols[0]-1:cols[-1]] - P0
        rows = inset[0] + torch.arange(P1.shape[-2]) + 1
        cols = inset[1] + torch.arange(P1.shape[-1]) + 1
        hh = h
        h = a[i-1] * P2
        h[..., rows[0]-1:rows[-1], cols[0]-1:cols[-1]] = h[..., rows[0]-1:rows[-1], cols[0]-1:cols[-1]] + torch_get_4D(hh,'HW')
        P0 = P1
        P1 = P2

    return h

def get_filter_2D_Simple(filter_type_low_high_bandpass_bandstop,filter_size,filter_order,low_cutoff,high_cutoff):
    filter_type_low_high_bandpass_bandstop = 'low'
    filter_size = 100
    filter_order = 30
    filter_parameter = 10
    low_cutoff = 20
    high_cutoff = 10
    filter_lenght = filter_order + 1

    low_cutoff = low_cutoff/2
    high_cutoff = high_cutoff/2

    X,Y = np.meshgrid(np.arange(filter_size), np.arange(filter_size))
    filter_center = np.ceil(filter_size/2) + (1-np.mod(filter_size,2))
    filter_end = np.floor((filter_size-1)/2)
    filter_start = -np.ceil((filter_size-1)/2)
    filter_length = filter_end - filter_start + 1
    distance_from_center = np.sqrt((X-filter_center)**2 + (Y-filter_center)**2)

    Fs = filter_size

    if filter_type_low_high_bandpass_bandstop == 'low':
        high_cutoff = Fs/2
        filter_current = np.exp(-(X-filter_center)**2/(low_cutoff**2) - (Y-filter_center)**2/(low_cutoff)**2)
    elif filter_type_low_high_bandpass_bandstop == 'high':
        low_cutoff = 0
        filter_current = 1 - np.exp(-(X-filter_center)**2/(high_cutoff)**2 - (Y-filter_center)**2/(high_cutoff)**2)
    elif filter_type_low_high_bandpass_bandstop == 'bandpass':
        max_cutoff = max(low_cutoff, high_cutoff)
        min_cutoff = min(low_cutoff, high_cutoff)
        filter_lowpass = np.exp(-(X-filter_center)**2/(max_cutoff**2) - (Y-filter_center)**2/(max_cutoff)**2)
        filter_highpass = 1 - np.exp(-(X-filter_center)**2/(min_cutoff)**2 - (Y-filter_center)**2/(min_cutoff)**2)
        filter_current = filter_lowpass * filter_highpass
    elif filter_type_low_high_bandpass_bandstop == 'bandstop':
        max_cutoff = max(low_cutoff, high_cutoff)
        min_cutoff = min(low_cutoff, high_cutoff)
        filter_lowpass = np.exp(-(X - filter_center) ** 2 / (max_cutoff ** 2) - (Y - filter_center) ** 2 / (max_cutoff) ** 2)
        filter_highpass = 1 - np.exp(-(X - filter_center) ** 2 / (min_cutoff) ** 2 - (Y - filter_center) ** 2 / (min_cutoff) ** 2)
        filter_current = 1 - filter_lowpass * filter_highpass

    return filter_current
    # plt.imshow(filter_current)

# %one must be carefull when using the butterworth filter because it is not FIR but IIR,
# %the filter_type can be: gaussian or butterworth
#
# %FOR NOW I WILL ONLY USE BUTTERWORTH BECAUSE IT'S SO SUPERIOR:
# flag_gaussian_or_butterworth = 2;
# % filter_type_low_high_bandpass_bandstop = 'bandstop';
# % filter_size = 100;
# % filter_order = 30;
# % filter_parameter = 10;
# % low_cutoff = 20;
# % high_cutoff = 30;
#
# %filter length = filter_order + 1
#
# %what i mean by "quick" is that the implementation is quick because it is
# %analytic and not interpolatory using chebychev polynomials.
#
# %(****) i divide by two to make it more closely resemble the stuff i get when i
# %define speckle size, in the end what i want is that when i define
# %speckle_size=50, that i can use that as a reference "frequency radius" in my functions
# low_cutoff = low_cutoff/2;
# high_cutoff = high_cutoff/2;
#
# % Initialize filter.
# %set up grid (including dummy check-up variables):
# [X,Y] = meshgrid(1:filter_size);
# filter_center = ceil(filter_size/2) + (1-mod(filter_size,2));
# % filter_center = ceil(filter_size/2);
# filter_end = floor((filter_size-1)/2);
# filter_start = -ceil((filter_size-1)/2);
# filter_length = filter_end-filter_start+1;
# %set up filter grid:
# distance_from_center  = sqrt((X-filter_center).^2 + (Y-filter_center).^2);
#
# %reverse 1D filter degisn logic, instead multiplying by 1/2 i multiply by 2:
# Fs = filter_size; %1[pixel]
#
#
# %check input to avoid thinking too much:
# if strcmp(filter_type_low_high_bandpass_bandstop,'low')==1
#    if flag_gaussian_or_butterworth==2
#        high_cutoff = Fs/2;
#        filter_current = 1./(1 + (distance_from_center/low_cutoff).^(2*filter_order));
#    elseif flag_gaussian_or_butterworth==1
#        high_cutoff = Fs/2;
#        filter_current = exp(-(X-filter_center).^2/(low_cutoff.^2) - (Y-filter_center).^2/(low_cutoff.^2));
#    end
# elseif strcmp(filter_type_low_high_bandpass_bandstop,'high')==1
#     if flag_gaussian_or_butterworth==2
#        low_cutoff = 0;
#        filter_current = 1 - 1./(1 + (distance_from_center/high_cutoff).^(2*filter_order));
#     elseif flag_gaussian_or_butterworth==1
#        low_cutoff = 0;
#        filter_current = 1-exp(-(X-filter_center).^2/(high_cutoff.^2) - (Y-filter_center).^2/(high_cutoff.^2));
#     end
# elseif (strcmp(filter_type_low_high_bandpass_bandstop,'bandpass')==1)
#    %i made this option dummy proof:
#    if low_cutoff<high_cutoff
#       warning('i made the filter but you stated frequencies wrong, with bandpass you need lowpass filter radius to be larger so high_cutoff<low_cutoff');
#    end
#    if flag_gaussian_or_butterworth==2
#        filter_low_pass = 1./(1 + (distance_from_center/max(low_cutoff,high_cutoff)).^(2*filter_order));
#        filter_high_pass = 1 - 1./(1 + (distance_from_center/min(low_cutoff,high_cutoff)).^(2*filter_order));
#    elseif flag_gaussian_or_butterworth==1
#        filter_low_pass = exp(-(X-filter_center).^2/(max(low_cutoff,high_cutoff).^2) - (Y-filter_center).^2/(max(low_cutoff,high_cutoff).^2));
#        filter_high_pass = 1 - exp(-(X-filter_center).^2/(min(low_cutoff,high_cutoff).^2) - (Y-filter_center).^2/(min(low_cutoff,high_cutoff).^2));
#    end
#    filter_current = filter_low_pass .* filter_high_pass;
# elseif (strcmp(filter_type_low_high_bandpass_bandstop,'bandstop')==1)
#    %i made this option dummy proof:
#    if low_cutoff<high_cutoff
#       warning('i made the filter but you stated frequencies wrong, with bandstop you need highpass filter radius to be larger so low_cutoff<high_cutoff');
#    end
#    if flag_gaussian_or_butterworth==2
#        filter_low_pass = 1./(1 + (distance_from_center/max(low_cutoff,high_cutoff)).^(2*filter_order));
#        filter_high_pass = 1 - 1./(1 + (distance_from_center/min(low_cutoff,high_cutoff)).^(2*filter_order));
#    elseif flag_gaussian_or_butterworth==1
#        filter_low_pass = exp(-(X-filter_center).^2/(max(low_cutoff,high_cutoff).^2) - (Y-filter_center).^2/(max(low_cutoff,high_cutoff).^2));
#        filter_high_pass = 1 - exp(-(X-filter_center).^2/(min(low_cutoff,high_cutoff).^2) - (Y-filter_center).^2/(min(low_cutoff,high_cutoff).^2));
#    end
#    filter_current = 1 - filter_low_pass .* filter_high_pass;
# end
#
# %get filter in real space for easy insert into filter2:
# filter_current_real = real(ift2(filter_current,1))/filter_size^2;
#
# % subplot(3,1,1)
# % imagesc(filter_current)
# % colorbar;
# % subplot(3,1,2)
# % filter_current_real = real(ift2(filter_current,1));
# % imagesc(filter_current_real);
# % colorbar;
# % subplot(3,1,3)
# % imagesc(real(ft2(filter_current_real,1/filter_size)));
# % colorbar;



# TODO: get overlap-add object for 1D signals layer
# TODO: write a real-space filtering layer and not just fft (for short filters)
# TODO: get dudy's FIR builder into pytorch version
class FFT_OLA_PerPixel_Layer_Torch(nn.Module):

    # Initialize this with a module
    def __init__(self, samples_per_frame, filter_name, filter_type, N, Fs, low_cutoff, high_cutoff, filter_parameter=True):
        super(FFT_OLA_PerPixel_Layer_Torch, self).__init__()

        ### Basic Parameters: ###
        self.samples_per_frame = samples_per_frame  #512
        self.overlap_samples_per_frame = int((self.samples_per_frame) * 1 / 2)
        self.non_overlapping_samples_per_frame = self.samples_per_frame - self.overlap_samples_per_frame

        ### Get the 1D filter: ###
        if filter_name == "hann" or filter_name == "hanning":
            self.filter_object, self.filter = self.get_filter_1D(filter_name, filter_type, N+2, Fs, low_cutoff,
                                             high_cutoff, filter_parameter=8)
        elif filter_name == "one":
            print(1)

        ### this part is supposedly mimics the matlab code above it not sure if it works: ###
        # TODO: self.frame_window = ones or hanning from scipy.windows whatever
        self.frame_window = torch.ones(samples_per_frame)  # Read this from file
        self.frame_window = torch_get_5D(self.frame_window, 'T')
        self.frame_window_length = len(self.frame_window)

        ### Filter Itself: ###
        self.filter_Numerator = torch.tensor(self.filter)
        self.filter_length = len(self.filter_Numerator)
        self.filter_Numerator = torch_get_5D(self.filter_Numerator, 'T')
        ## Prepare FFT: ###
        self.FFT_size = int(pow(2, np.ceil(np.log(self.samples_per_frame + self.filter_length - 1) / np.log(2))))
        self.filter_fft = torch.fft.fftn(self.filter_Numerator.squeeze(), s=int(self.FFT_size), dim=[0])
        self.filter_fft = torch_get_5D(self.filter_fft, 'T')

        # ### Show FFT Transfer function of filter: ###
        # filter_fft = self.filter_fft.squeeze()
        # filter_fft = torch_fftshift(filter_fft, first_spatial_dim=0)
        # plot_torch(filter_fft.abs())

        # Initialize lookahead buffer for overlap add operation:
        self.lookahead_buffer_for_overlap_add = None


    def get_filter_1D(self, filter_name, filter_type, N, Fs, f_low_cutoff, f_high_cutoff, filter_parameter=True,
                      attenuation=0, beta=14):
        ### Get normalized frequencies by dividing by the nyquist frequency: ###
        f_low_cutoff = f_low_cutoff/(Fs/2)
        f_high_cutoff = f_high_cutoff/(Fs/2)
        print(f_high_cutoff, Fs / 2)

        # Get window type wanted:
        # if filter_name == 'kaiser':
        #     window = signal.windows.kaiser(N + 1, beta, filter_parameter)
        # elif filter_name == 'hann' or filter_name == 'hanning':
        #     window = signal.windows.hann(N + 1, filter_parameter)
        # elif filter_name == 'hamming' or filter_name == "hamm":
        #     window = signal.windows.hamming(N + 1, filter_parameter)
        # elif filter_name == 'blackmanharris':
        #     window = signal.windows.blackmanharris(N + 1, filter_parameter)
        # elif filter_name == 'cheb':
        #     window = signal.windows.chebwin(N + 1, attenuation, filter_parameter)

        ### Get filter name and match is to whatever signal.firwin expects the string to be: ###
        if filter_name == 'kaiser':
            filter_name = "kaiser"
        elif filter_name == 'hann' or filter_name == 'hanning':
            filter_name = "hann"
        elif filter_name == 'hamming' or filter_name == "hamm":
            filter_name = "hamming"
        elif filter_name == 'blackmanharris':
            filter_name = "blackmanharris"
        elif filter_name == 'cheb':
            filter_name = "chebwin"

        # Calculate the coefficients using the fir1 function
        if filter_name in ["chebwin", "kaiser"]:
            if filter_type in ['bandpass', 'stop']:
                coefficients = signal.firwin(N, [f_low_cutoff, f_high_cutoff], width=attenuation, window=filter_name, pass_zero="bandpass", fs=Fs)
            elif filter_type in ['low', 'lowpass']:
                coefficients = signal.firwin(N, f_low_cutoff, window=filter_name, width=attenuation, pass_zero="lowpass", fs=Fs)
            elif filter_type in ['high', 'highpass']:
                coefficients = signal.firwin(N, f_high_cutoff, window=filter_name, width=attenuation, pass_zero="highpass", fs=Fs)
        else:
            if filter_type in ['bandpass', 'stop']:
                coefficients = signal.firwin(N, [f_low_cutoff, f_high_cutoff], window=filter_name, pass_zero="bandpass", fs=Fs)
            elif filter_type in ['low', 'lowpass']:
                coefficients = signal.firwin(N, f_low_cutoff, window=filter_name, pass_zero="lowpass", fs=Fs)
            elif filter_type in ['high', 'highpass']:
                coefficients = signal.firwin(N, f_high_cutoff, window=filter_name, pass_zero="highpass", fs=Fs)
        actual_filter = signal.dlti(coefficients, [1], dt=1)

        # coefficients_torch = torch.tensor(coefficients)
        # plt.figure()
        # plot_torch(torch_fftshift(torch.fft.fftn(coefficients_torch, dim=0), first_spatial_dim=0).abs())

        return actual_filter, coefficients

        # actual_filter.persistent_memory = True
        # actual_filter.states = 0

    def forward(self, input_tensor):
        #### Get Mat Size: ###
        # TODO: create a version for 1D signal later (either simply [T] or something else)
        B, T, C, H, W = input_tensor.shape

        ### Initialize look-ahead buffer: ###
        if self.lookahead_buffer_for_overlap_add is None:
            self.lookahead_buffer_for_overlap_add = torch.zeros(1, self.FFT_size, 1, H, W).to(input_tensor.device)
            self.zeros_mat = torch.zeros(1, self.overlap_samples_per_frame, 1, H, W).to(input_tensor.device)
            self.frame_window = torch.hann_window(window_length=input_tensor.shape[1])
            self.frame_window = torch_get_5D(self.frame_window, 'T')

        ### Actually Filter: ###
        # (1). window signal:
        input_tensor = input_tensor * self.frame_window
        # (2). calculate buffered and windowed frame fft:
        input_tensor_fft = torch.fft.fftn(input_tensor, s=self.FFT_size, dim=[1])
        # (3). calculate time domain filtered signal:
        filtered_signal = torch.fft.ifftn(input_tensor_fft * self.filter_fft, dim=[1]).real

        ### Overlap-Add Method: ###
        # (1). overlap add:
        self.lookahead_buffer_for_overlap_add = self.lookahead_buffer_for_overlap_add + filtered_signal
        # (2). get current valid part of the overlap-add:
        filtered_signal_final_valid = self.lookahead_buffer_for_overlap_add[:, 0:self.non_overlapping_samples_per_frame]
        # (3). shift lookahead buffer tail to beginning of buffer for next overlap-add:
        self.lookahead_buffer_for_overlap_add = torch.cat(
            (self.lookahead_buffer_for_overlap_add[:, self.non_overlapping_samples_per_frame:], self.zeros_mat), 1)

        return filtered_signal_final_valid.real



