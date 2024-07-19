import torch

from torch import Tensor
from typing import Union, TypeVar, Optional

from RapidBase.import_all import *
from RapidBase.Anvil._internal_utils.torch_utils import NoMemoryMatrixOrigami
from RapidBase.Anvil._internal_utils.test_exceptions import raise_if, raise_if_not
from RapidBase.Anvil._internal_utils.parameter_checking import type_check_parameters, validate_warp_method
from RapidBase.Anvil._internal_utils.torch_utils import construct_tensor, true_dimensionality
from typing import Tuple
import warnings
from os import path as osp
# todo list:
#   1) take care of device
#   2) update all comments/documentation/declaration

class TurbulenceFlowFieldGenerationLayerAnvil(nn.Module):
    def __init__(self, H=None, W=None, batch_size=None, patch_size=15, device='cpu'):
        super(TurbulenceFlowFieldGenerationLayerAnvil, self).__init__()
        ### Parameters: ###
        self.PatchSize = patch_size

        ### Get Current Image Shape And Appropriate Meshgrid: ###
        PatchNumRow = int(np.round(H / self.PatchSize))
        PatchNumCol = int(np.round(W / self.PatchSize))

        self.device = device
        self.H = H
        self.W = W
        self.batch_size = batch_size
        if all([self.batch_size, self.H, self.W]):
            self._generate_meshgrid(self.H, self.W, self.batch_size)

        self.PatchNumRow = PatchNumRow
        self.PatchNumCol = PatchNumCol

    def _generate_meshgrid(self, H: int, W: int, batch_size: int):
        """
        Generates a new meshgrid with given dimensions

        return: assign a mesh grid attribute to object
        """
        [y_large, x_large] = np.meshgrid(np.arange(H), np.arange(W))
        x_large = torch.Tensor(x_large).unsqueeze(-1)
        y_large = torch.Tensor(y_large).unsqueeze(-1)
        x_large = (x_large - W / 2) / (W / 2 - 1)
        y_large = (y_large - H / 2) / (H / 2 - 1)

        new_grid = torch.cat([x_large, y_large], 2)
        new_grid = torch.Tensor([new_grid.numpy()] * batch_size)

        self.new_grid = new_grid

    def _format_params(self, std: Union[float, int]) -> Tuple[float]:
        """

        check format of all params given to forward
        note - currently its useless but maybe if we have more serious params we will use it

        """
        type_check_parameters([(std, (float, int))])

        std = float(std)
        return std

    def update_dimensions(self, batch_size: int, H: int, W: int, patch_size: int = 15):
        """
        update internal dimensions
        Args:
            batch_size:
            H:
            W:
            Patch_size:

        """
        self.H = H
        self.W = W
        self.batch_size = batch_size
        self.PatchSize = patch_size

        self.PatchNumRow = int(np.round(H / self.PatchSize))
        self.PatchNumCol = int(np.round(W / self.PatchSize))

        self._generate_meshgrid(self.H, self.W, self.batch_size)

    def _function_logic(self, std: float) -> Tuple[Tensor, Tensor]:
        """
        create a turbulence flow field(x and y fields)
        Args:
            std: starting noise, higher means more turbulence, 0 is no turbulence

        Returns:

        """
        ShiftMatX0 = torch.empty((self.batch_size, 1, self.PatchNumRow, self.PatchNumCol)).normal_(mean=0, std=std)
        ShiftMatY0 = torch.empty((self.batch_size, 1, self.PatchNumRow, self.PatchNumCol)).normal_(mean=0, std=std)
        ShiftMatX0 = ShiftMatX0 * self.W
        ShiftMatY0 = ShiftMatY0 * self.H

        ShiftMatX = torch.nn.functional.interpolate(ShiftMatX0, size=(self.new_grid.shape[2], self.new_grid.shape[1]), mode='bicubic')
        ShiftMatY = torch.nn.functional.interpolate(ShiftMatY0, size=(self.new_grid.shape[2], self.new_grid.shape[1]), mode='bicubic')

        ShiftMatX = ShiftMatX.squeeze()
        ShiftMatY = ShiftMatY.squeeze()

        return ShiftMatX, ShiftMatY

    def forward(self, std: Union[float, int] = 0.01) -> Tuple[Tensor, Tensor]:
        """
        wrapping the _function_logic function
        Args:
            std:

        Returns:

        """
        std = self._format_params(std)

        ShiftMatX, ShiftMatY = self._function_logic(std)

        return ShiftMatX, ShiftMatY


class TurbulenceDeformationLayerAnvil(nn.Module):
    def __init__(self, H=None, W=None, batch_size=None, patch_size=15, device='cpu'):
        super(TurbulenceDeformationLayerAnvil, self).__init__()
        self.W = W
        self.H = H
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.device = device
        self.turbulence_generator = TurbulenceFlowFieldGenerationLayerAnvil(self.H, self.W, self.batch_size, self.patch_size, self.device)
        # we use a fixed size warp object module and not the regular one.
        # because we don't want to create a new mesh grid every single time,
        # so we instantiate it from the beginning with given dimensions.
        self.warp_object = WarpObjectAnvil(self.batch_size, 1, 3, self.H, self.W, self.device)

    def _format_params(self, input_tensor: Union[Tensor, tuple, list, np.array], std: Union[float, int]) -> Tuple[Tensor, float]:
        """

        check format of all params given to forward
        note - currently its useless but maybe if we have more serious params we will use it

        """
        type_check_parameters([(input_tensor, (Tensor, tuple, list, np.array)), (std, (float, int))])

        input_tensor = construct_tensor(input_tensor)
        std = float(std)

        dimensions_memory = NoMemoryMatrixOrigami(input_tensor)
        input_tensor = dimensions_memory.expand_matrix(input_tensor)

        B, T, C, H, W = input_tensor.shape
        # print(f'printing dimensions inside of format_params in turbulence deformation layer : {B, T, C, H, W}')
        raise_if_not(C == 1 or C == 3, message='channel dimension is not recognized, not bw/rgb')

        return input_tensor, std

    def _validate_internal_parameters(self, input_tensor: Tensor, std: float):

        # self.device = input_tensor.device()
        B, T, C, H, W = input_tensor.shape
        # keep an updated meshgrid
        if B != self.batch_size or H != self.H or W != self.W:
            self.batch_size = B
            self.H = H
            self.W = W

            self.turbulence_generator.update_dimensions(self.batch_size, self.H, self.W, self.patch_size)
            # the warping object will create a new meshgrid when called

    def update_patch_size(self, patch_size: int):
        self.patch_size = patch_size
        self.turbulence_generator.update_dimensions(self.batch_size, self.H, self.W, self.patch_size)

    def _forward(self, input_tensor: Tensor, std: float = 2.0) -> Tuple[Tensor, Tensor, Tensor]:
        """
        creating a turbulence field and applying it to the image
        Args:
            input_tensor: image to noise with turbulence
            std: level of noise

        Returns: input image after adding turbulence and both(x & y) turbulence fields

        """
        delta_x_map, delta_y_map = self.turbulence_generator.forward(std)

        # output_tensor = self.warp_object.forward(matrix=input_tensor, delta_x=delta_x_map / self.W,
        #                                          delta_y=delta_y_map / self.H, warp_method='bicubic')

        output_tensor = self.warp_object.forward(matrix=input_tensor.squeeze(), delta_x=delta_x_map / self.W,
                                                 delta_y=delta_y_map / self.H, warp_method='bicubic')

        return output_tensor, delta_x_map, delta_y_map

    def forward(self, input_tensor: Union[Tensor, tuple, list, np.array], std: Union[float, int] = 2.0) -> Tuple[Tensor, Tensor, Tensor]:
        """
        wrapping forward function
        Args:
            input_tensor:
            std:

        Returns:

        """
        input_tensor, std = self._format_params(input_tensor, std)

        self._validate_internal_parameters(input_tensor, std)

        output_tensor, delta_x_map, delta_y_map = self._forward(input_tensor, std)

        return output_tensor, delta_x_map, delta_y_map


class WarpObjectAnvil(nn.Module):
    def __init__(self, B=None, T=None, C=None, H=None, W=None,
                 device='cpu'):
        super().__init__()
        self.B = B
        self.T = T
        self.C = C
        self.H = H

        self.W = W
        self.device = device

        # Calculate meshgrid to save future computation time
        if all([self.B, self.T, self.H, self.W]):
            self._generate_meshgrid()

    def _generate_meshgrid(self):
        """
        Generates a meshgrid based on the data sizes B, T, H, W. The generated meshgrid is a property of the WarpObject

        return: assigns meshgrid to object
        """
        # Create meshgrid:
        meshgrid_W, meshgrid_H = torch.meshgrid(torch.arange(self.W), torch.arange(self.H))
        # Turn meshgrid to be tensors:
        meshgrid_W = meshgrid_W.type(torch.FloatTensor).to(self.device).permute(1, 0)
        meshgrid_H = meshgrid_H.type(torch.FloatTensor).to(self.device).permute(1, 0)

        # Normalize meshgrid
        # meshgrid_W_max = meshgrid_W[-1, -1]
        # meshgrid_W_mcdin = meshgrid_W[0, 0]
        # meshgrid_H_max = meshgrid_H[-1, -1]
        # meshgrid_H_min = meshgrid_H[0, 0]
        # meshgrid_W_centered = meshgrid_W - (meshgrid_W_max - meshgrid_W_min) / 2
        # meshgrid_H_centered = meshgrid_H - (meshgrid_H_max - meshgrid_H_min) / 2
        # meshgrid_W = meshgrid_W_centered
        # meshgrid_H = meshgrid_H_centered

        # stack T*B times
        # todo: only tested with T=1
        meshgrid_W = torch.cat([meshgrid_W.unsqueeze(-1)] * (self.B * self.T), -1).unsqueeze(0)
        meshgrid_H = torch.cat([meshgrid_H.unsqueeze(-1)] * (self.B * self.T), -1).unsqueeze(0)

        self.meshgrid_W = meshgrid_W
        self.meshgrid_H = meshgrid_H

    def _warp_image(self, image: Tensor, grid: Tensor, warp_method: str):
        """
        warp image based on a grid

        param image: image to warp
        param grid: grid to warp by
        param warp_method: interpolation mode - 'bilinear'/'bicubic'/'nearest'
        return: warped image based on grid
        """
        # no need to test data types since it is already done by now
        B, T, C, H, W = image.shape
        reshaped_image = image.reshape((B * T, C, H, W))  # grid sample can only handle 4D
        output_tensor = torch.nn.functional.grid_sample(reshaped_image, grid, mode=warp_method)

        return output_tensor.reshape((B, T, C, H, W))

    def _warp_logic_grid(self, image: Tensor, grid: Tensor, warp_method: str):
        """
        logic behind warping based on grid - not much in here
        Args:
            image: image to warp
            grid: grid to warp by

        Returns: warped image

        """
        output_tensor = self._warp_image(image, grid, warp_method)
        return output_tensor

    def _warp_logic_delta_map(self, image: Tensor, delta_x: Tensor, delta_y: Tensor, warp_method: str):
        """
        logic behind warping based on deltas
        Args:
            image: image to warp
            delta_x: delta map to warp in x direction : image[x axis] + delta_x = warped_image[x axis]
            delta_y: delta map to warp in y direction : image[y axis] + delta_y = warped_image[y axis]

        Returns: warped image based on delta maps

        """
        new_X = 2 * ((self.meshgrid_W + delta_x.to(self.device))) / max(self.W - 1, 1) - 1  # scale to accepted values
        new_Y = 2 * ((self.meshgrid_H + delta_y.to(self.device))) / max(self.H - 1, 1) - 1  # from [0, H] to [-1, 1]
        grid = torch.cat([new_X, new_Y], dim=3)  # append grid
        # once we have a grid send it to by warped by grid
        output_tensor = self._warp_logic_grid(image, grid, warp_method)
        return output_tensor

    def _warp_logic(self, image: Tensor, grid: Tensor = None, delta_map: Tensor = None, delta_x: Tensor = None,
                    delta_y: Tensor = None, warp_method='bicubic', flag_use_grid=True):
        """
        # warp by grid if possible, otherwise warp by deltas
        # todo: by now it should have delta_x and delta_y even if only given delta_map, but make it formal
        Args: should all be tensors
            image:image to be warped by some parameters given
            grid: optional grid to warp by
            delta_map: optional delta maps to warp by
            delta_x: x component of delta map
            delta_y: y component of delta map
            flag_use_grid: if True and sent grid, use grid.

        Returns:

        """
        # I have already made sure that if use_grid flag is true but no grid was sent then the flag will be set to False..
        if flag_use_grid:  # means that there is a grid as well
            output = self._warp_logic_grid(image, grid, warp_method)
        # otherwise I want a delta map to be given, either as both components concatenated or as different x,y components:
        # again, I have made sure that when given a delta_map i deconstruct it into deltas and pass them on
        elif (delta_x is not None and delta_y is not None):
            output = self._warp_logic_delta_map(image, delta_x, delta_y, warp_method)
        else:
            raise RuntimeError('messed up')

        return output

    def _format_params(self,
                       matrix: Union[Tensor, tuple, list, np.array],
                       grid: Union[Tensor, np.array, tuple, list] = None,
                       delta_map: Union[Tensor, np.array, tuple, list] = None,
                       delta_x: Union[Tensor, np.array, tuple, list] = None,
                       delta_y: Union[Tensor, np.array, tuple, list] = None,
                       warp_method: str = 'bilinear',
                       flag_use_grid: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str]:
        """

        check format of all params given to forward

        """
        type_check_parameters(
            [(matrix, (Tensor, np.array, tuple, list)),
             (grid, (Tensor, np.array, tuple, list, type(None))),
             (delta_map, (Tensor, np.array, tuple, list, type(None))),
             (delta_x, (Tensor, np.array, tuple, list, type(None))),
             (delta_y, (Tensor, np.array, tuple, list, type(None))),
             (warp_method, str),
             (flag_use_grid, bool)])

        matrix = construct_tensor(matrix)

        dimensions_memory = NoMemoryMatrixOrigami(matrix)
        matrix = dimensions_memory.expand_matrix(matrix)

        # TODO think of a more elegant solution to the device problem
        # choose device according to the input data if is tensor
        if torch.is_tensor(matrix):
            device = matrix.device
        elif isinstance(matrix, list) and len(matrix) > 0 and torch.is_tensor(matrix[0]):
            device = matrix[0].device
        else:
            device = self.device

        self.device = device

        if grid is not None:
            grid = construct_tensor(grid)

            if len(grid.shape) == 3:
                grid = grid.unsqueeze(0)  # add batch dimension
            if len(grid.shape) == 4 and grid.shape[-1] == 2:
                pass  # all good
            elif len(grid.shape) == 4 and grid.shape[1] == 2:
                grid = grid.permute(0, 2, 3, 1)  # [b,2,h,w] -> [b,h,w,2]

            raise_if_not(grid.shape[1:3] == matrix.shape[-2:], message='grid dimensions different from image')
        else:
            flag_use_grid = False

        if delta_map is not None:
            delta_map = construct_tensor(delta_map)

            if len(delta_map.shape) == 3:
                delta_map = delta_map.unsqueeze(0)  # add batch dimension
            if len(delta_map.shape) == 4 and delta_map.shape[1] == 2:
                delta_map = delta_map.permute(0, 2, 3, 1)
            # elif len(delta_map.shape) == 4 and delta_map.shape[3] == 2:
            delta_x = delta_map[:, :, :, 0]
            delta_y = delta_map[:, :, :, 1]

            raise_if_not(delta_map.shape[1:3] == matrix.shape[-2:], message='delta_map dimensions different from image')

        if delta_x is not None:
            delta_x = construct_tensor(delta_x)

            if len(delta_x.shape) == 2:
                delta_x = delta_x.unsqueeze(0)
            if len(delta_x.shape) == 3:  # [B,H,W] - > [B,H,W,1]
                delta_x = delta_x.unsqueeze(-1)
            # (2). Dim=4 <-> [B,C,H,W], Different Flow For Each Channel:(todo: test this scenario - not used)
            elif (len(delta_x.shape) == 4 and delta_x.shape[
                1] == self.C):  # [B,C,H,W] - > [BXC,H,W,1] (because pytorch's function only warps all channels of a tensor the same way so in order to warp each channel seperately we need to transfer channels to batch dim)
                delta_x = delta_x.view(self.B * self.C, self.H, self.W).unsqueeze(-1)
            # (3). Dim=4 but C=1 <-> [B,1,H,W], Same Flow On All Channels:
            elif len(delta_x.shape) == 4 and delta_x.shape[1] == 1:
                delta_x = delta_x.permute([0, 2, 3, 1])  # [B,1,H,W] -> [B,H,W,1]

            raise_if_not(delta_x.shape[1:3] == matrix.shape[-2:], message='delta_x dimensions different from image')

        if delta_y is not None:
            delta_y = construct_tensor(delta_y)

            if len(delta_y.shape) == 2:
                delta_y = delta_y.unsqueeze(0)
            if len(delta_y.shape) == 3:  # [B,H,W] - > [B,H,W,1]
                delta_y = delta_y.unsqueeze(-1)
            # (2). Dim=4 <-> [B,C,H,W], Different Flow For Each Channel:(todo: test this scenario - not used)
            elif (len(delta_y.shape) == 4 and delta_y.shape[
                1] == self.C):  # [B,C,H,W] - > [BXC,H,W,1] (because pytorch's function only warps all channels of a tensor the same way so in order to warp each channel seperately we need to transfer channels to batch dim)
                delta_y = delta_y.view(self.B * self.C, self.H, self.W).unsqueeze(-1)
            # (3). Dim=4 but C=1 <-> [B,1,H,W], Same Flow On All Channels:
            elif len(delta_y.shape) == 4 and delta_y.shape[1] == 1:
                delta_y = delta_y.permute([0, 2, 3, 1])  # [B,1,H,W] -> [B,H,W,1]

            raise_if_not(delta_y.shape[1:3] == matrix.shape[-2:], message='delta_y dimensions different from image')

        if not warp_method in ['bilinear', 'bicubic', 'nearest']:
            warp_method = 'bicubic'
            raise RuntimeWarning(
                'please note that input to warping mode(interpolation style) was not valid - using bicubic interpolation')
        # if not self.warping_method in ['bilinear', 'bicubic', 'nearest']:
        #     self.warping_method = 'bicubic'
        #     raise RuntimeWarning(
        #         'please note that input to warping mode(interpolation style) was not valid - using bicubic interpolation')

        return matrix, grid, delta_map, delta_x, delta_y, warp_method, flag_use_grid

    def _validate_internal_parameters(self, matrix: Tensor):
        """
        Validate all the WarpObject intrnal params to match the given input. if any difference exists, change the
        corresponding internal values
        main function of this currently is to change the meshgrid when a different input is given, that is also why we pass the matrix.

        param matrix: image to be warped later on

        return:
        """
        if self.device != matrix.device:
            self.device = matrix.device
            self.meshgrid_H = self.meshgrid_H.to(self.device)
            self.meshgrid_W = self.meshgrid_W.to(self.device)
            # TODO if affine grids are precalculated convert to device

        B, T, C, H, W = matrix.shape
        # keep an updated meshgrid
        if B != self.B or T != self.T or H != self.H or W != self.W:
            self.B = B
            self.T = T
            self.H = H
            self.W = W
            warnings.warn("generating new meshgrid due to dimensionality change")
            self._generate_meshgrid()
        # Clean Memory just in case
        # torch.cuda.empty_cache()

    def forward(self,
                matrix: Union[Tensor, np.array, tuple, list],
                grid: Union[Tensor, np.array, tuple, list] = None,
                delta_map: Union[Tensor, np.array, tuple, list] = None,
                delta_x: Union[Tensor, np.array, tuple, list] = None,
                delta_y: Union[Tensor, np.array, tuple, list] = None,
                warp_method: str = 'bicubic',
                flag_use_grid: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        """

        Args:
            matrix: image to warp
            grid: optional grid to warp by
            delta_map: optional delta maps to warp by using mesh_grid + delta_map
            pass deltas seperatly:
                delta_x, delta_y
            warp_method: interpolation method : 'bicubic'/'bilinear'/nearest'
            flag_use_grid: if True and provided grid we are going to use the grid, otherwise use the deltas/delta_map

        Returns: the warped image based on params provided

        """
        # validate and format parameters
        matrix, grid, delta_map, delta_x, delta_y, warp_method, flag_use_grid = \
            self._format_params(matrix, grid, delta_map, delta_x, delta_y, warp_method, flag_use_grid)

        self._validate_internal_parameters(matrix)

        warped_matrix = self._warp_logic(matrix, grid, delta_map, delta_x, delta_y, warp_method, flag_use_grid)

        return warped_matrix


class OcclusionCheckAnvil(nn.Module):
    def __init__(self, W=None, H=None, device='cpu'):
        super(OcclusionCheckAnvil, self).__init__()
        self.backWarp = WarpObjectAnvil(W, H)
        self.device = device

    def _format_params(self,
                       ford_flow: Union[Tensor, np.array, tuple, list],
                       back_flow: Union[Tensor, np.array, tuple, list]) -> Tuple[Tensor, Tensor]:
        """
            check format of all params given to forward, assumes dimension of flow channels(x flow and y flow) are either
            in third or last position
        """
        type_check_parameters(
            [(ford_flow, (Tensor, np.array, tuple, list)),
             (back_flow, (Tensor, np.array, tuple, list))])

        ford_flow = construct_tensor(ford_flow)
        back_flow = construct_tensor(back_flow)

        raise_if(ford_flow.shape != back_flow.shape, message='shape mismatch between both flows')

        dimensions_memory = NoMemoryMatrixOrigami(ford_flow)
        ford_flow = dimensions_memory.expand_matrix(ford_flow, num_dims=4)
        back_flow = dimensions_memory.expand_matrix(back_flow, num_dims=4)

        #TODO : handle flow dimensions(all to B,H,W,C), dim should be 4 after expanding
        if ford_flow.shape[2] == 1:
            ford_flow = ford_flow.permute(0, 2, 3, 1)
            back_flow = back_flow.permute(0, 2, 3, 1)


        # TODO think of a more elegant solution to the device problem
        # choose device according to the input data if is tensor
        if torch.is_tensor(ford_flow):
            device = ford_flow.device
        elif isinstance(ford_flow, list) and len(ford_flow) > 0 and torch.is_tensor(ford_flow[0]):
            device = ford_flow[0].device
        else:
            device = self.device
        self.device = device

        return ford_flow, back_flow

    def _forward_logic(self, ford_flow: Tensor, back_flow: Tensor) -> Tensor:
        """
        compute the occlusion mask
        Args:
            ford_flow: forward flow
            back_flow: backward flow

        Returns: occlusion mask

        """
        #   we warp the backward flow according to the forward flow,
        #   which aligns the backward flow pixels with the forward flow pixels,
        #   for a non occluded area, adding the above aligned flow with the forward flow should
        #   result in a 0 flow.
        #   if the area was occluded than the flow couldn't possibly do a good job and
        #   had probably sent the flow vector to the closest area it knows, and its direction/magnitude
        #   might as well be random.
        mask = ford_flow + self.backWarp(matrix=back_flow, delta_map=ford_flow)
        # mask = torch.norm(mask, p=2, dim=1).unsqueeze(1)
        mask = torch.norm(mask, p=2, dim=2)
        mask = 2 * torch.sigmoid(mask) - 1
        mask = 1 - mask
        return mask

    def _validate_internal_parameters(self, flow: Tensor):
        """
            validate object's internal parameters
        """
        if self.device != flow.device:
            self.device = flow.device

        B, H, W, C = flow.shape
        # currently no need to do anything, warp object will automatically
        # resize itself when sent a different size


    def forward(self,
                ford_flow: Union[Tensor, np.array, tuple, list],
                back_flow: Union[Tensor, np.array, tuple, list]) -> Tensor:
        """
        call the wrapper functions that validate parameters and compute the occlusion mask
        Args:
            ford_flow: forward flow
            back_flow: backward flow

        Returns: an occlusion mask
        """

        ford_flow, back_flow = self._format_params(ford_flow, back_flow)
        self._validate_internal_parameters(ford_flow)  # to validate dimensions

        mask = self._forward_logic(ford_flow, back_flow)
        return mask



def combine_flows_anvil(flows: Union[torch.Tensor, np.array, tuple, list],
                        reverse: bool = False,
                        flip_sign: bool = False):
    """
    compute the total flow between frame 0 and N, given N micro-flows between each consecutive frames.
    Args:
        flows: a set of continuous flows between N frames, shape should be (N, 1, 2, H, W)
        reverse: True if we should reverse the order of the flow, when having a sequence of images where the ref is the first one and not the last.
        flip_sign: see Note below.

        Note -  here I assume that if we are in reverse mode I don't need to flip the sign of the flow as well,
                The assumption is that the flow was already calculated in reverse.
                As is in a multi_frame scenario where I calculate flow(ref, origin) in that order for all frames,
                regardless of order(meaning I don't do flow(origin, ref) for when ref is chronologically before origin)

    Returns:
        combined_flow: a flow between the first and last frame(when reverse=False)

    """
    type_check_parameters([(flows, (Tensor, np.array, tuple, list)), (reverse, bool), (flip_sign, bool)])

    flows = construct_tensor(flows)
    raise_if(true_dimensionality(flows) == 0, message="no Optical Flow passed")

    dimensions_memory = NoMemoryMatrixOrigami(flows)
    flows = dimensions_memory.expand_matrix(flows)
    B, T, C, H, W = flows.shape
    warp_obj = WarpObjectAnvil(B, T, C, H, W, flows.device)

    if reverse:
        flows = flows.flip(0)
    if flip_sign:
        flows = -flows

    # of shape [b, t, c, h, w]
    combined_flows = torch.zeros_like(flows[0:1])
    inverse_flows = torch.zeros_like(flows)

    for id, flow in enumerate(flows):
        inverse_flows[id:id+1] = -warp_obj.forward(flow, delta_x=flow[:, 0:1], delta_y=flow[:, 1:2])

    b,t,c,h,w = combined_flows.shape
    for id ,flow in enumerate(flows[1:]):
        ## delta_x, delta_y = inverse_flow(combined_flows)
        res = flow + 0
        for inv_flow in torch.flip(inverse_flows[:id], [0]):
            res = warp_obj.forward(res, delta_x=inv_flow[:, 0:1], delta_y=inv_flow[:, 1:2])

        combined_flows += res
        # combined_flows += prev_flow

    if len(combined_flows.shape) == 4:
        combined_flows = combined_flows.unsqueeze(0)

    return combined_flows

#
# def testing_occlusion_mask():
#     frame_0, frame_1, ff, bf, fo, bo = read_defult_OF_data()
#     OCC_OBJ = OcclusionCheckAnvil()
#     mask = OCC_OBJ.forward(ff, bf)
#     imshow_torch(mask[0][0])
#     imshow_torch(fo[0][0])
#     imshow_torch(bo[0][0])
#     plt.show()
#
#
# def testing_warping_and_turbulence():
#     im_d = read_image_torch("/raid/yoav/checkpoints/Testing_Flow_General/100_pixel/_FlowFormer/Inference/Flow_Scenarios_General.py/RapidBase__TrainingCore__datasets__Dataset_MultipleImagesFromSingleImage_AGN_YOAV_validation/42000/0000/0000_Input_Deformed.png")
#     B, C, H, W = im_d.shape
#     T = 1
#     # flow_x = read_image_torch("/raid/yoav/checkpoints/Testing_Flow_General/1_pixel/_FlowFormer/Inference/Flow_Scenarios_General.py/RapidBase__TrainingCore__datasets__Dataset_MultipleImagesFromSingleImage_AGN_YOAV_validation/42000/0000/0000_OpticalFlow_GT_X.png")
#     # flow_y = read_image_torch("/raid/yoav/checkpoints/Testing_Flow_General/1_pixel/_FlowFormer/Inference/Flow_Scenarios_General.py/RapidBase__TrainingCore__datasets__Dataset_MultipleImagesFromSingleImage_AGN_YOAV_validation/42000/0000/0000_OpticalFlow_GT_Y.png")
#     # flow_x = flow_x[:, 0:1, :, :] / 5
#     # flow_y = flow_y[:, 0:1, :, :] / 5
#
#     std = 2
#
#     turb_deform = TurbulenceDeformationLayerAnvil(H-200, W-200, 6)
#     turb_deform.update_patch_size(patch_size=20)
#
#     cropped_image = im_d[:, :, 30:-30, 30:-30]
#     output, delta_X, delta_y = turb_deform(im_d, std=std)
#     imshow_torch(output[0] / 255, title_str='output from object');
#     plt.show();
#
#
# def testing_combine_flows():
#     def get_flow_files_from_folder(root_flow_folder):
#         flows_filenames_list = get_flow_filenames_from_folder(root_flow_folder, 100, ['.flo'], True, '*')
#         return flows_filenames_list
#
#     def get_filenames_and_images_from_folder(root_images_folder):
#         image_filenames_list = get_image_filenames_from_folder(root_images_folder, 100, ['.png'], True, '*')
#         return image_filenames_list
#
#     def get_single_image_from_folder_flowformer(image_filenames_list, index):
#
#         ### Load Image: ###
#         filename = image_filenames_list[index]
#
#         # current_image = IO_dict.image_loader(filename)
#         ext = splitext(filename)[-1]
#         if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
#             current_image = Image.open(filename)
#         elif ext == '.bin' or ext == '.raw':
#             current_image = np.load(filename)
#
#         current_image = np.array(current_image).astype(np.uint8)
#         if len(current_image.shape) == 2:
#             current_image = np.tile(current_image[..., None], (1, 1, 3))
#         else:
#             current_image = current_image[..., :3]
#
#         current_image = torch.from_numpy(current_image).permute(2, 0, 1).float()
#
#         return current_image
#
#     def get_single_flow_from_folder(flow_filenames_list, index):
#         ### Load Image: ###
#         filename = flow_filenames_list[index]
#
#         # current_image = IO_dict.image_loader(filename)
#         f = open(filename, 'rb')
#         magic = np.fromfile(f, np.float32, count=1)[0]
#         flow = None
#
#         if 202021.25 != magic:
#             print('Magic number incorrect. Invalid .flo file')
#         else:
#             w = np.fromfile(f, np.int32, count=1)[0]
#             h = np.fromfile(f, np.int32, count=1)[0]
#             flow = np.fromfile(f, np.float32, count=2 * w * h)
#             # reshape data into 3D array (columns, rows, channels)
#             flow = np.resize(flow, (h, w, 2))
#
#             flow = np.array(flow).astype(np.float32)
#             flow = torch.from_numpy(flow).permute(2, 0, 1).float()
#
#         f.close()
#
#         return flow
#
#     def load_flows(N=10):
#         flows = []
#
#         flow_filenames_list = get_flow_files_from_folder(osp.join('/raid/datasets', 'sintel_train/flow/alley_1'))
#         for i in range(N):
#             flows.append(get_single_flow_from_folder(flow_filenames_list, i))
#
#         return flows
#
#     def load_images(N=10):
#         flows = []
#
#         images_filenames_list = get_filenames_and_images_from_folder(
#             osp.join('/raid/datasets', 'sintel_train/clean/alley_1'))
#         for i in range(N):
#             flows.append(get_single_image_from_folder_flowformer(images_filenames_list, i))
#
#         return flows
#
#     n = 7
#     flows = load_flows(n)
#     frames = load_images(n)
#     # for i in range(10):
#     #     imshow_torch(flows[i][0] / 255, title_str=f'{i}');
#     # plt.show()
#     imshow_torch(frames[0] / 255, title_str=f'{0}');
#     imshow_torch(frames[-1] / 255, title_str=f'{-1}');
#
#     flows = torch.cat([f.unsqueeze(0).unsqueeze(0) for f in flows], 0)
#     flow_yoav = combine_flows_anvil(flows) * 1
#
#     imshow_torch(flow_yoav[0, 0, 0], title_str='flow_X')
#     imshow_torch(flow_yoav[0, 0, 1], title_str='flow_y')
#     B,T,C,H,W = flows.shape
#     warp_object = WarpObjectAnvil(B, T, C, H, W, flows.device)
#     warped_first = warp_object.forward(matrix=frames[0],
#                                       delta_x=-flow_yoav[0, 0, 0:1],
#                                       delta_y=-flow_yoav[0, 0, 1:2])
#     warped_last = warp_object.forward(matrix=frames[-1],
#                                       delta_x=-flow_yoav[0, 0, 0:1],
#                                       delta_y=-flow_yoav[0, 0, 1:2])
#
#     imshow_torch(warped_first[0] / 255, title_str='warped_image1')
#     imshow_torch(warped_last[0] / 255, title_str='warped_image2')
#     plt.show()
#
#
# if __name__ == "__main__":
#     testing_occlusion_mask()