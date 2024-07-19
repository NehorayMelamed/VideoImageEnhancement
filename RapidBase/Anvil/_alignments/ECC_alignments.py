import torch
import torchvision.transforms
from torch import Tensor

from _internal_utils.torch_utils import RGB2BW, NoMemoryMatrixOrigami


# TODO Enable parallelism, currently not possible!

class Result:
    H_matrix: Tensor
    rho: float


def ECC_pytorch(input_tensor: Tensor,
                reference_tensor: Tensor,
                number_of_levels: int = 1,
                number_of_iterations_per_level: int = 25,
                transform_string: str = 'affine',
                delta_p_init: Tensor = None):
    # --- I. Initialize Parameters ---
    # TODO currently CHW, to be refactored to BTCHW
    dimensions_memory = NoMemoryMatrixOrigami(input_tensor)
    input_tensor = dimensions_memory.expand_matrix(input_tensor, num_dims=3)
    reference_tensor = dimensions_memory.expand_matrix(reference_tensor, num_dims=3)

    break_flag = 0
    transform_string = str.lower(transform_string)
    C_input, H_input, W_input = input_tensor.shape
    C_reference, H_reference, W_reference = reference_tensor.shape

    results = [[Result() for j in range(number_of_iterations_per_level)] for i in range(number_of_levels)]

    # Initialize New Images For the Algorithm To Change
    init_image = input_tensor
    init_template = reference_tensor
    input_tensor = RGB2BW(input_tensor).astype(float).squeeze()
    reference_tensor = RGB2BW(reference_tensor).astype(float).squeeze()
    reference_tensor_output_list = [0] * number_of_levels
    input_tensor_output_list = [0] * number_of_levels

    # --- II. Create Pyramid of Images ---
    # To enable pyramid structure of the algorithm, the following loop will produce two lists:
    # 1. current_level_input_tensor
    # 2. reference_tensor_output_list
    # where the first element of each list holds the highest resolution

    # Smoothing of original images
    # TODO: in the matlab version they overwrite the gaussian blur, so they're not actually blurring anything
    # reference_tensor_output_list[0] = cv2.GaussianBlur(reference_tensor, [7,7], 0.5)
    # input_tensor_output_list[0] = cv2.GaussianBlur(input_tensor, [7,7], 0.5)

    reference_tensor_output_list[0] = reference_tensor
    input_tensor_output_list[0] = input_tensor
    for level_index in torch.arange(1, number_of_levels):
        H, W = input_tensor_output_list[level_index - 1].shape
        resize_transform = torchvision.transforms.Resize((W // 2, H // 2))
        input_tensor_output_list[level_index] = resize_transform(input_tensor_output_list[level_index - 1])
        reference_tensor_output_list[level_index] = resize_transform(reference_tensor_output_list[level_index - 1])

    # --- III. Initialize the Homography Matrix ---
    # 1. Translation
    if transform_string == 'translation':
        number_of_parameters = 2  # number of parameters
        if delta_p_init is None:
            H_matrix = torch.zeros(2, 1)
        else:
            H_matrix = delta_p_init

    # 2. Euclidean
    elif transform_string == 'euclidean':
        number_of_parameters = 3  # number of parameters
        if delta_p_init is None:
            H_matrix = torch.eye(3)
            H_matrix[-1, -1] = 0
        else:
            H_matrix = torch.cat([delta_p_init, torch.zeros(1, 3)], 0)

    # 3. Affine
    elif transform_string == 'affine':
        number_of_parameters = 6  # number of parameters
        if delta_p_init is None:
            H_matrix = torch.eye(3)
            H_matrix[-1, -1] = 0
        else:
            H_matrix = torch.cat([delta_p_init, torch.zeros(1, 3)], 0)

    # 4. Homography
    elif transform_string == 'homography':
        number_of_parameters = 8  # number of parameters
        if delta_p_init is None:
            H_matrix = torch.eye(3)
        else:
            H_matrix = delta_p_init

    # in case of pyramid implementation, the initial transformation must be appropriately modified
    for level_index in torch.arange(0, number_of_levels - 1):
        H_matrix = correct_H_matrix_for_coming_level_torch(H_matrix, transform_string, 'lower_resolution')

    # --- IV. Run ECC Alignment For Each Level of the Pyramid ---
    # Loop from the lowest resolution to the highest
    for level_index in torch.arange(number_of_levels, 0, -1):
        # Get Current Level input_tensor and reference_tensor
        current_level_input_tensor = input_tensor_output_list[level_index - 1]
        current_level_reference_tensor = reference_tensor_output_list[level_index - 1]
        H, W, C = current_level_reference_tensor.shape

        # Get input_tensor's gradients
        [vy, vx] = torch.gradient(current_level_reference_tensor, dim=[0, 1])
        # vx = current_level_reference_tensor[:, 1:] - current_level_reference_tensor[:, 0:-1] # manually estimate gradients?
        # imshow(vx)

        # Define the rectangular Region of Interest (ROI) by x_vec and y_vec (you can modify the ROI)
        # Here we just ignore some image margins.
        # Margin is equal to 5 percent of the mean of [height,width].
        m0 = np.mean([H, W])
        # margin = floor(m0 * .05 / (2 ** (level_index - 1)))
        margin = 0  # no - margin - modify these two lines if you want to exclude a margin
        x_vec = torch.arange(margin, W - margin)
        y_vec = torch.arange(margin, H - margin)
        current_level_input_tensor = current_level_input_tensor[..., margin: H - margin, margin: W - margin].astype(
            float)

        # --- Run ECC, Forward Additive Algorithm: ---
        for iteration_index in torch.arange(number_of_iterations_per_level):
            print(f'Level: {level_index}, Iteration: {iteration_index}')
            print(H_matrix)
            wim = spatial_interpolation_torch(current_level_reference_tensor, H_matrix, 'bilinear', transform_string,
                                              x_vec, y_vec, H, W)  # inverse(backward) warping

            ### define a mask to deal with warping outside the image borders: ###
            # (they may have negative values due to the subtraction of the mean value)
            # TODO: there must be an easier way to do this!!! no way i need all these calculations
            ones_map = spatial_interpolation_torch(torch.ones_like(current_level_input_tensor), H_matrix, 'nearest',
                                                   transform_string, x_vec, y_vec, H, W)  # inverse(backward) warping
            numOfElem = (ones_map != 0).sum()

            meanOfWim = (wim * (ones_map != 0)).sum() / numOfElem
            meanOfTemp = (current_level_input_tensor * (ones_map != 0)).sum() / numOfElem

            wim = wim - meanOfWim  # zero - mean image; is useful for brightness change compensation, otherwise you can comment this line
            tempzm = current_level_input_tensor - meanOfTemp  # zero - mean reference_tensor

            wim[ones_map == 0] = 0  # for pixels outside the overlapping area
            tempzm[ones_map == 0] = 0

            # ### Save current transform: ###
            # # TODO: find an appropriate data structure / object for this
            # if transform_string == 'affine' or transform_string == 'euclidean':
            #     results[level_index, iteration_index].H_matrix = H_matrix[0:2, :]
            # else:
            #     results[level_index, iteration_index].H_matrix = H_matrix
            # results[level_index, iteration_index].rho = dot(current_level_reference_tensor[:], wim[:]) / norm(tempzm[:]) / norm(wim[:])

            ### Break the loop if reached max number of iterations per level: ###
            if iteration_index == number_of_iterations_per_level:  # the algorithm is executed (number_of_iterations_per_level-1) times
                break

            ### Gradient Image interpolation (warped gradients): ###
            wvx = spatial_interpolation_torch(vx, H_matrix, 'bilinear', transform_string, x_vec, y_vec, H, W)
            wvy = spatial_interpolation_torch(vy, H_matrix, 'bilinear', transform_string, x_vec, y_vec, H, W)

            ### Compute the jacobian of warp transform_string: ###
            J = get_jacobian_for_warp_transform_numpy(x_vec, y_vec, H_matrix, transform_string, H, W)
