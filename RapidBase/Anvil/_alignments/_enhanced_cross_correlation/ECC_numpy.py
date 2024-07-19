import cv2
import numpy as np
import scipy.ndimage

from RapidBase.import_all import *


def numpy_flatten(input_array, flag_make_Nx1=True, order='C'):
    if flag_make_Nx1:
        return input_array.flatten(order).reshape(np.prod(input_array.shape), -1)
    else:
        return input_array.flatten(order)


def get_jacobian_for_warp_transform_numpy(x_vec, y_vec, H_matrix, transform_string, H, W):
    #     %J = get_jacobian_for_warp_transform_numpy(x_vec, y_vec, H_matrix, transform_string)
    # % This function computes the jacobian J of H_matrix transform with respect
    # % to parameters. In case of homography/euclidean transform, the jacobian depends on
    # % the parameter values, while in affine/translation case is totally invariant.
    # %
    # % Input variables:
    # % x_vec:           the x-coordinate values of the horizontal side of ROI (i.e. [xmin:xmax]),
    # % y_vec:           the y-coordinate values of vertical side of ROI (i.e. [ymin:ymax]),
    # % H_matrix:         the H_matrix transform (used only in homography and euclidean case),
    # % transform_string:    the type of adopted transform
    # % {'affine''homography','translation','euclidean'}
    # %
    # % Output:
    # % J:            The jacobian matrix J

    ### Get vec sizes: ###
    x_vec_length = len(x_vec)
    y_vec_length = len(y_vec)

    ### Initialize the jacobians: ###
    x_vec_unsqueezed = numpy_unsqueeze(x_vec, 0)
    y_vec_unsqueezed = numpy_unsqueeze(y_vec, -1)
    Jx = np.repeat(x_vec_unsqueezed, y_vec_length, 0)
    Jy = np.repeat(y_vec_unsqueezed, x_vec_length, -1)
    J0 = 0 * Jx  #could also use zeros_like
    J1 = J0 + 1  #could also use ones_like

    ### Flatten Arrays: ###
    # J0 = numpy_flatten(J0).squeeze()
    # J1 = numpy_flatten(J1).squeeze()
    # Jx = numpy_flatten(Jx)
    # Jy = numpy_flatten(Jy)

    if str.lower(transform_string) == 'homography':
        ### Concatenate the flattened jacobians: ###
        #TODO: aren't these all simply ones?!?! why?! we're just copying the H_matrix, which i can do without all of this
        xy = np.concatenate([np.transpose(numpy_flatten(Jx, True, 'F')),
                             np.transpose(numpy_flatten(Jy, True, 'F')),
                             np.ones((1, x_vec_length * y_vec_length))], 0)  #TODO: before axis was -1

        ### 3x3 matrix transformation: ###
        A = H_matrix
        A[2, 2] = 1

        ### new coordinates after H_matrix: ###
        xy_prime = np.matmul(A, xy)  #matrix multiplication

        ### division due to homogeneous coordinates: ###
        xy_prime[0,:] = xy_prime[0,:] / xy_prime[2, :] #element-wise
        xy_prime[1,:] = xy_prime[1,:] / xy_prime[2, :]
        den = np.transpose(xy_prime[2,:])  #TODO: understand if this is needed
        den = np.reshape(den, (H,W), order='F')
        Jx = Jx / den #element-wise
        Jy = Jy / den #element-wise
        
        ### warped jacobian(???): ###
        Jxx_prime = Jx
        Jxx_prime = Jxx_prime * np.reshape(xy_prime[0,:], (H,W), order='F')  #element-wise
        Jyx_prime = Jy
        Jyx_prime = Jyx_prime * np.reshape(xy_prime[0,:], (H,W), order='F')

        Jxy_prime = Jx
        Jxy_prime = Jxy_prime * np.reshape(xy_prime[1,:], (H,W), order='F')  #element-wise
        Jyy_prime = Jy
        Jyy_prime = Jyy_prime * np.reshape(xy_prime[1,:], (H,W), order='F')

        ### Get final jacobian of the H_matrix with respect to the different parameters: ###
        J_up = np.concatenate([Jx, J0, -Jxx_prime, Jy, J0, - Jyx_prime, J1, J0], -1)
        J_down = np.concatenate([J0, Jx, -Jxy_prime, J0, Jy, -Jyy_prime, J0, J1], -1)
        J = np.concatenate([J_up, J_down], 0)

    elif str.lower(transform_string) == 'affine':
        Jx = Jx.squeeze()
        Jy = Jy.squeeze()
        J_up = np.concatenate([Jx, J0, Jy, J0, J1, J0], -1)
        J_down = np.concatenate([J0, Jx, J0, Jy, J0, J1], -1)
        J = np.concatenate([J_up, J_down], 0)

    elif str.lower(transform_string) == 'translation':
        Jx = Jx.squeeze()
        Jy = Jy.squeeze()
        J_up = np.concatenate([J1, J0], -1)
        J_down = np.concatenate([J0, J1], -1)
        J = np.concatenate([J_up, J_down], 0)

    elif str.lower(transform_string) == 'euclidean':
        Jx = Jx.squeeze()
        Jy = Jy.squeeze()
        mycos = H_matrix[1, 1]
        mysin = H_matrix[2, 1]

        Jx_prime = -mysin * Jx - mycos * Jy
        Jy_prime = mycos * Jx - mysin * Jy
        
        J_up = np.concatenate([Jx_prime, J1, J0], -1)
        J_down = np.concatenate([Jy_prime, J0, J1], -1)
        J = np.concatenate([J_up, J_down], 0)
        
    return J



def spatial_interpolation_numpy(input_image, H_matrix, interpolation_method, transform_string, x_vec, y_vec, H, W):
    # %OUT = spatial_interpolation_numpy(IN, H_matrix, STR, transform_string, x_vec, y_vec)
    # % This function implements the 2D spatial interpolation of image IN
    # %(inverse warping). The coordinates defined by x_vec,y_vec are projected through
    # % H_matrix thus resulting in new subpixel coordinates. The intensity values in
    # % new pixel coordinates are computed via bilinear interpolation
    # % of image IN. For other valid interpolation methods look at the help
    # % of Matlab function INTERP2.
    # %
    # % Input variables:
    # % IN:           the input image which must be warped,
    # % H_matrix:         the H_matrix transform,
    # % STR:          the string corresponds to interpolation method: 'linear',
    # %               'cubic' etc (for details look at the help file of
    # %               Matlab function INTERP2),
    # % transform_string:    the type of adopted transform: {'translation','euclidean','affine','homography'}
    # % x_vec:           the x-coordinate values of horizontal side of ROI (i.e. [xmin:xmax]),
    # % y_vec:           the y-coordinate values of vertical side of ROI (i.e. [ymin:ymax]),
    # %
    # % Output:
    # % OUT:          The warped (interpolated) image
    
    
    ### Correct H_matrix If Needed: ###
    if transform_string == 'affine' or transform_string == 'euclidean':
        if H_matrix.shape[0] == 2:
            H_matrix = np.concatenate([H_matrix, np.zeros(1,3)], 0)
    if transform_string == 'translation':
        H_matrix = np.concatenate([np.eye(2), H_matrix], -1)
        H_matrix = np.concatenate([H_matrix, np.zeros((1,3))], 0)


    # ###############################################
    # ### TODO: temp, delete: ####
    # ### HOW TO USE DIFFERENT INTERPOLATION FUNCTIONS: ###
    # #(1). scipy.ndimage.map_coordinates:
    # a = np.arange(12.).reshape((4, 3))
    # coordinates =  [[0.5, 2], [0.5, 1]]  #a list of lists or a list of tuples where to predict
    # result = scipy.ndimage.map_coordinates(a, coordinates, order=1)
    # #(2). scipy.interpolate.griddata:
    # def func(x, y):
    #     return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2
    # grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
    # rng = np.random.default_rng()
    # coordinates_given = rng.random((1000, 2))  #[N,2] array
    # values_given = func(coordinates_given[:, 0], coordinates_given[:, 1])  #[N] array
    # points_to_predict = (grid_x, grid_y)  #a tuple of (X,Y) meshgrids!!!
    # grid_z2 = scipy.interpolate.griddata(coordinates_given, values_given, points_to_predict, method='cubic')
    # #(3). scipy.interpolate.interp2d:  seems to accept only x_vec's and meshgrids
    # x_vec = np.arange(-5.01, 5.01, 0.25)
    # y_vec = np.arange(-5.01, 5.01, 0.25)
    # X, Y = np.meshgrid(x_vec, y_vec)
    # Z_meshgrid = np.sin(X ** 2 + Y ** 2)
    # f_interpolation_function = scipy.interpolate.interp2d(x_vec, y_vec, Z_meshgrid, kind='cubic')
    # x_vec_new = np.arange(-5.01, 5.01, 1e-2)
    # y_vec_new = np.arange(-5.01, 5.01, 1e-2)
    # znew = f_interpolation_function(x_vec_new, y_vec_new)
    # #(4). scipy.interpolate.interpn:
    # def value_func_3d(x, y, z):
    #     return 2 * x + 3 * y - z
    # x_vec = np.linspace(0, 4, 5)
    # y_vec = np.linspace(0, 5, 6)
    # z_vec = np.linspace(0, 6, 7)
    # coordinates_tuple_given = (x_vec, y_vec, z_vec)
    # values_given = value_func_3d(*np.meshgrid(*coordinates_tuple_given, indexing='ij'))
    # new_point = np.array([2.21, 3.12, 1.15])
    # final_results = scipy.interpolate.interpn(coordinates_tuple_given, values_given, new_point)
    # #(5). cv2.remap:
    #
    # ###############################################


    ### create meshgrid and flattened coordinates array ([x,y,1] basis): ###
    [xx, yy] = np.meshgrid(x_vec, y_vec)
    xy = np.concatenate([np.transpose(numpy_flatten(xx,True,'F')), np.transpose(numpy_flatten(yy,True,'F')), np.ones((1, len(numpy_flatten(yy,True,'F'))))], 0)

    ### 3X3 matrix transformation: ###
    A = H_matrix
    A[-1, -1] = 1

    ### new coordinates: ###
    xy_prime = np.matmul(A, xy)

    ### division due to homogenous coordinates: ###
    #TODO: if we do not use homography THERE'S NO NEED TO CALCULATE xy_prime[2,:] and so the above is also not relevant and it simply becomes a matter of affine warp
    if transform_string == 'homography':
        xy_prime[0, :] = xy_prime[0, :] / xy_prime[2, :]  #element-wise
        xy_prime[1, :] = xy_prime[1, :] / xy_prime[2, :]

    ### Ignore third row: ###
    xy_prime = xy_prime[0:2, :]

    ### Turn to float32 instead of float64: ###
    xy_prime = xy_prime.astype(float32)

    ### Subpixel interpolation: ###
    # out = cv2.remap(input_image, np.reshape(xy_prime[0,:]+1, (H,W)), np.reshape(xy_prime[1,:]+1, (H,W)), cv2.INTER_CUBIC)
    final_X_grid = np.reshape(xy_prime[0,:], (H,W), order='F')
    final_Y_grid = np.reshape(xy_prime[1,:], (H,W), order='F')
    if interpolation_method == 'linear':
        out = cv2.remap(input_image, final_X_grid, final_Y_grid, cv2.INTER_LINEAR)
    elif interpolation_method == 'nearest':
        out = cv2.remap(input_image, final_X_grid, final_Y_grid, cv2.INTER_NEAREST)

    ### Make sure to output the same number of dimensions as input: ###
    if len(out.shape) == 2 and len(input_image.shape) == 3:
        out = numpy_unsqueeze(out, -1)

    return out



def image_jacobian_numpy(gx, gy, jac, number_of_parameters):
    #%G = image_jacobian_numpy(GX, GY, JAC, number_of_parameters)
    # % This function computes the jacobian G of the warped image wrt parameters.
    # % This matrix depends on the gradient of the warped image, as
    # % well as on the jacobian JAC of the warp transform wrt parameters.
    # % For a detailed definition of matrix G, see the paper text.
    # %
    # % Input variables:
    # % GX:           the warped image gradient in x (horizontal) direction,
    # % GY:           the warped image gradient in y (vertical) direction,
    # % JAC:            the jacobian matrix J of the warp transform wrt parameters,
    # % number_of_parameters:          the number of parameters.
    # %
    # % Output:
    # % G:            The jacobian matrix G.
    #
    
    ### Get image shape: ###
    if len(gx.shape) == 2:
        h, w = gx.shape
    elif len(gx.shape) == 3:
        h, w, c = gx.shape
    
    ### Repeat image gradients by the number of parameters: ###
    gx = np.concatenate([gx]*number_of_parameters, -1)
    gy = np.concatenate([gy]*number_of_parameters, -1)
    # gx = np.repeat(gx, number_of_parameters, -1)
    # gy = np.repeat(gy, number_of_parameters, -1)
    
    ### Get Jacobian of warped image with respect to parameters (chain rule i think): ###
    G = gx * jac[0:h, :] + gy * jac[h:, :]
    G = np.reshape(G, (h * w, number_of_parameters), order='F')
    return G


def correct_H_matrix_for_coming_level_numpy(H_matrix_in, transform_string, high_flag):
    #%H_matrix=correct_H_matrix_for_coming_level_numpy(H_matrix_in, transform_string, HIGH_FLAG)
    # % This function modifies appropriately the WARP values in order to apply
    # % the warp in the next level. If HIGH_FLAG is equal to 1, the function
    # % makes the warp appropriate for the next level of higher resolution.
    # If HIGH_FLAG is equal to 0, the function makes the warp appropriate for the previous level of lower resolution.
    # %
    # % Input variables:
    # % H_matrix_in:      the current warp transform,
    # % transform_string:    the type of adopted transform, accepted strings:
    # %               'tranlation','affine' and 'homography'.
    # % HIGH_FLAG:    The flag which defines the 'next' level. 1 means that the
    # %               the next level is a higher resolution level,
    # %               while 0 means that it is a lower resolution level.
    # % Output:
    # % H_matrix:         the next-level warp transform

    H_matrix = H_matrix_in
    if high_flag == 'higher_resolution':
        if transform_string == 'homography':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] * 2
            H_matrix[-1, 0:2] = H_matrix[-1, 0:2] / 2


        if transform_string == 'affine':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] * 2

        if transform_string == 'translation':
            H_matrix = H_matrix * 2

        if transform_string == 'euclidean':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] * 2

    elif high_flag == 'lower_resolution':
        if transform_string == 'homography':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] / 2
            H_matrix[-1, 0:2] = H_matrix[-1, 0:2] * 2

        if transform_string == 'affine':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] / 2

        if transform_string == 'translation':
            H_matrix = H_matrix / 2

        if transform_string == 'euclidean':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] / 2
    
    return H_matrix


def update_transform_params_numpy(H_matrix_in, delta_p, transform_string):
    #% H_matrix_out=update_transform_params_numpy(H_matrix_in,DELTA_P,transform_string)
    # % This function updates the parameter values by adding the correction values
    # % of DELTA_P to the current warp values in H_matrix_in.
    # %
    # % Input variables:
    # % H_matrix_in:      the current warp transform,
    # % DELTA_P:      the current correction parameter vector,
    # % transform_string:    the type of adopted transform, accepted strings:
    # %               {'translation','euclidean','affine','homography'}.
    # % Output:
    # % H_matrix:         the new (updated) warp transform

    if transform_string == 'homography':
        delta_p = np.concatenate([delta_p, np.zeros((1,1))], 0) #TODO: understand what's this
        H_matrix_out = H_matrix_in + np.reshape(delta_p, (3,3), order='F')
        H_matrix_out[2,2] = 1

    if transform_string == 'affine':
        H_matrix_out = np.zeros((2,3))
        H_matrix_out[0:2, :] = H_matrix_in[0:2, :] + np.reshape(delta_p, (2,3))
        H_matrix_out = np.concatenate([H_matrix_out, np.zeros((1,3))], 0)
        H_matrix_out[2,2] = 1

    if transform_string == 'translation':
        H_matrix_out = H_matrix_in + delta_p

    if transform_string == 'euclidean':
        theta = sign(H_matrix_in[1,0]) * np.arccos(H_matrix_in[0,0]) + delta_p[0]
        tx = H_matrix_in[0,2] + delta_p[1]
        ty = H_matrix_in[1,2] + delta_p[2]
        H_matrix_out = np.eye(3)
        H_matrix_out[0,:] = np.array([np.cos(theta), -sin(theta), tx])
        H_matrix_out[1,:] = np.array([np.sin(theta), cos(theta), ty])
    
    return H_matrix_out


def ECC_numpy(input_tensor, reference_tensor, number_of_levels, number_of_iterations_per_level, transform_string, delta_p_init=None):
    # %ECC image alignment algorithm
    # %[RESULTS, H_matrix, WARPEDiMAGE] = ECC(input_tensor, reference_tensor, number_of_levels, number_of_iterations_per_level, transform_string, DELTA_P_INIT)
    # %
    # % This m-file implements the ECC image alignment algorithm as it is
    # % presented in the paper "G.D.Evangelidis, E.Z.Psarakis, Parametric Image Alignment
    # % using Enhanced Correlation Coefficient.IEEE Trans. on PAMI, vol.30, no.10, 2008"
    # %
    # % ------------------
    # % Input variables:
    # % input_tensor:        the profile needs to be warped in order to be similar to reference_tensor,
    # % reference_tensor:     the profile needs to be reached,
    # % number_of_iterations_per_level:          the number of iterations per level; the algorithm is executed
    # %               (number_of_iterations_per_level-1) times
    # % number_of_levels:       the number of number_of_levels in pyramid scheme (set number_of_levels=1 for a
    # %               non pyramid implementation), the level-index 1
    # %               corresponds to the highest (original) image resolution
    # % transform_string:    the image transformation {'translation', 'euclidean', 'affine', 'homography'}
    # % DELTA_P_INIT: the initial transformation matrix for original images (optional); The identity
    # %               transformation is the default value (see 'transform initialization'
    # %               subroutine in the code). In case of affine or euclidean transform,
    # %               DELTA_P_INIT must be a 2x3 matrix, in homography case it must be a 3x3 matrix,
    # %               while with translation transform it must be a 2x1 vector.
    # %
    # % For example, to initialize the warp with a rotation by x radians, DELTA_P_INIT must
    # % be [cos(x) sin(x) 0 ; -sin(x) cos(x) 0] for affinity
    # % [cos(x) sin(x) 0 ; -sin(x) cos(x) 0 ; 0 0 1] for homography.
    # %
    # %
    # % Output:
    # %
    # % RESULTS:   A struct of size number_of_levels x number_of_iterations_per_level with the following fields:
    # %
    # % RESULTS(m,n).H_matrix:     the warp needs to be applied to IMAGE at n-th iteration of m-th level,
    # % RESULTS(m,n).rho:      the enhanced correlation coefficient value at n-th iteration of m-th level,
    # % H_matrix :              the final estimated transformation [usually also stored in RESULTS(1,number_of_iterations_per_level).H_matrix ].
    # % WARPEDiMAGE:        the final warped image (it should be similar to reference_tensor).
    # %
    # % The first stored .H_matrix and .rho values are due to the initialization. In
    # % case of poor final alignment results check the warp initialization
    # % and/or the overlap of the images.

    # %% transform initialization
    # % In case of translation transform the initialiation matrix is of size 2x1:
    # %  delta_p_init = [p1;
    # %                  p2]
    # % In case of affine transform the initialiation matrix is of size 2x3:
    # %
    # %  delta_p_init = [p1, p3, p5;
    # %                  p2, p4, p6]
    # %
    # % In case of euclidean transform the initialiation matrix is of size 2x3:
    # %
    # %  delta_p_init = [p1, p3, p5;
    # %                  p2, p4, p6]
    # %
    # % where p1=cos(theta), p2 = sin(theta), p3 = -p2, p4 =p1
    # %
    # % In case of homography transform the initialiation matrix is of size 3x3:
    # %  delta_p_init = [p1, p4, p7;
    # %                 p2, p5, p8;
    # %                 p3, p6,  1]

    ### Initialize Parameters: ###
    break_flag = 0
    transform_string = str.lower(transform_string)
    H_input, W_input, C_input = input_tensor.shape
    H_reference, W_reference, C_reference = reference_tensor.shape
    
    ### Initialize New Images For Algorithm To Change: ###
    initImage = input_tensor
    initTemplate = reference_tensor
    input_tensor = RGB2BW(input_tensor).astype(float).squeeze()
    reference_tensor = RGB2BW(reference_tensor).astype(float).squeeze()
    reference_tensor_output_list = [0] * number_of_levels
    input_tensor_output_list = [0] * number_of_levels

    # pyramid images
    # The following for-loop creates pyramid images in cells current_level_input_tensor and reference_tensor_output_list with varying names
    # The variables input_tensor_output_list1} and reference_tensor_output_list{1} are the images with the highest resolution

    ### Smoothing of original images: ###
    #TODO: in the matlab version they overwrite the gaussian blur, so they're not actually blurring anything
    # reference_tensor_output_list[0] = cv2.GaussianBlur(reference_tensor, [7,7], 0.5)
    # input_tensor_output_list[0] = cv2.GaussianBlur(input_tensor, [7,7], 0.5)
    reference_tensor_output_list[0] = reference_tensor
    input_tensor_output_list[0] = input_tensor
    for level_index in np.arange(1, number_of_levels):
        H,W = input_tensor_output_list[level_index - 1].shape
        input_tensor_output_list[level_index] = cv2.resize(input_tensor_output_list[level_index - 1], dsize=(W//2, H//2))
        reference_tensor_output_list[level_index] = cv2.resize(reference_tensor_output_list[level_index - 1], dsize=(W//2, H//2))
    
    
    ### Initialize H_matrix matrix: ###
    #(1). Translation:
    if transform_string == 'translation':
        number_of_parameters = 2 #number of parameters
        if delta_p_init is None:
            H_matrix = np.zeros(2, 1)
        else:
            H_matrix = delta_p_init
    #(2). Euclidean:
    elif transform_string == 'euclidean':
        number_of_parameters = 3 #number of parameters
        if delta_p_init is None:
            H_matrix = np.eye(3)
            H_matrix[-1, -1] = 0
        else:
            H_matrix = np.concatenate([delta_p_init, np.zeros(1, 3)], 0)
    #(3). Affine:
    elif transform_string == 'affine':
        number_of_parameters = 6 #number of parameters
        if delta_p_init is None:
            H_matrix = np.eye(3)
            H_matrix[-1, -1] = 0
        else:
            H_matrix = np.concatenate([delta_p_init, np.zeros((1,3))], 0)
    #(4). Homography:
    elif transform_string == 'homography':
        number_of_parameters = 8 #number of parameters
        if delta_p_init is None:
            H_matrix = np.eye(3)
        else:
            H_matrix = delta_p_init


    ### in case of pyramid implementation, the initial transformation must be appropriately modified: ###
    for level_index in np.arange(0, number_of_levels-1):
        H_matrix = correct_H_matrix_for_coming_level_numpy(H_matrix, transform_string, 'lower_resolution')


    ### Run ECC algorithm for each level of the pyramid: ###
    for level_index in np.arange(number_of_levels, 0, -1):  #start with lowest resolution (highest level of the pyramid)
        ### Get Current Level input_tensor and reference_tensor: ###
        current_level_input_tensor = input_tensor_output_list[level_index-1]
        current_level_reference_tensor = reference_tensor_output_list[level_index-1]
        if len(current_level_reference_tensor.shape) == 3:
            A, B, C = current_level_reference_tensor.shape
            H, W, C = current_level_reference_tensor.shape
        elif len(current_level_reference_tensor.shape) == 2:
            A, B = current_level_reference_tensor.shape
            H, W = current_level_reference_tensor.shape

        ### Get input_tensor gradients: ###
        [vy, vx] = np.gradient(current_level_input_tensor, axis=[0,1])
        # imshow(vx)

        ### Define the rectangular Region of Interest (ROI) by x_vec and y_vec (you can modify the ROI): ###
        # Here we just ignore some image margins.
        # Margin is equal to 5 percent of the mean of [height,width].
        m0 = mean([A, B])
        # margin = floor(m0 * .05 / (2 ** (level_index - 1)))
        margin = 0 # no - margin - modify these two lines if you want to exclude a margin
        x_vec = np.arange(margin, B - margin)
        y_vec = np.arange(margin, A - margin)
        current_level_reference_tensor = current_level_reference_tensor[margin:A-margin, margin:B-margin].astype(float)


        ### ECC, Forward Additive Algorithm: ###
        for iteration_index in np.arange(number_of_iterations_per_level):
            print('Level: ' + str(level_index) + ', Iteration: ' + str(iteration_index))
            wim = spatial_interpolation_numpy(current_level_input_tensor, H_matrix, 'linear', transform_string, x_vec, y_vec, H, W)  # inverse(backward) warping

            ### define a mask to deal with warping outside the image borders: ###
            # (they may have negative values due to the subtraction of the mean value)
            #TODO: there must be an easier way to do this!!! no way i need all these calculations
            ones_map = spatial_interpolation_numpy(np.ones_like(current_level_input_tensor), H_matrix, 'nearest', transform_string, x_vec, y_vec, H, W) # inverse(backward) warping
            numOfElem = (ones_map != 0).sum()

            meanOfWim = (wim * (ones_map!=0)).sum() / numOfElem
            meanOfTemp = (current_level_reference_tensor * (ones_map!=0)).sum() / numOfElem

            wim = wim - meanOfWim # zero - mean image; is useful for brightness change compensation, otherwise you can comment this line
            tempzm = current_level_reference_tensor - meanOfTemp # zero - mean reference_tensor

            wim[ones_map == 0] = 0 # for pixels outside the overlapping area
            tempzm[ones_map == 0] = 0

            # ### Save current transform: ###
            # # TODO: find an appropriate data structure / object for this
            # if transform_string == 'affine' or transform_string == 'euclidean':
            #     results[level_index, iteration_index].H_matrix = H_matrix[0:2, :]
            # else:
            #     results[level_index, iteration_index].H_matrix = H_matrix
            # results[level_index, iteration_index].rho = dot(current_level_reference_tensor[:], wim[:]) / norm(tempzm[:]) / norm(wim[:])

            ### Break the loop if reached max number of iterations per level: ###
            if iteration_index == number_of_iterations_per_level:  #the algorithm is executed (number_of_iterations_per_level-1) times
                break

            ### Gradient Image interpolation (warped gradients): ###
            wvx = spatial_interpolation_numpy(vx, H_matrix, 'linear', transform_string, x_vec, y_vec, H, W)
            wvy = spatial_interpolation_numpy(vy, H_matrix, 'linear', transform_string, x_vec, y_vec, H, W)

            ### Compute the jacobian of warp transform_string: ###
            J = get_jacobian_for_warp_transform_numpy(x_vec+1, y_vec+1, H_matrix, transform_string, H, W)
            
            ### Compute the jacobian of warped image wrt parameters (matrix G in the paper): ###
            G = image_jacobian_numpy(wvx, wvy, J, number_of_parameters)
            
            ### Coompute Hessian and its inverse: ###
            C = np.matmul(np.transpose(G), G)  #matrix multiplication, C = Hessian matrix
            cond = np.linalg.cond(C)
            i_C = np.linalg.inv(C)
            
            ### Compute projections of images into G: ###
            Gt = np.transpose(G) @ numpy_flatten(tempzm, True, 'F')
            Gw = np.transpose(G) @ numpy_flatten(wim, True, 'F')
            
            ### ECC Closed Form Solution: ###
            #(1). compute lambda parameter:
            num = (np.linalg.norm(numpy_flatten(wim, True, 'F')))**2 - np.transpose(Gw) @ i_C @ Gw
            den = (dot(numpy_flatten(tempzm, True, 'F').squeeze(), numpy_flatten(wim, True, 'F').squeeze())) - np.transpose(Gt) @ i_C @ Gw
            lambda_correction = num / den
            #(2). compute error vector:
            imerror = lambda_correction * tempzm - wim
            #(3). compute the projection of error vector into Jacobian G:
            Ge = np.transpose(G) @ numpy_flatten(imerror, True, 'F')
            #(4). compute the optimum parameter correction vector:
            delta_p = np.matmul(i_C, Ge)
            
            ### Update Parameters: ###
            H_matrix = update_transform_params_numpy(H_matrix, delta_p, transform_string)

            print(H_matrix)
            # print(delta_p)

        ### END OF INTERNAL LOOP
        
        ### break loop of reached errors: ###
        if break_flag == 1:
            break
            
        ### modify the parameters appropriately for next pyramid level: ###
        if level_index>1 and break_flag==0:
            H_matrix = correct_H_matrix_for_coming_level_numpy(H_matrix, transform_string, 'higher_resolution')
    
    ### END OF PYRAMID number_of_levels LOOP:
        
    
    # ### this conditional part is only executed when algorithm stops due to Hessian singularity: ###
    # if break_flag == 1:
    #     for jj in np.arange(level_index-1):
    #         H_matrix = correct_H_matrix_for_coming_level_numpy(H_matrix, transform_string, 'higher_resolution')
    
    ### Get final H_matrix: ###
    final_warp = H_matrix
    
    ### return the final warped image using the whole support area (including margins): ###
    nx2 = np.arange(0, B)
    ny2 = np.arange(0, A)
    warpedImage = np.zeros_like(initImage)
    for ii in np.arange(C_input):
        warpedImage[:, :, ii] = spatial_interpolation_numpy(initImage[:, :, ii], final_warp, 'linear', transform_string, nx2, ny2, H, W)
    H_matrix = final_warp
    
    return H_matrix, warpedImage
    
    
def ECC_demo_numpy():
    # ### Read Input: ###
    # reference_tensor = read_image_default_torch() * 255
    # reference_tensor = RGB2BW(reference_tensor).type(torch.uint8)
    # ### Warp Input: ###
    # input_tensor = shift_matrix_subpixel_torch(reference_tensor, 1, 1)

    # path1 = r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/image_1_before_homography.png'
    # path2 = r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/image_1_after_homography.png'
    path1 = r'/home/dudy/Projects/data/pngsh/pngs/2/01789.png'
    path2 = r'/home/dudy/Projects/data/pngsh/pngs/2/02134.png'
    reference_tensor = read_image_torch(path1)
    input_tensor = read_image_torch(path2)

    ### Other Parameters: ###
    transform_string = 'homography'
    # delta_p_init = None
    delta_p_init = np.eye(3)
    delta_p_init[0, -1] = 0
    delta_p_init[1, -1] = 0


    ### Perform ECC Alignment: ###
    input_tensor = RGB2BW(input_tensor[0]).permute([1,2,0]).cpu().numpy().astype(np.uint8).astype(float)
    reference_tensor = RGB2BW(reference_tensor[0]).permute([1,2,0]).cpu().numpy().astype(np.uint8).astype(float)
    H_matrix, warpedImage = ECC_numpy(reference_tensor,
                                               input_tensor,
                                               number_of_levels=1,
                                               number_of_iterations_per_level=50,
                                               transform_string=transform_string,
                                               delta_p_init=delta_p_init)

    1
    # aligned_tensor = scipy.ndimage.affine_transform(input_tensor, H_matrix)

    imshow(BW2RGB(warpedImage)/255)
    plt.figure();
    imshow(BW2RGB(input_tensor)/255)
    plt.figure();
    imshow(BW2RGB(reference_tensor)/255)
    plt.figure();
    imshow(warpedImage-input_tensor)

    bla = warpedImage - input_tensor
    bla = crop_numpy_batch(bla, (350,950))
    imshow(bla)
    imshow(bla>10)

ECC_demo_numpy()


        
        
            
            
            
            
    