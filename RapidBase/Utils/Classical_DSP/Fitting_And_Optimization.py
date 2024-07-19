


import numpy as np
from RapidBase.Utils.Tensor_Manipulation.linspace_arange import my_linspace
import matplotlib.pyplot as plt
from RapidBase.Utils.IO.Imshow_and_Plots import imshow_torch, plot_torch
from RapidBase.Utils.IO.tic_toc import tic, toc
from RapidBase.Utils.Classical_DSP.Convolution_Utils import convn_torch
from torch import Tensor
from typing import Tuple
########################

def initial_BB_from_time_average_and_blob_detection(input_tensor, flag_use_square=False):
    ### Get Time Average: ###
    if flag_use_square:
        input_tensor_time_average = (input_tensor ** 2).mean(0, True)
    else:
        input_tensor_time_average = input_tensor.mean(0, True)

    ### Estimate BB Size Using Blob Detection (assumes circular blob): ###
    blobs_list = blob_detection_scipy_DifferenceOfGaussian_torch(
        scale_array_to_range(input_tensor_time_average, (0, 255)), min_sigma=0.3, max_sigma=10, threshold=0.5, flag_return_image_with_circles=False)
    x_center, y_center, bla, r = blobs_list[0]
    BB_size = np.ceil(2 * r * np.sqrt(2)) + 1
    BB_BD_X_start = x_center - np.ceil(BB_size / 2)
    BB_BD_Y_start = y_center - np.ceil(BB_size / 2)
    BB_BD_W = BB_size
    BB_BD_H = BB_size
    BB_BD_tuple = [BB_BD_X_start, BB_BD_Y_start, BB_BD_X_start + BB_BD_W, BB_BD_Y_start + BB_BD_H]
    return BB_BD_tuple


def COM_for_heuristics(TrjMov: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    ### Square the TrjMov to get more accurate assessment of the COM: ###
    #WI: figure out if this code is duplicated
    bgs_squared_trj_frames = TrjMov ** 2
    bgs_squared_trj_frames = bgs_squared_trj_frames
    torch.cuda.empty_cache()

    ### Normalize Input Tensor Before Getting Statistics: ###
    input_tensor_sum = bgs_squared_trj_frames.sum([-1, -2], True)
    input_tensor_normalized = bgs_squared_trj_frames / (input_tensor_sum + 1e-6)
    H, W = input_tensor_normalized.shape[-2:]
    x_vec = torch.arange(W)
    y_vec = torch.arange(H)
    x_vec = x_vec - W // 2
    y_vec = y_vec - H // 2
    x_vec = x_vec.to(input_tensor_normalized.device)
    y_vec = y_vec.to(input_tensor_normalized.device)
    x_grid = x_vec.unsqueeze(0).repeat(H, 1)
    y_grid = y_vec.unsqueeze(1).repeat(1, W)
    x_grid = x_grid.unsqueeze(0).unsqueeze(0)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)
    cx = (input_tensor_normalized * x_grid).sum([-1, -2], True).squeeze()
    cy = (input_tensor_normalized * y_grid).sum([-1, -2], True).squeeze()
    x_grid_batch = x_grid.repeat(cx.shape[0], 1, 1, 1)
    y_grid_batch = y_grid.repeat(cy.shape[0], 1, 1, 1)
    x_grid_batch_modified = (x_grid_batch - cx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
    y_grid_batch_modified = (y_grid_batch - cy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

    cx2_modified = (input_tensor_normalized * x_grid_batch_modified ** 2).sum([-1, -2], True).squeeze().sqrt()
    cy2_modified = (input_tensor_normalized * y_grid_batch_modified ** 2).sum([-1, -2], True).squeeze().sqrt()
    return cx, cy, cx2_modified, cy2_modified


def initial_BB_from_time_average_max_location_and_MOI(input_tensor: Tensor, use_square_for_time_avg=False):
    ### Get Time Average: ###
    if use_square_for_time_avg:
        input_tensor_time_average = (input_tensor ** 2).mean(0, True)
    else:
        input_tensor_time_average = input_tensor.mean(0, True)
    H_max, W_max = get_max_2D_image_torch(input_tensor_time_average)

    ### Get COM and MOI Stats: ###
    cx, cy, cx2_modified, cy2_modified = COM_for_heuristics(input_tensor)

    ### Get BB From Stats: ###
    x_center = int(W_max)
    y_center = int(H_max)
    BB_size_X = np.ceil(2 * cx2_modified.mean().item() * np.sqrt(2)) + 1
    BB_size_Y = np.ceil(2 * cy2_modified.mean().item() * np.sqrt(2)) + 1
    BB_MOI_X_start = x_center - np.ceil(BB_size_X / 2)
    BB_MOI_Y_start = y_center - np.ceil(BB_size_Y / 2)
    BB_MOI_W = BB_size_X
    BB_MOI_H = BB_size_Y
    BB_output_tuple = [BB_MOI_X_start, BB_MOI_Y_start, BB_MOI_W, BB_MOI_H]
    return BB_output_tuple


def initial_BB_from_COM_and_MOI(input_tensor: Tensor):
    cx, cy, cx2_modified, cy2_modified = COM_for_heuristics(input_tensor)
    x_center = int(input_tensor.shape[-1] // 2 + cx.mean().item())
    y_center = int(input_tensor.shape[-2] // 2 + cy.mean().item())
    BB_size_X = np.ceil(2 * cx2_modified.mean().item() * np.sqrt(2)) + 1
    BB_size_Y = np.ceil(2 * cy2_modified.mean().item() * np.sqrt(2)) + 1
    BB_MOI_X_start = x_center - np.ceil(BB_size_X / 2)
    BB_MOI_Y_start = y_center - np.ceil(BB_size_Y / 2)
    BB_MOI_W = BB_size_X
    BB_MOI_H = BB_size_Y
    BB_output_tuple = [BB_MOI_X_start, BB_MOI_Y_start, BB_MOI_W, BB_MOI_H]
    return BB_output_tuple

def initial_BB_from_time_average_and_IntensityFallOff(input_tensor: Tensor, flag_use_square=False):
    if flag_use_square:
        input_tensor_time_average = (input_tensor ** 2).mean(0, True).unsqueeze(1)
    else:
        input_tensor_time_average = input_tensor.mean(0, True).abs().unsqueeze(1)
    H_max, W_max = get_max_2D_image_torch(input_tensor_time_average)
    max_value = input_tensor_time_average[0, 0, H_max, W_max]
    threshold_value = max_value / 4
    ### Get Cross-Section in X and Y and determin BB Size: ###
    x_cross_section_vec = input_tensor_time_average[:, :, H_max, :].squeeze()
    y_cross_section_vec = input_tensor_time_average[:, :, :, W_max].squeeze()
    x_cross_section_vec_positive_direction = x_cross_section_vec[W_max + 1:]
    x_cross_section_vec_negative_direction = x_cross_section_vec[0:W_max].flip(0)
    y_cross_section_vec_positive_direction = y_cross_section_vec[H_max + 1:]
    y_cross_section_vec_negative_direction = y_cross_section_vec[0:H_max].flip(0)
    x_cross_section_vec_positive_direction_logical_mask = x_cross_section_vec_positive_direction < threshold_value
    x_cross_section_vec_negative_direction_logical_mask = x_cross_section_vec_negative_direction < threshold_value
    y_cross_section_vec_positive_direction_logical_mask = y_cross_section_vec_positive_direction < threshold_value
    y_cross_section_vec_negative_direction_logical_mask = y_cross_section_vec_negative_direction < threshold_value
    if len(x_cross_section_vec_positive_direction_logical_mask.nonzero()) == 0 or len(
            x_cross_section_vec_negative_direction_logical_mask.nonzero()) == 0:  # 0
        BB_size_X = 4
    else:
        BB_size_X = x_cross_section_vec_positive_direction_logical_mask.nonzero()[0] + \
                    x_cross_section_vec_negative_direction_logical_mask.nonzero()[0] + 1
        BB_size_X = BB_size_X.item()
    if len(y_cross_section_vec_positive_direction_logical_mask.nonzero()) == 0 or len(
            y_cross_section_vec_negative_direction_logical_mask.nonzero()) == 0:
        BB_size_Y = 4
    else:
        BB_size_Y = y_cross_section_vec_positive_direction_logical_mask.nonzero()[0] + \
                    y_cross_section_vec_negative_direction_logical_mask.nonzero()[0] + 1
        BB_size_Y = BB_size_Y.item()
    ### Get BB From Stats: ###
    x_center = int(W_max)
    y_center = int(H_max)
    BB_FWHM_X_start = x_center - np.ceil(BB_size_X / 2)
    BB_FWHM_Y_start = y_center - np.ceil(BB_size_Y / 2)
    BB_FWHM_W = BB_size_X
    BB_FWHM_H = BB_size_Y
    BB_FWHM_tuple = [BB_FWHM_X_start, BB_FWHM_Y_start, BB_FWHM_W, BB_FWHM_H]
    return BB_FWHM_tuple

def get_max_2D_image_torch(input_tensor):
    flat_indexes = input_tensor.flatten(start_dim=-2).argmax(-1)
    max_position_tuple = [divmod(idx.item(), input_tensor.shape[-1]) for idx in flat_indexes]
    return max_position_tuple[0]



def get_cx_and_cy(trajectory_through_bgs: Tensor):
    ### Normalize Input Tensor Before Getting Statistics
    input_tensor_sum = trajectory_through_bgs.sum([-1, -2], True)
    input_tensor_normalized = trajectory_through_bgs / (input_tensor_sum + 1e-6)
    H, W = input_tensor_normalized.shape[-2:]
    x_vec = torch.arange(W)
    y_vec = torch.arange(H)
    x_vec = x_vec - W // 2
    y_vec = y_vec - H // 2
    x_vec = x_vec.to(input_tensor_normalized.device)
    y_vec = y_vec.to(input_tensor_normalized.device)
    x_grid = x_vec.unsqueeze(0).repeat(H, 1)
    y_grid = y_vec.unsqueeze(1).repeat(1, W)
    cx = (input_tensor_normalized * x_grid).sum([-1, -2], True).squeeze()
    cy = (input_tensor_normalized * y_grid).sum([-1, -2], True).squeeze()
    # if cx.abs().max() > 15 or cy.abs().max() > 15:
    #     bla = 1
    # else:
    #     # print(cx.abs().max())
    #     # print(cy.abs().max())
    #     bl=1
    return cx, cy


#########################




import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import torch
from torch import Tensor

def np_get_diag_inds(diag, mat_sz):
    # the smallet value for the first index, if we are in the lower part of the matrix choose -diag, else 0
    # the smallet value for the second index, if we are in the lower part of the matrix choose 0, else diag
    ind_start = np.array([max(-diag, 0), max(diag, 0)], dtype=np.uint)

    # how many steps can we take along the diagonal? well we stop once we hit any edge (note i'm not assuming a squre matrix where things are easy)i.e
    total_diag_length = np.min(mat_sz - ind_start)

    return np.vstack((np.arange(ind_start[0], ind_start[0] + total_diag_length, dtype=np.uint),
                      np.arange(ind_start[1], ind_start[1] + total_diag_length, dtype=np.uint)))
    # first_ind_end = min(mat_sz[0], first_ind_start+ ) # last valus of the first index
    # first_ind = np.arange( max(-diag,0), diag + )


def torch_get_diag_inds(diag, mat_sz, cuda_device='cpu'):
    # the smallet value for the first index, if we are in the lower part of the matrix choose -diag, else 0
    # the smallet value for the second index, if we are in the lower part of the matrix choose 0, else diag
    ind_start = np.array([max(-diag, 0), max(diag, 0)], dtype=np.int32)

    # how many steps can we take along the diagonal? well we stop once we hit any edge (note i'm not assuming a squre matrix where things are easy)i.e
    total_diag_length = np.min(mat_sz - ind_start)
    # note that these preliminary calcs are done on the cpu only the inds are build on the gpu

    return torch.vstack(
        (torch.arange(ind_start[0], ind_start[0] + total_diag_length, dtype=torch.long, device=cuda_device),
         torch.arange(ind_start[1], ind_start[1] + total_diag_length, dtype=torch.long, device=cuda_device)))
    # first_ind_end = min(mat_sz[0], first_ind_start+ ) # last v


def make_1d_spline(t, Y, smooth=1, W=None):
    # this routine make a smoothing cubic spline as documented by
    # https://en.wikipedia.org/wiki/Smoothing_spline,
    #  De Boor, C. (2001). A Practical Guide to Splines (Revised Edition)
    # and the article : Reinsch, Christian H (1967). "Smoothing by Spline Functions". Numerische Mathematik. 10 (3): 177–183. doi:10.1007/BF02162161
    # which gives a more detailed derivation of the matrices involved (also see my lyx note for a long full derivtion of this shit)
    # note that there are two similar formulations as detailed in wikipedia. the matrices used in them are almost idential they differ by a p (1-p) factor)
    # i.e we use here we use "De Boor's approach", as detailed in wikipedia

    # the construction of the smoothing spline involves solving a linear system on the coefficients  of the linear system detialed in

    # inputs
    # x: a 1d tensor of length N detailing the "t" position of the data. note that i use t, as let's say in parametric 3d curve this some parameter that parmetrizes the curve
    # i.e we have (x(t), y(y), z(t))
    # a tensor whos last dim hase length N. The spline  and whose other dims we are fitting.
    #
    # y: a 2d tensor of the data to fit along the last dimention i.e the length of the last dimention must be eqaul to size x
    #
    # W: optinal paramter : the weights vector of the same length as x. It is defined in a different way than De Boor,
    #    in my deffintion this the wieght of each point is w^2*(y-a)^2, where y is the value of the data at the point
    #    a is the fitted value of the data, hence the wieght of points increses with w. this is in contrast to De Boor these are 1. / sqrt(w) of my deffintions,
    #    which to me is counter intuetive
    #
    # smooth: the smoothing  paramter. where 0 is no smooth and 1 is compelte smoth (which makes a fit that doesnt take the y data into account at all)
    #         this is differnt from the  original csaps as there 1 is no smothing, which is cunter intutive to me.
    #
    #
    cuda_device = t.device

    pcount = t.shape[0]
    dt = torch.diff(t)
    # this is the (sparse) matrix R as defined in De Boor, C. (2001). A Practical Guide to Splines (Revised Edition)
    R_diag_inds = torch.hstack((torch_get_diag_inds(-1, [pcount - 2, pcount - 2], cuda_device),
                                torch_get_diag_inds(0, [pcount - 2, pcount - 2], cuda_device),
                                torch_get_diag_inds(1, [pcount - 2, pcount - 2], cuda_device)))
    R = torch.sparse_coo_tensor(R_diag_inds, torch.hstack(
        (dt[1:-1], 2 * (dt[1:] + dt[:-1]), dt[1:-1])), (pcount - 2, pcount - 2))
    # we now caclute the matrix Q_transpose (i.e Qt is Q transpose) as defined in De Boor, C. (2001). A Practical Guide to Splines (Revised Edition)
    dt_recip = 1. / dt
    Q_diag_inds = torch.hstack((torch_get_diag_inds(0, [pcount - 2, pcount], cuda_device),
                                torch_get_diag_inds(1, [pcount - 2, pcount], cuda_device),
                                torch_get_diag_inds(2, [pcount - 2, pcount], cuda_device)))
    Qt = torch.sparse_coo_tensor(Q_diag_inds, torch.hstack(
        (dt_recip[:-1], - (dt_recip[1:] + dt_recip[:-1]), dt_recip[1:])), (pcount - 2, pcount))
    if (W is not None):
        QwQ = Qt @ torch.sparse_coo_tensor(torch_get_diag_inds(0,
                                                               [pcount, pcount], cuda_device), W)
        QwQ = QwQ @ (QwQ.t())
    else:
        QwQ = Qt.to_dense() @ (Qt.t().to_dense())
        #
        # try:
        #     QwQ = Qt @ (Qt.t())
        # except RuntimeError:
        #     pass
    p = 1 - smooth
    # we can now start solving for the coffesiants of thr polinomuial f( xi+ z )= ai +bi*z+ci*z^2+di*z^3
    # Solve linear system for the 2nd derivatives
    # Qt.mm(Y.t())
    # c is the vector of 2nd derivaties
    c = torch.empty((Y.shape[0], Y.shape[1]), dtype=Y.dtype, device=cuda_device)

    # c[:, 1:-1] = torch.linalg.solve((6. * (1. - p)
    #                                  * QwQ + p * R).to_dense(), 3 * p * Qt.mm(Y.t())).T
    c[:, 1:-1] = torch.linalg.solve((6. * (1. - p)
                                     * QwQ + p * R), 3 * p * Qt.mm(Y.t())).T
    c[:, 0] = 0
    c[:, -1] = 0

    # see my (AKA yuri) attached lyx notes as to how to get a from c
    if (W is not None):
        a = Y - W * ((((2 * (1 - p) / p) * Qt.t()) @ (c[:, 1:-1].T)).T)
    else:
        a = Y - ((((2 * (1 - p) / p) * Qt.t()) @ (c[:, 1:-1].T)).T)

    # d is simply cacalued by the finite differnce equation for the 2 derivative.
    d = torch.diff(c, 1) / dt
    # b can by cacluted from the eqation  f( xi+ z )= ai +bi*z+ci*z^2+di*z^3, where we demnad that the 1st derivtive is contiues
    b = torch.diff(a) / dt - dt * (c[:, :-1] + dt * d)
    return torch.stack([a[:, :-1], b, c[:, :-1], d])


def eval_1d_spline(t, Coefficients, t_interp, deriv=0):
    interp_inds = torch.searchsorted(t[:-1], t_interp, side="right") - 1
    interp_inds[interp_inds < 0] = 0
    dist_to_grid_point = t_interp - t[interp_inds]

    if (deriv == 0):
        interpulated_values = Coefficients[0, ..., interp_inds] + \
                              dist_to_grid_point[None,] * (Coefficients[1, ..., interp_inds] + \
                                                           dist_to_grid_point[None,] * (
                                                                       Coefficients[2, ..., interp_inds] + \
                                                                       dist_to_grid_point[None,] * Coefficients[
                                                                           3, ..., interp_inds]))

    elif (deriv == 1):
        interpulated_values = (Coefficients[1, ..., interp_inds] + \
                               dist_to_grid_point[None,] * (2 * Coefficients[2, ..., interp_inds] + \
                                                            3 * dist_to_grid_point[None,] * Coefficients[
                                                                3, ..., interp_inds]))
    elif (deriv == 2):
        interpulated_values = 2 * Coefficients[2, ..., interp_inds] + \
                              6 * dist_to_grid_point[None,] * Coefficients[3, ..., interp_inds]

    elif (deriv == 3):
        interpulated_values = 6 * Coefficients[3, ..., interp_inds]

    else:
        interpulated_values = torch.zeros_like(Coefficients[0, ..., interp_inds])

    return interpulated_values


def csaps_smoothn_torch(t_original=None, Y=None, t_final=None, smoothing=0, W=None):
    ### make sure [Y] is at least [B,N]: ###
    if len(Y.shape) == 1:
        Y = Y.unsqueeze(0)

    ### Create simple linear grid vec is None is input: ###
    if t_original is None:
        t_original = torch.arange(Y.shape[1]).to(Y.device)
    if t_final is None:
        t_final = t_original

    ### Get Coefficients: ###
    Coefficients = make_1d_spline(t_original, Y, smooth=smoothing, W=W)

    ### Get Final Smooth Estimate: ###
    Y_smooth = eval_1d_spline(t_original, Coefficients, t_final, deriv=0)

    return Y_smooth


def simple_linear_expanded_smoothing(vec: Tensor, expanded_dim_0: int) -> Tensor:
    vec = vec.squeeze()
    initial_dim = vec.shape[0]

    if len(vec.shape) > 1:
        raise RuntimeError("Only works for 1D vectors")
    if ((expanded_dim_0) % initial_dim  != 0): # really shouldbe -1
        raise RuntimeError("Expanded dim must be multiple of initial shape")
    repeat = (expanded_dim_0) // initial_dim
    expanded_vec = vec.repeat_interleave(repeat).type(torch.float)
    diffs = vec.diff()
    diffs_expanded = diffs.repeat_interleave(repeat)
    fractions = torch.arange(expanded_dim_0-repeat)%repeat
    fractions = fractions/repeat
    expanded_vec[:expanded_dim_0-repeat] += fractions * diffs_expanded
    return expanded_vec







def polyfit_torch_FitSimplestDegree(x, y, max_degree_to_check=2, flag_get_prediction=True, flag_get_residuals=False):
    poly_residual = []
    for i in np.arange(max_degree_to_check):
        current_poly_degree = i
        coefficients, prediction, residual_values = polyfit_torch(x, y, current_poly_degree, flag_get_prediction=True, flag_get_residuals=True)
        poly_residual.append(residual_values.abs().mean().item() / (len(residual_values) - current_poly_degree - 1)**2)
    poly_residual_torch = torch.tensor(poly_residual)

    ### Choose Degre: ###
    #(1). simply choose the minimum:
    best_polynomial_degree = torch.argmin(poly_residual_torch, dim=0)
    #(2). choose the first index which starts to decrease slowly:
    poly_residual_torch_diff = torch.diff(poly_residual_torch)
    #(3). Fit to a distribution and find it's effective width or something:
    #TODO: do it

    ### Get "Best" Fit: ###
    coefficients, prediction, residual_values = polyfit_torch(x, y, best_polynomial_degree, flag_get_prediction=flag_get_prediction, flag_get_residuals=flag_get_residuals)

    return coefficients, prediction, residual_values, poly_residual_torch, best_polynomial_degree

def polyfit_torch_FitSimplestDegree_parallel(x, y, max_degree_to_check=2, flag_get_prediction=True, flag_get_residuals=False):
    poly_residual = []
    #TODO: make this parallel all the way!!!!
    for i in np.arange(max_degree_to_check):
        current_poly_degree = i
        coefficients, prediction, residual_values = polyfit_torch_parallel(x, y, current_poly_degree, flag_get_prediction=True, flag_get_residuals=True)
        poly_residual.append(residual_values.abs().mean().item() / (len(residual_values) - current_poly_degree - 1)**2)
    poly_residual_torch = torch.tensor(poly_residual)

    ### Choose Degre: ###
    #(1). simply choose the minimum:
    best_polynomial_degree = torch.argmin(poly_residual_torch, dim=0)
    #(2). choose the first index which starts to decrease slowly:
    poly_residual_torch_diff = torch.diff(poly_residual_torch)
    #(3). Fit to a distribution and find it's effective width or something:
    #TODO: do it

    ### Get "Best" Fit: ###
    coefficients, prediction, residual_values = polyfit_torch(x, y, best_polynomial_degree, flag_get_prediction=flag_get_prediction, flag_get_residuals=flag_get_residuals)

    return coefficients, prediction, residual_values, poly_residual_torch, best_polynomial_degree

def polyfit_torch_parallel(x, y, polynom_degree, flag_get_prediction=True, flag_get_residuals=False):
    ####################################################################
    ### New Version - Parallel: ###
    ### Assuming x is a flattened, 1D torch tensor: ###
    A = torch.ones_like(x).to(y.device).float()
    x = x.unsqueeze(-1)
    A = A.unsqueeze(-1)

    ### Polyfit using least-squares solution: ###
    for current_degree in np.arange(1, polynom_degree + 1):
        A = torch.cat((A, (x ** current_degree)), -1)

    ### Solve LSTQ: ###

    # [A] = [Batch, M(number of samples) ,N(number of coefficients)]
    # [y] = [Batch, M(number of samples), k(final vector size, usually 1 for polynom)]
    y = y.unsqueeze(-1)
    A = A.unsqueeze(0)
    returned_solution = torch.linalg.lstsq(A, y)
    coefficients = returned_solution.solution
    rank = returned_solution.rank
    residuals = returned_solution.residuals
    singular_values = returned_solution.singular_values

    ### Predict y using smooth polynom: ###
    if flag_get_prediction:
        x = x.squeeze(-1)
        prediction = 0
        for current_degree in np.arange(0, polynom_degree + 1):
            prediction += coefficients[:, current_degree:current_degree + 1] * x ** current_degree
    else:
        prediction = None

    ### calculate residual: ###
    if flag_get_residuals:
        residual_values = prediction - y
        residual_std = residual_values.std(-1)
    else:
        residual_values = None
        residual_std = None

    # index = 6
    # plot_torch(x[index,:], y[index,:])
    # plot_torch(x[index,:], prediction[index,:])

    return coefficients, prediction, residual_values, residual_std
    ####################################################################

def polyfit_torch(x, y, polynom_degree, flag_get_prediction=True, flag_get_residuals=False):
    # ### Possible Values: ###
    # polynom_degree = 2
    # x = torch.arange(0,5,0.1)
    # y = 1*1 + 2.1*x - 3.3*x**2 + 0.2*torch.randn_like(x)

    # ### Temp: ###
    # full_filename = r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/y.pt'
    # y = torch.load(full_filename)
    # x = torch.arange(len(y)).to(y.device)
    ####################################################################

    ### Old Version - Unparallel: ###
    ### Assuming x is a flattened, 1D torch tensor: ###
    A = torch.ones(len(x)).to(y.device)
    x = x.unsqueeze(-1)
    A = A.unsqueeze(-1)

    ### Polyfit using least-squares solution: ###
    for current_degree in np.arange(1, polynom_degree + 1):
        A = torch.cat((A, (x ** current_degree)), -1)

    ### Perform Least Squares: ###
    returned_solution = torch.linalg.lstsq(A, y)
    # returned_solution_2 = (torch.linalg.inv(A.T @ A) @ A.T) @ y
    coefficients = returned_solution.solution
    rank = returned_solution.rank
    residuals = returned_solution.residuals
    singular_values = returned_solution.singular_values

    ### Predict y using smooth polynom: ###
    if flag_get_prediction:
        x = x.squeeze()
        prediction = 0
        for current_degree in np.arange(0, polynom_degree + 1):
            prediction += coefficients[current_degree] * x ** current_degree
    else:
        prediction = None

    ### calculate residual: ###
    if flag_get_residuals:
        residual_values = prediction - y
        residual_std = residual_values.std()
    else:
        residual_values = None
        residual_std = None
    ####################################################################


    return coefficients, prediction, residual_values



def polyval_torch(coefficients, x):
    x = x.squeeze()
    polynom_degree = len(coefficients) - 1
    prediction = 0
    for current_degree in np.arange(0, polynom_degree + 1):
        prediction += coefficients[current_degree] * x ** current_degree
    return prediction



def fit_polynomial(x, y):
    # solve for 2nd degree polynomial deterministically using three points
    a = (y[2] + y[0] - 2*y[1])/2
    b = -(y[0] + 2*a*x[1] - y[1] - a)
    c = y[1] - b*x[1] - a*x[1]**2
    return [c, b, a]

def fit_polynomial_torch(x, y):
    # solve for 2nd degree polynomial deterministically using three points
    a = (y[0,0,2] + y[0,0,0] - 2*y[0,0,1])/2
    b = -(y[0,0,0] + 2*a*x[1] - y[0,0,1] - a)
    c = y[0,0,1] - b*x[1] - a*x[1]**2
    return [c, b, a]


#TODO: create a pytorch version
def return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(y_vec, x_vec=None):
    # (*). Assumed 1D Input!!!!!!!!!!!!!!!
    ### take care of input: ###
    y_vec = np.ndarray.flatten(y_vec)
    if x_vec is None:
        x_vec = my_linspace(0 ,len(y_vec) ,len(y_vec))
    x_vec = np.ndarray.flatten(x_vec)

    ### get max index around which to interpolate: ###
    max_index = np.argmax(y_vec)
    if max_index == 0:  # TODO: maybe if max is at beginning of array return 0
        indices_to_fit = np.arange(0 ,2)
    elif max_index == len(y_vec) - 1:  # TODO: maybe if max is at end of array return last term
        indices_to_fit = np.arange(len(x_vec ) - 1 -2, len(x_vec ) -1 + 1)
    else:
        indices_to_fit = np.arange(max_index -1, max_index +1 + 1)

    ### Actually Fit: ###
    x_vec_to_fit = x_vec[indices_to_fit]
    y_vec_to_fit = y_vec[indices_to_fit]  # use only 3 points around max to make sub-pixel fit
    P = np.polynomial.polynomial.polyfit(x_vec_to_fit, y_vec_to_fit, 2)
    x_max = -P[1] / (2 * P[2])
    y_max = np.polynomial.polynomial.polyval(x_max ,P)

    return x_max, y_max

def return_shifts_using_parabola_fit_numpy(CC):
    W = CC.shape[-1]
    H = CC.shape[-2]
    max_index = np.argmax(CC)
    max_row = max_index // W
    max_col = max_index % W
    center_row = H // 2
    center_col = W // 2

    if max_row == 0 or max_row == H or max_col == 0 or max_col == W:
        z_max_vec = (0 ,0)
        shifts_total = (0 ,0)
    else:
        # fitting_points_x = CC[max_row:max_row + 1, max_col - 1:max_col + 2]
        # fitting_points_y = CC[max_row - 1:max_row + 2, max_col:max_col + 1]
        fitting_points_x = CC[max_row ,:]
        fitting_points_y = CC[:, max_col]

        ### Use PolyFit: ###
        x_vec = np.arange(-(H // 2), H // 2 + 1)
        shiftx, x_parabola_max = return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(fitting_points_x, x_vec=x_vec)
        shifty, y_parabola_max = return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(fitting_points_y, x_vec=x_vec)
        shiftx = shiftx - (max_row - center_row)
        shifty = shifty - (max_col - center_col)
        shifts_total = (shiftx, shifty)
        z_max_vec = (x_parabola_max, y_parabola_max)

    # ### Fast, minimum amount of operations: ###
    # y1 = fitting_points_x[:,0]
    # y2 = fitting_points_x[:,1]
    # y3 = fitting_points_x[:,2]
    # shiftx = (y1 - y3) / (2 * (y1 + y3 - 2 * y2))
    # y1 = fitting_points_y[0,:]
    # y2 = fitting_points_y[1,:]
    # y3 = fitting_points_y[2,:]
    # shifty = (y1 - y3) / (2 * (y1 + y3 - 2 * y2))
    # shifts_total = (shiftx, shifty)
    return shifts_total, z_max_vec


def return_shifts_using_parabola_fit_torch(CC):
    W = CC.shape[-1]
    H = CC.shape[-2]
    max_index = np.argmax(CC)
    max_row = max_index // W
    max_col = max_index % W
    max_value = CC[0 ,0 ,max_row, max_col]
    center_row = H // 2
    center_col = W // 2
    if max_row == 0 or max_row == H or max_col == 0 or max_col == W:
        z_max_vec = (0 ,0)
        shifts_total = (0 ,0)
    else:
        # fitting_points_x = CC[:,:, max_row:max_row+1, max_col-1:max_col+2]
        # fitting_points_y = CC[:,:, max_row-1:max_row+2, max_col:max_col+1]
        fitting_points_x = CC[: ,:, max_row, :]
        fitting_points_y = CC[: ,:, :, max_col]
        fitting_points_x = fitting_points_x.cpu().numpy()
        fitting_points_y = fitting_points_y.cpu().numpy()

        x_vec = np.arange(-(W // 2), W // 2 + 1)
        y_vec = np.arange(-(H // 2), H // 2 + 1)
        shiftx, x_parabola_max = return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(fitting_points_x,
                                                                                               x_vec=x_vec)
        shifty, y_parabola_max = return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(fitting_points_y,
                                                                                               x_vec=y_vec)
        shifts_total = (shiftx, shifty)
        z_max_vec = (x_parabola_max, y_parabola_max)

    # ### Fast Way to only find shift: ###
    # y1 = fitting_points_x[:,:,:,0]
    # y2 = fitting_points_x[:,:,:,1]
    # y3 = fitting_points_x[:,:,:,2]
    # shiftx = (y1 - y3) / (2 * (y1 + y3 - 2 * y2))
    # y1 = fitting_points_y[:,:,0,:]
    # y2 = fitting_points_y[:,:,1,:]
    # y3 = fitting_points_y[:,:,2,:]
    # shifty = (y1 - y3) / (2 * (y1 + y3 - 2 * y2))
    # shifts_total = (float(shiftx[0][0][0]), float(shifty[0][0][0]))
    return shifts_total, z_max_vec


def return_shifts_using_paraboloid_fit(CC):
    ### Get sequence of (x,y) locations: ###
    cloc = 1  # Assuming 3X3. TODO: generalize to whatever size
    rloc = 1
    x = np.ndarray([1, cloc - 1, cloc - 1, cloc, cloc, cloc, cloc + 1, cloc + 1,
                    cloc + 1]) - 1  # the -1 is because we immigrated from matlab (1 based) to python (0 based)
    y = np.ndarray([rloc - 1, rloc, rloc + 1, rloc - 1, rloc, rloc + 1, rloc - 1, rloc, rloc + 1]) - 1

    ### Get corresponding z(x,y) values: ###
    cross_correlation_samples = np.zeros((len(x)))
    for k in np.arange(len(x)):
        cross_correlation_samples[k] = CC[x[k], y[k]]

        ### Fit paraboloid surface and get corresponding paraboloid coefficients: ###
    [coeffs] = fit_polynom_surface(x, y, cross_correlation_samples, 2)
    shifty = (-(coeffs(2) * coeffs(5) - 2 * coeffs(3) * coeffs(4)) / (coeffs(5) ^ 2 - 4 * coeffs(3) * coeffs(6)))
    shiftx = ((2 * coeffs(2) * coeffs(6) - coeffs(4) * coeffs(5)) / (coeffs(5) ^ 2 - 4 * coeffs(3) * coeffs(6)))

    ### Find Z_max at found (shiftx,shifty): ###
    z_max = evaluate_2d_polynom_surface(shiftx, shifty, coeffs)

    return (shiftx, shifty)

def center_of_mass(z):
    """Return the center of mass of an array with coordinates in the
    hyperspy convention

    Parameters
    ----------
    z : np.array

    Returns
    -------
    (x,y) : tuple of floats
        The x and y locations of the center of mass of the parsed square
    """

    s = np.sum(z)
    if s != 0:
        z *= 1 / s
    dx = np.sum(z, axis=0)
    dy = np.sum(z, axis=1)
    h, w = z.shape
    cx = np.sum(dx * np.arange(w))
    cy = np.sum(dy * np.arange(h))
    return cx, cy

def evaluate_2d_polynom_surface(x, y, coeffs_mat):
    original_size = np.shape(x)
    x = x[:]
    y = y[:]
    z_values = np.zeros(x.shape)

    ### Combined x^i*y^j: ###
    for current_deg_x in np.arange(np.size(coeffs_mat)):
        for current_deg_y in np.arange(np.size(coeffs_mat) - current_deg_x + 1):
            z_values = z_values + coeffs_mat[current_deg_x, current_deg_y] * (x ** current_deg_x) * (
                        y ** current_deg_y)

    z_values = np.reshape(z_values, original_size)
    return z_values

def fit_polynom_surface(x, y, z, order):
    #  Fit a polynomial f(x,y) so that it provides a best fit
    #  to the data z.
    #  Uses SVD which is robust even if the data is degenerate.  Will always
    #  produce a least-squares best fit to the data even if the data is
    #  overspecified or underspecified.
    #  x, y, z are column vectors specifying the points to be fitted.
    #  The three vectors must be the same length.
    #  Order is the order of the polynomial to fit.
    #  Coeffs returns the coefficients of the polynomial.  These are in
    #  increasing power of y for each increasing power of x, e.g. for order 2:
    #  zbar = coeffs(1) + coeffs(2).*y + coeffs(3).*y^2 + coeffs(4).*x +
    #  coeffs(5).*x.*y + coeffs(6).*x^2
    #  Use eval2dPoly to evaluate the polynomial.

    [sizexR, sizexC] = np.shape(x)
    [sizeyR, sizeyC] = np.shape(y)
    [sizezR, sizezC] = np.shape(z)
    numVals = sizexR

    ### scale to prevent precision problems: ###
    scalex = 1.0 / max(abs(x))
    scaley = 1.0 / max(abs(y))
    scalez = 1.0 / max(abs(z))
    xs = x * scalex
    ys = y * scaley
    zs = z * scalez

    ### number of combinations of coefficients in resulting polynomial: ###
    numCoeffs = (order + 2) * (order + 1) / 2

    ### Form array to process with SVD: ###
    A = np.zeros(numVals, numCoeffs)

    column = 1
    for xpower in np.arange(order):
        for ypower in np.arange(order - xpower):
            A[:, column] = (xs ** xpower) * (ys ** ypower)
            column = column + 1

    ### Perform SVD: ###
    [u, s, v] = np.linalg.svd(A)

    ### pseudo-inverse of diagonal matrix s: ###
    eps = 2e-16
    sigma = eps ** (1 / order)  # minimum value considered non-zero
    qqs = np.diag(s)
    qqs[abs(qqs) >= sigma] = 1 / qqs[abs(qqs) >= sigma]
    qqs[abs(qqs) < sigma] = 0
    qqs = np.diag(qqs)
    if numVals > numCoeffs:
        qqs[numVals, 1] = 0  # add empty rows

    ### calculate solution: ###
    coeffs = v * np.transpose(qqs) * np.transpose(u) * zs

    ### scale the coefficients so they are correct for the unscaled data: ###
    column = 1
    for xpower in np.arange(order):
        for ypower in np.arange(order - xpower):
            coeffs[column] = coeffs(column) * (scalex ** xpower) * (scaley ** ypower) / scalez
            column = column + 1


from skimage.measure import label, regionprops, regionprops_table, find_contours
from RapidBase.Utils.IO.Imshow_and_Plots import draw_bounding_boxes_with_labels_on_image_XYXY, draw_circle_with_label_on_image, draw_ellipse_with_label_on_image, draw_trajectories_on_images, draw_text_on_image, draw_polygons_on_image, draw_polygon_points_on_image, draw_circles_with_labels_on_image, draw_ellipses_with_labels_on_image
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import make_tuple_int, get_COM_and_MOI_tensor_torch

def fit_ellipse_2D_using_scipy_regionprops_torch(input_tensor):
    ### Use regionprops to get ellipse properties: ###
    input_tensor_numpy = input_tensor.cpu().numpy()
    input_tensor_label_numpy = input_tensor.bool().cpu().numpy().astype(np.uint8)
    regionpprops_output = regionprops(input_tensor_label_numpy, input_tensor_numpy)
    ellipse_a = regionpprops_output[0].axis_major_length
    ellipse_b = regionpprops_output[0].axis_minor_length
    ellipse_axes_length = (ellipse_a, ellipse_b)
    ellipse_area = regionpprops_output[0].area
    ellipse_BB = regionpprops_output[0].bbox
    ellipse_centroid = regionpprops_output[0].centroid
    ellipse_central_moments = regionpprops_output[0].moments_central
    ellipse_orientation = regionpprops_output[0].orientation
    ellipse_intertia_tensor = regionpprops_output[0].inertia_tensor
    ellipse_intertia_eigenvalues = regionpprops_output[0].inertia_tensor_eigvals

    ### Plot Regionprops: ###
    ellipse_axes_length = (ellipse_axes_length[0] / np.sqrt(2), ellipse_axes_length[1] / np.sqrt(2))
    # ellipse_axes_length = (ellipse_axes_length[0] / 2, ellipse_axes_length[1] / 2)
    ellipse_axes_length = (ellipse_axes_length[1], ellipse_axes_length[0])
    ellipse_centroid = (ellipse_centroid[1], ellipse_centroid[0])

    ### Draw ellipse on numpy image: ###
    output_frame = draw_ellipse_with_label_on_image(input_tensor_numpy * 255,
                                                    center_coordinates=make_tuple_int(ellipse_centroid),
                                                    ellipse_angle=ellipse_orientation * 180 / np.pi,
                                                    axes_lengths=make_tuple_int(ellipse_axes_length),
                                                    ellipse_label='bla', line_thickness=3)

    # ### Plot: ###
    # plt.imshow(output_frame)
    # plt.imshow(input_tensor_numpy)

    return output_frame, ellipse_centroid, ellipse_axes_length

def fit_ellipse_2D_using_moment_of_intertia_torch(input_tensor):
    ### Use moment of inertia (MOI): ###
    cx, cy, cx2, cy2, cxy, MOI_tensor = get_MOI_tensor_torch(input_tensor.unsqueeze(0).unsqueeze(0))
    # torch.inverse(MOI_tensor)
    eigen_values, eigen_vectors = torch.linalg.eig(MOI_tensor)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    major_axis = eigen_vectors[:, 0]
    minor_axis = eigen_vectors[:, 1]
    ellipse_orientation = torch.arctan(minor_axis[1] / minor_axis[0])  #TODO: make sure to understand this better

    ellipse_centroid = (cx.item(), cy.item())
    ellipse_axes_length = (eigen_values[0].item(), eigen_values[1].item())
    ellipse_axes_length = (ellipse_axes_length[0] / 2 / np.sqrt(2), ellipse_axes_length[1] / 2 / np.sqrt(2))
    ellipse_axes_length = (ellipse_axes_length[1], ellipse_axes_length[0])

    ### Draw ellipse on numpy image: ###
    output_frame = draw_ellipse_with_label_on_image(input_tensor.cpu().numpy() * 255,
                                                    center_coordinates=make_tuple_int(ellipse_centroid),
                                                    ellipse_angle=ellipse_orientation.item() * 180 / np.pi,
                                                    axes_lengths=make_tuple_int(ellipse_axes_length),
                                                    ellipse_label='bla', line_thickness=3)

    # ### Plot: ###
    # plt.imshow(output_frame)
    # plt.imshow(input_tensor.cpu().numpy())
    # # imshow_torch(input_tensor)

    return output_frame, ellipse_centroid, ellipse_axes_length


def fit_ellipse_2D_outer_ring_using_least_squares(input_tensor):
    alpha = 5
    beta = 3
    N = 500
    DIM = 2

    np.random.seed(2)

    # Generate random points on the unit circle by sampling uniform angles
    theta = np.random.uniform(0, 2 * np.pi, (N, 1))
    eps_noise = 0.2 * np.random.normal(size=[N, 1])
    circle = np.hstack([np.cos(theta), np.sin(theta)])

    # Stretch and rotate circle to an ellipse with random linear tranformation
    B = np.random.randint(-3, 3, (DIM, DIM))
    noisy_ellipse = circle.dot(B) + eps_noise

    # Extract x coords and y coords of the ellipse as column vectors
    X = noisy_ellipse[:, 0:1]
    Y = noisy_ellipse[:, 1:]

    # Formulate and solve the least squares problem ||Ax - b ||^2
    A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b)[0].squeeze()

    # Print the equation of the ellipse in standard form
    print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1], x[2], x[3], x[4]))

    # Plot the noisy data
    plt.scatter(X, Y, label='Data Points')

    # Plot the original ellipse from which the data was generated
    phi = np.linspace(0, 2 * np.pi, 1000).reshape((1000, 1))
    c = np.hstack([np.cos(phi), np.sin(phi)])
    ground_truth_ellipse = c.dot(B)
    plt.plot(ground_truth_ellipse[:, 0], ground_truth_ellipse[:, 1], 'k--', label='Generating Ellipse')

    # Plot the least squares ellipse
    x_coord = np.linspace(-5, 5, 300)
    y_coord = np.linspace(-5, 5, 300)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord ** 2 + x[3] * X_coord + x[4] * Y_coord
    plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()



def center_of_mass(z):
    """Return the center of mass of an array with coordinates in the
    hyperspy convention

    Parameters
    ----------
    z : np.array

    Returns
    -------
    (x,y) : tuple of floats
        The x and y locations of the center of mass of the parsed square
    """

    s = np.sum(z)
    if s != 0:
        z *= 1 / s
    dx = np.sum(z, axis=0)
    dy = np.sum(z, axis=1)
    h, w = z.shape
    cx = np.sum(dx * np.arange(w))
    cy = np.sum(dy * np.arange(h))
    return cx, cy



from scipy.optimize import curve_fit
from scipy.stats import chi2, norm, lognorm, loglaplace, laplace, poisson
from scipy.special import factorial
from scipy.stats import poisson
def fit_Gaussian(x, y):
    # N = 100
    # x = my_linspace(0, 5, N)
    # A = 1.5
    # mu = 2
    # sigma = 1.2
    # noise_sigma = 0.1
    # y = A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + noise_sigma*np.random.randn(N)

    def gaus(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    n = len(x)  # the number of data
    mean = sum(x * y) / n  # note this correction
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / n)  # note this correction

    popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])
    y_fitted = gaus(x, *popt)

    # plot(x,y)
    # plot(x,y_fitted,'--')
    # legend(['real','fitted'])

    # hist, bin_edges = np.histogram(bla,100)
    # bin_centers = (bin_edges[1:]+bin_edges[0:-1])/2
    # x = bin_centers
    # y = hist
    popt[1] = popt[1]
    popt[2] = popt[2]

    return y_fitted, popt, pcov

def fit_LogGaussian(x, y):
    # sqrt(1 / (2 * pi)) / (s * x) * exp(- (log(x - m) ^ 2) / (2 * s ^ 2))
    # N = 100
    # x = linspace(3, 5, N)
    # A = 1.5
    # mu = 2
    # sigma = 1.2
    # noise_sigma = 0.1
    # y = A * np.exp(-(np.log(x - mu) ** 2 / (2 * sigma ** 2))) / (sigma * x) + noise_sigma * np.random.randn(N)

    n = len(x)  # the number of data
    mean = sum(x * y) / n  # note this correction
    sigma = sum(y * (x - mean) ** 2) / n  # note this correction

    def gaus(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) / (sigma * x)

    popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])
    y_fitted = gaus(x, *popt)

    # plot(x, y)
    # plot(x, y_fitted, '--')
    # legend(['real', 'fitted'])

    return y_fitted

def fit_Laplace(x, y):
    # 1 / (2 * b) * exp(-abs(x - u) / b)
    # N = 100
    # x = linspace(0, 5, N)
    # A = 1.5
    # mu = 2
    # sigma = 1.2
    # noise_sigma = 0.1
    # y = A * np.exp(-np.abs(x - mu) / sigma) + noise_sigma * np.random.randn(N)

    n = len(x)  # the number of data
    mean = sum(x * y) / n  # note this correction
    sigma = sum(y * (x - mean) ** 2) / n  # note this correction

    def gaus(x, a, x0, sigma):
        return a * np.exp(-abs(x - x0) / sigma)

    popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])
    y_fitted = gaus(x, *popt)
    # plot(x, y)
    # plot(x, y_fitted, '--')
    # legend(['real', 'fitted'])

    return y_fitted, popt, pcov

def fit_Poisson(x, y):
    # N = 255
    # x = np.arange(0,255)
    # A = 1.5
    # mu = 143
    # sigma = 1.2
    # noise_sigma = 0.
    # y = A * np.exp(-mu) * (mu**x) / scipy.special.factorial(x) + noise_sigma * np.random.randn(N)

    def fit_function(x, lamb, A):
        '''poisson function, parameter lamb is the fit parameter'''
        return A * poisson.pmf(x, lamb)

    # fit with curve_fit
    parameters, cov_matrix = curve_fit(fit_function, x, y)

    y_fitted = fit_function(x, *parameters)

    # plot(x, y)
    # plot(x, y_fitted, '--')
    # legend(['real', 'fitted'])

    return y_fitted


import numbers


################################################################################################################
### Simple Model/Function/Objective Optimization Using Pytorch Gradient-Based Optimization: ###
import torch.nn as nn
import torch
class Model(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((3,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        self.real_coefficients = [1, -0.2, 0.3]

    def forward(self, X):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-b * X) + c),
        """
        a, b, c = self.coefficients
        return a * torch.exp(-b * X) + c
        # return self.a * torch.exp(-self.k*X) + self.b

    def real_forward(self, X):
        a, b, c = self.real_coefficients
        return a * torch.exp(-b * X) + c


class FitFunctionTorch_Gaussian(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        """Implement function to be optimised. In this case, an exponential decay
        function (a * exp(-b * X) + c),
        """
        a, b, c, x0 = self.coefficients
        return a * torch.exp(-b * (X - x0) ** 2) + c


class FitFunctionTorch_Laplace(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0 = self.coefficients
        return a * torch.exp(-b * (X - x0).abs()) + c


class FitFunctionTorch_Maxwell(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0 = self.coefficients
        return a * (X ** 2) * torch.exp(-b * (X - x0).abs()) + c


class FitFunctionTorch_LogNormal(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0 = self.coefficients
        return a / (X ** 2) * torch.exp(-b * (torch.log(X) - x0).abs()) + c


class FitFunctionTorch_FDistribution(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, d1=None, d2=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if d1 is not None:
                self.coefficients[1] = d1
            if d2 is not None:
                self.coefficients[2] = d2
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, d1, d2, x0 = self.coefficients
        return a / X * torch.sqrt((d1 * X) ** d1 / (d1 * X + d2) ** (d1 + d2))


class FitFunctionTorch_Rayleigh(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0 = self.coefficients
        return a * X * torch.exp(-b * (X - x0) ** 2) + c


class FitFunctionTorch_Lorenzian(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0 = self.coefficients
        return a * 1 / (1 + ((X - x0) / b) ** 2) + c


class FitFunctionTorch_Sin(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0 = self.coefficients
        return a * torch.sin(b * (X - x0)) + c

    def function_string(self):
        return 'a * sin(b*(x-x0)) + c,     variables: a,b,c,x0 '


class FitFunctionTorch_DecayingSin(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None, x1=None, t=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((6,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0, x1, t = self.coefficients
        return a * torch.sin(b * (X - x0)) * torch.exp(-t * (X - x1).abs()) + c

    def function_string(self):
        return 'a * sin(b*(x-x0)) * exp(-t|x-x1|) + c,    '


class FitFunctionTorch_DecayingSinePlusLine(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None, x1=None, t=None, d=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((7,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0
            if x1 is not None:
                self.coefficients[4] = x1
            if t is not None:
                self.coefficients[5] = t
            if d is not None:
                self.coefficients[6] = d

    def forward(self, X):
        a, b, c, x0, x1, t, d = self.coefficients
        return a * torch.sin(b * (X - x0)) * torch.exp(-t * (X - x1).abs()) + c * X + d

    def function_string(self):
        return 'a * sin(b*(x-x0)) * exp(-t|x-x1|) + c*x + d,    '


def Coefficients_Optimizer_Torch(x, y,
                                 model,
                                 learning_rate=1e-3,
                                 loss_function=torch.nn.L1Loss(reduction='sum'),
                                 max_number_of_iterations=500,
                                 tolerance=1e-3,
                                 print_frequency=10):
    # learning_rate = 1e-6
    # loss_func = torch.nn.MSELoss(reduction='sum')

    # (*). x is the entire dataset inputs with shape [N,Mx]. N=number of observations, Mx=number of dimensions
    # (*). y is the entire dataset outputs/noisy-results with shape [N,My], My=number of output dimensions

    ### Define Optimizer: ###
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    flag_continue_optimization = True
    time_step = 1
    while flag_continue_optimization:
        ### Basic Optimization Step: ###
        # print(time_step)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        if time_step % print_frequency == print_frequency - 1:
            print('time step:  ' + str(time_step) + ';    current loss: ' + str(loss.item()))
        loss.backward()
        optimizer.step()

        ### Check whether to stop optimization: ###
        if loss < tolerance:
            flag_continue_optimization = False
            print('reached tolerance')
        if time_step == max_number_of_iterations:
            flag_continue_optimization = False
            print('reached max number of iterations')

        ### Advance time step: ###
        time_step += 1

    return model, model.coefficients


def get_initial_guess_for_sine(x, y_noisy):
    y_noisy_meaned = y_noisy - y_noisy.mean()
    y_noisy_smooth = convn_torch(y_noisy_meaned, torch.ones(10) / 10, dim=0).squeeze()
    y_noisy_smooth_sign_change = y_noisy_smooth[1:] * y_noisy_smooth[0:-1] < 0
    y_number_of_zero_crossings = y_noisy_smooth_sign_change.sum()
    zero_crossing_period = y_noisy_smooth_sign_change.float().nonzero().squeeze().diff().float().mean()
    full_cycle_period = zero_crossing_period * 2
    b_initial = (full_cycle_period) * (x[1] - x[0]) / (2 * pi)
    c_initial = y_noisy.mean()
    a_initial = (y_noisy.max() - y_noisy.min()) / 2
    return a_initial, b_initial, c_initial


# import csaps
# import interpol  #TODO: this brings out an error! cannot use this! maybe try linux!
from RapidBase.MISC_REPOS.torchcubicspline.torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


def Testing_model_fit():
    ### Get Random Signal: ###
    x = torch.linspace(-5, 5, 100)
    y = torch.randn(100).cumsum(0)
    y = y - torch.linspace(0, y[-1].item(), y.size(0))
    # y = y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # plot_torch(y)

    ### Test Spline Fit Over Above Signal: ###
    # (1). CSAPS:
    x_numpy = x.cpu().numpy()
    y_numpy = y.squeeze().cpu().numpy()
    y_numpy_smooth_spline = csaps.csaps(xdata=x_numpy,
                                        ydata=y_numpy,
                                        xidata=x_numpy,
                                        weights=None,
                                        smooth=0.85)
    plt.plot(x_numpy, y_numpy)
    plt.plot(x_numpy, y_numpy_smooth_spline)
    # (2). TorchCubicSplines
    t = x
    x = y
    coeffs = natural_cubic_spline_coeffs(t, x.unsqueeze(-1))
    spline = NaturalCubicSpline(coeffs)
    x_estimate = spline.evaluate(t)
    plot_torch(t, x);
    plot_torch(t + 0.5, x_estimate)
    # (3). Using torch's Fold/Unfold Layers:
    # TODO: come up with a general solution for the division factor per index!!!!!!
    kernel_size = 20
    stride_size = 10
    number_of_index_overlaps = (kernel_size // stride_size)
    # y_image = y.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    y_image = torch.tensor(y_numpy_smooth_spline).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    y_image.shape
    y_image_unfolded = torch.nn.Unfold(kernel_size=(kernel_size, 1), dilation=1, padding=0, stride=(stride_size, 1))(y_image)
    # TODO: add polynomial fit for each segment/interval of the y_image_unfolded before folding back (as cheating, instead of proper boundary conditions in the least squares)
    y_image_folded = torch.nn.Fold(y_image.shape[-2:], kernel_size=(kernel_size, 1), dilation=1, padding=0, stride=(stride_size, 1))(y_image_unfolded)
    plot_torch(y)
    plot_torch(y_image.squeeze())
    plot_torch(y_image_folded.squeeze() / 2)
    # (4). Using Tensors Object's unfold method, need to add averaging of "in between" values or perform conditional least squares like i did in my function!!!: ###
    patch_size = 20
    interval_size = 20
    y_unfolded = y.unfold(-1, patch_size, interval_size)
    # y_unfolded = torch.tensor(y_numpy_smooth_spline).float().unfold(-1, patch_size, interval_size)
    x_unfolded = x.unfold(-1, patch_size, interval_size)
    outputs_list = []
    for patch_index in np.arange(y_unfolded.shape[0]):
        coefficients, prediction, residual_values = polyfit_torch(x_unfolded[patch_index], y_unfolded[patch_index], 2, True, True)
        outputs_list.append(prediction.unsqueeze(0))
    final_output = torch.cat(outputs_list, -1)
    final_output
    plot_torch(x, y);
    plot_torch(x, final_output)
    # (4). Loess (still, the same shit applies...i need to get a handle on folding/unfolding and boundary conditions):

    ### Get Sine Signal: ###
    x = torch.linspace(-10, 10, 1000)
    y = 1.4 * torch.sin(x) + 0.1 * x + 3
    y_noisy = y + torch.randn_like(x) * 0.1
    a_initial, b_initial, c_initial = get_initial_guess_for_sine(x, y_noisy)
    # plot_torch(x,y)
    # plot_torch(x,y_noisy)
    # plt.show()

    ### Fit Sine Signal: ###
    # model_to_fit = FitFunctionTorch_Sin(a_initial, b_initial, c_initial, None)
    # model_to_fit = FitFunctionTorch_DecayingSin(a_initial, b_initial, c_initial, None, None, 0)
    model_to_fit = FitFunctionTorch_DecayingSinePlusLine(a_initial, b_initial, None, None, None, 0)
    device = 'cuda'
    x = x.to(device)
    y_noisy = y_noisy.to(device)
    model_to_fit = model_to_fit.to(device)
    tic()
    model, coefficients = Coefficients_Optimizer_Torch(x, y_noisy,
                                                       model_to_fit,
                                                       learning_rate=0.5e-3,
                                                       loss_function=torch.nn.L1Loss(),
                                                       max_number_of_iterations=5000,
                                                       tolerance=1e-3, print_frequency=10)
    toc('optimization')
    print(model.coefficients)
    print(model_to_fit.function_string())
    plot_torch(x, y_noisy)
    plot_torch(x, model_to_fit.forward((x)))
    plt.legend(['input noisy', 'fitted model'])
    plt.show()


# Testing_model_fit()
################################################################################################################


################################################################################################################
### RANSAC: ###
def _check_data_dim(data, dim):
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError('Input data must have shape (N, %d).' % dim)


def _check_data_atleast_2D(data):
    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError('Input data must be at least 2D.')


def _norm_along_axis(x, axis):
    """NumPy < 1.8 does not support the `axis` argument for `np.linalg.norm`."""
    return np.sqrt(np.einsum('ij,ij->i', x, x))


def _norm_along_axis_Torch(x, axis):
    """NumPy < 1.8 does not support the `axis` argument for `np.linalg.norm`."""
    return torch.sqrt(torch.einsum('ij,ij->i', x, x))


class BaseModel(object):
    def __init__(self):
        self.params = None


class LineModelND(BaseModel):
    """Total least squares estimator for N-dimensional lines.

    In contrast to ordinary least squares line estimation, this estimator
    minimizes the orthogonal distances of points to the estimated line.

    Lines are defined by a point (origin) and a unit vector (direction)
    according to the following vector equation::

        X = origin + lambda * direction

    Attributes
    ----------
    params : tuple
        Line model parameters in the following order `origin`, `direction`.

    # Examples
    # --------
    # >>> x = np.linspace(1, 2, 25)
    # >>> y = 1.5 * x + 3
    # >>> lm = LineModelND()
    # >>> lm.estimate(np.array([x, y]).T)
    # True
    # >>> tuple(np.round(lm.params, 5))
    # (array([1.5 , 5.25]), array([0.5547 , 0.83205]))
    # >>> res = lm.residuals(np.array([x, y]).T)
    # >>> np.abs(np.round(res, 9))
    # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #        0., 0., 0., 0., 0., 0., 0., 0.])
    # >>> np.round(lm.predict_y(x[:5]), 3)
    # array([4.5  , 4.562, 4.625, 4.688, 4.75 ])
    # >>> np.round(lm.predict_x(y[:5]), 3)
    # array([1.   , 1.042, 1.083, 1.125, 1.167])
    #
    # """

    def estimate(self, data):
        """Estimate line model from data.

        This minimizes the sum of shortest (orthogonal) distances
        from the given data points to the estimated line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimensionality dim >= 2.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        _check_data_atleast_2D(data)

        origin = data.mean(axis=0)
        data = data - origin

        if data.shape[0] == 2:  # well determined
            direction = data[1] - data[0]
            norm = np.linalg.norm(direction)
            if norm != 0:  # this should not happen to be norm 0
                direction /= norm
        elif data.shape[0] > 2:  # over-determined
            # Note: with full_matrices=1 Python dies with joblib parallel_for.
            _, _, v = np.linalg.svd(data, full_matrices=False)
            direction = v[0]
        else:  # under-determined
            raise ValueError('At least 2 input points needed.')

        self.params = (origin, direction)

        return True

    def residuals(self, data, params=None):
        """Determine residuals of data to model.

        For each point, the shortest (orthogonal) distance to the line is
        returned. It is obtained by projecting the data onto the line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimension dim.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """
        _check_data_atleast_2D(data)
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, direction = params
        res = (data - origin) - \
              ((data - origin) @ direction)[..., np.newaxis] * direction
        return _norm_along_axis(res, axis=1)

    def predict(self, x, axis=0, params=None):
        """Predict intersection of the estimated line model with a hyperplane
        orthogonal to a given axis.

        Parameters
        ----------
        x : (n, 1) array
            Coordinates along an axis.
        axis : int
            Axis orthogonal to the hyperplane intersecting the line.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        data : (n, m) array
            Predicted coordinates.

        Raises
        ------
        ValueError
            If the line is parallel to the given axis.
        """
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, direction = params

        if direction[axis] == 0:
            # line parallel to axis
            raise ValueError('Line parallel to axis %s' % axis)

        l = (x - origin[axis]) / direction[axis]
        data = origin + l[..., np.newaxis] * direction
        return data


class LineModelND_Torch(BaseModel):
    """Total least squares estimator for N-dimensional lines.

    In contrast to ordinary least squares line estimation, this estimator
    minimizes the orthogonal distances of points to the estimated line.

    Lines are defined by a point (origin) and a unit vector (direction)
    according to the following vector equation::

        X = origin + lambda * direction

    Attributes
    ----------
    params : tuple
        Line model parameters in the following order `origin`, `direction`.

    # Examples
    # --------
    # >>> x = np.linspace(1, 2, 25)
    # >>> y = 1.5 * x + 3
    # >>> lm = LineModelND()
    # >>> lm.estimate(np.array([x, y]).T)
    # True
    # >>> tuple(np.round(lm.params, 5))
    # (array([1.5 , 5.25]), array([0.5547 , 0.83205]))
    # >>> res = lm.residuals(np.array([x, y]).T)
    # >>> np.abs(np.round(res, 9))
    # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #        0., 0., 0., 0., 0., 0., 0., 0.])
    # >>> np.round(lm.predict_y(x[:5]), 3)
    # array([4.5  , 4.562, 4.625, 4.688, 4.75 ])
    # >>> np.round(lm.predict_x(y[:5]), 3)
    # array([1.   , 1.042, 1.083, 1.125, 1.167])
    #
    # """

    def estimate(self, data):
        """Estimate line model from data.

        This minimizes the sum of shortest (orthogonal) distances
        from the given data points to the estimated line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimensionality dim >= 2.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        _check_data_atleast_2D(data)

        origin = data.mean(0)
        data = data - origin

        if data.shape[0] == 2:  # well determined
            direction = data[1] - data[0]
            norm = torch.linalg.norm(direction)
            if norm != 0:  # this should not happen to be norm 0
                direction /= norm
        elif data.shape[0] > 2:  # over-determined
            # Note: with full_matrices=1 Python dies with joblib parallel_for.
            _, _, v = torch.linalg.svd(data, full_matrices=False)
            direction = v[0]
        else:  # under-determined
            raise ValueError('At least 2 input points needed.')

        self.params = (origin, direction)

        return True

    def residuals(self, data, params=None):
        """Determine residuals of data to model.

        For each point, the shortest (orthogonal) distance to the line is
        returned. It is obtained by projecting the data onto the line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimension dim.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """

        ### Make sure data is fine (at least 2d): ###
        _check_data_atleast_2D(data)

        ### Make sure we have some parameters which define the model so we can calculate the parameters: ###
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        ### Get residuals from model data points (in this case it's the distance between the data point to the direction line): ###
        origin, direction = params

        # ### Before Bug: ###
        # res = (data - origin) - ((data - origin) @ direction).unsqueeze(-1) * direction
        ### After Bug: ###
        bug_patch = torch.matmul(direction.unsqueeze(0), (data - origin).T).squeeze()
        res = (data - origin) - (bug_patch).unsqueeze(-1) * direction

        # bla1 = ((data-origin) @ direction)
        # bla2 = torch.mm(data-origin, direction.unsqueeze(1)).squeeze()
        return _norm_along_axis_Torch(res, axis=1)

    def predict(self, x, axis=0, params=None):
        """Predict intersection of the estimated line model with a hyperplane
        orthogonal to a given axis.

        Parameters
        ----------
        x : (n, 1) array
            Coordinates along an axis.
        axis : int
            Axis orthogonal to the hyperplane intersecting the line.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        data : (n, m) array
            Predicted coordinates.

        Raises
        ------
        ValueError
            If the line is parallel to the given axis.
        """
        ### Make sure params is defined so we can work with something here: ###
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, direction = params

        if direction[axis] == 0:
            # line parallel to axis
            raise ValueError('Line parallel to axis %s' % axis)

        l = (x - origin[axis]) / direction[axis]
        data = origin + l[..., np.newaxis] * direction
        return data


def _dynamic_max_trials(n_inliers, n_samples, min_samples_for_model, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.
    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.
    n_samples : int
        Total number of samples in the data.
    min_samples_for_model : int
        Minimum number of samples chosen randomly from original data.
    probability : float
        Probability (confidence) that one outlier-free sample is generated.
    Returns
    -------
    trials : int
        Number of trials.
    """
    if n_inliers == 0:
        return np.inf

    nom = 1 - probability
    if nom == 0:
        return np.inf

    inlier_ratio = n_inliers / float(n_samples)
    denom = 1 - inlier_ratio ** min_samples_for_model
    if denom == 0:
        return 1
    elif denom == 1:
        return np.inf

    nom = np.log(nom)
    denom = np.log(denom)
    if denom == 0:
        return 0

    return int(np.ceil(nom / denom))


def check_random_state(seed):
    """Turn seed into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : None, int or np.random.RandomState
           If `seed` is None, return the RandomState singleton used by `np.random`.
           If `seed` is an int, return a new RandomState instance seeded with `seed`.
           If `seed` is already a RandomState instance, return it.

    Raises
    ------
    ValueError
        If `seed` is of the wrong type.

    """
    # Function originally from scikit-learn's module sklearn.utils.validation
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def ransac(data, model_class, min_samples_for_model, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None, initial_inliers=None):
    """Fit a model to data with the RANSAC (random sample consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. Each iteration
    performs the following tasks:

    1. Select `min_samples_for_model` random samples from the original data and check
       whether the set of data is valid (see `is_data_valid`).
    2. Estimate a model to the random subset
       (`model_cls.estimate(*data[random_subset]`) and check whether the
       estimated model is valid (see `is_model_valid`).
    3. Classify all data as inliers or outliers by calculating the residuals
       to the estimated model (`model_cls.residuals(*data)`) - all data samples
       with residuals smaller than the `residual_threshold` are considered as
       inliers.
    4. Save estimated model as best model if number of inlier samples is
       maximal. In case the current estimated model has the same number of
       inliers, it is only considered as the best model if it has less sum of
       residuals.

    These steps are performed either a maximum number of times or until one of
    the special stop criteria are met. The final model is estimated using all
    inlier samples of the previously determined best model.

    Parameters
    ----------
    data : [list, tuple of] (N, ...) array
        Data set to which the model is fitted, where N is the number of data
        points and the remaining dimension are depending on model requirements.
        If the model class requires multiple input data arrays (e.g. source and
        destination coordinates of  ``skimage.transform.AffineTransform``),
        they can be optionally passed as tuple or list. Note, that in this case
        the functions ``estimate(*data)``, ``residuals(*data)``,
        ``is_model_valid(model, *random_data)`` and
        ``is_data_valid(*random_data)`` must all take each data array as
        separate arguments.
    model_class : object
        Object with the following object methods:

         * ``success = estimate(*data)``
         * ``residuals(*data)``

        where `success` indicates whether the model estimation succeeded
        (`True` or `None` for success, `False` for failure).
    min_samples_for_model : int in range (0, N)
        The minimum number of data points to fit a model to.
    residual_threshold : float larger than 0
        Maximum distance for a data point to be classified as an inlier.
    is_data_valid : function, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(*random_data)`.
    is_model_valid : function, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, *random_data)`, .
    max_trials : int, optional
        Maximum number of iterations for random sample selection.
    stop_sample_num : int, optional
        Stop iteration if at least this number of inliers are found.
    stop_residuals_sum : float, optional
        Stop iteration if sum of residuals is less than or equal to this
        threshold.
    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the
        training data is sampled with ``probability >= stop_probability``,
        depending on the current best model's inlier ratio and the number
        of trials. This requires to generate at least N samples (trials):

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to a high value
        such as 0.99, e is the current fraction of inliers w.r.t. the
        total number of samples, and m is the min_samples_for_model value.
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    initial_inliers : array-like of bool, shape (N,), optional
        Initial samples selection for model estimation


    Returns
    -------
    model : object
        Best model with largest consensus set.
    inliers : (N, ) array
        Boolean mask of inliers classified as ``True``.

    References
    ----------
    .. [1] "RANSAC", Wikipedia, https://en.wikipedia.org/wiki/RANSAC

    Examples
    --------

    Generate ellipse data without tilt and add noise:

    # >>> t = np.linspace(0, 2 * np.pi, 50)
    # >>> xc, yc = 20, 30
    # >>> a, b = 5, 10
    # >>> x = xc + a * np.cos(t)
    # >>> y = yc + b * np.sin(t)
    # >>> data = np.column_stack([x, y])
    # >>> np.random.seed(seed=1234)
    # >>> data += np.random.normal(size=data.shape)
    #
    # Add some faulty data:
    #
    # >>> data[0] = (100, 100)
    # >>> data[1] = (110, 120)
    # >>> data[2] = (120, 130)
    # >>> data[3] = (140, 130)
    #
    # Estimate ellipse model using all available data:
    #
    # >>> model = EllipseModel()
    # >>> model.estimate(data)
    # True
    # >>> np.round(model.params)  # doctest: +SKIP
    # array([ 72.,  75.,  77.,  14.,   1.])
    #
    # Estimate ellipse model using RANSAC:
    #
    # >>> ransac_model, inliers = ransac(data, EllipseModel, 20, 3, max_trials=50)
    # >>> abs(np.round(ransac_model.params))
    # array([20., 30.,  5., 10.,  0.])
    # >>> inliers # doctest: +SKIP
    # array([False, False, False, False,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True], dtype=bool)
    # >>> sum(inliers) > 40
    # True
    #
    # RANSAC can be used to robustly estimate a geometric transformation. In this section,
    # we also show how to use a proportion of the total samples, rather than an absolute number.
    #
    # >>> from skimage.transform import SimilarityTransform
    # >>> np.random.seed(0)
    # >>> src = 100 * np.random.rand(50, 2)
    # >>> model0 = SimilarityTransform(scale=0.5, rotation=1, translation=(10, 20))
    # >>> dst = model0(src)
    # >>> dst[0] = (10000, 10000)
    # >>> dst[1] = (-100, 100)
    # >>> dst[2] = (50, 50)
    # >>> ratio = 0.5  # use half of the samples
    # >>> min_samples_for_model = int(ratio * len(src))
    # >>> model, inliers = ransac((src, dst), SimilarityTransform, min_samples_for_model, 10,
    # ...                         initial_inliers=np.ones(len(src), dtype=bool))
    # >>> inliers
    # array([False, False, False,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True])

    # """

    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None

    random_state = check_random_state(random_state)

    # in case data is not pair of input and output, male it like it
    if not isinstance(data, (tuple, list)):
        data = (data,)
    total_number_of_samples = len(data[0])

    if not (0 < min_samples_for_model < total_number_of_samples):
        raise ValueError("`min_samples_for_model` must be in range (0, <number-of-samples>)")

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != total_number_of_samples:
        raise ValueError("RANSAC received a vector of initial inliers (length %i)"
                         " that didn't match the number of samples (%i)."
                         " The vector of initial inliers should have the same length"
                         " as the number of samples and contain only True (this sample"
                         " is an initial inlier) and False (this one isn't) values."
                         % (len(initial_inliers), total_number_of_samples))

    # for the first run use initial guess of inliers
    random_indices_to_sample = (initial_inliers if initial_inliers is not None
                                else random_state.choice(total_number_of_samples, min_samples_for_model, replace=False))

    for num_trials in range(max_trials):
        # do sample selection according data pairs
        samples = [d[random_indices_to_sample] for d in data]
        # for next iteration choose random sample set and be sure that no samples repeat
        random_indices_to_sample = random_state.choice(total_number_of_samples, min_samples_for_model, replace=False)

        # optional check if random sample set is valid
        if is_data_valid is not None and not is_data_valid(*samples):
            continue

        # estimate model for current random sample set
        sample_model = model_class()

        success = sample_model.estimate(*samples)
        # backwards compatibility
        if success is not None and not success:
            continue

        # optional check if estimated model is valid
        if is_model_valid is not None and not is_model_valid(sample_model, *samples):
            continue

        sample_model_residuals = np.abs(sample_model.residuals(*data))
        # consensus set / inliers
        sample_model_inliers = sample_model_residuals < residual_threshold
        sample_model_residuals_sum = np.sum(sample_model_residuals ** 2)

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = np.sum(sample_model_inliers)
        if (
                # more inliers
                sample_inlier_num > best_inlier_num
                # same number of inliers but less "error" in terms of residuals
                or (sample_inlier_num == best_inlier_num
                    and sample_model_residuals_sum < best_inlier_residuals_sum)
        ):
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                                                     total_number_of_samples,
                                                     min_samples_for_model,
                                                     stop_probability)
            if (best_inlier_num >= stop_sample_num
                    or best_inlier_residuals_sum <= stop_residuals_sum
                    or num_trials >= dynamic_max_trials):
                break

    # estimate final model using all inliers
    if best_inliers is not None:
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        best_model.estimate(*data_inliers)

    return best_model, best_inliers


def ransac_Torch(data, model_class, min_samples_for_model, residual_threshold,
                 is_data_valid=None, is_model_valid=None,
                 max_trials=100, stop_sample_num=torch.inf, stop_residuals_sum=0,
                 stop_probability=1, random_state=None, initial_inliers=None):
    ### Initialize Paramters: ###
    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None

    ### Get Random State: ###
    random_state = check_random_state(random_state)

    ### in case data is not pair of input and output, make it so: ###
    if not isinstance(data, (tuple, list)):
        data = (data,)
    total_number_of_samples = len(data[0])

    ### Check Inputs: ###
    if not (0 < min_samples_for_model < total_number_of_samples):
        raise ValueError("`min_samples_for_model` must be in range (0, <number-of-samples>)")
    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")
    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")
    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")
    if initial_inliers is not None and len(initial_inliers) != total_number_of_samples:
        raise ValueError("RANSAC received a vector of initial inliers (length %i)"
                         " that didn't match the number of samples (%i)."
                         " The vector of initial inliers should have the same length"
                         " as the number of samples and contain only True (this sample"
                         " is an initial inlier) and False (this one isn't) values."
                         % (len(initial_inliers), total_number_of_samples))

    ### for the first run use initial guess of inliers: ###
    random_indices_to_sample = (initial_inliers if initial_inliers is not None
                                else random_state.choice(total_number_of_samples, min_samples_for_model, replace=False))

    for num_trials in range(max_trials):
        ### do sample selection according data pairs: ###
        samples = [d[random_indices_to_sample] for d in data]

        ### for next iteration choose random sample set and be sure that no samples repeat: ###
        random_indices_to_sample = random_state.choice(total_number_of_samples, min_samples_for_model, replace=False)

        ### optional check if random sample set is valid: ###
        if is_data_valid is not None and not is_data_valid(*samples):
            continue

        ### Initialize new model and estimate parameters from current random sample set: ###
        sample_model = model_class()
        success = sample_model.estimate(*samples)
        # (*) backwards compatibility
        if success is not None and not success:
            continue

        ### optional check if estimated model is valid: ###
        if is_model_valid is not None and not is_model_valid(sample_model, *samples):
            continue

        ### Get residuals from model which was fit: ###
        sample_model_residuals = torch.abs(sample_model.residuals(*data))

        ### Get Consensus set (Inliers) and Residuals sum: ###
        sample_model_inliers = sample_model_residuals < residual_threshold
        sample_model_residuals_sum = (sample_model_residuals ** 2).sum()

        ### choose as new best model if number of inliers is maximal: ###
        sample_inlier_num = (sample_model_inliers).sum()
        flag_current_model_with_more_inliers = (sample_inlier_num > best_inlier_num)
        flag_current_model_with_less_error = (sample_inlier_num == best_inlier_num and sample_model_residuals_sum < best_inlier_residuals_sum)
        flag_current_model_is_the_best_so_far = flag_current_model_with_more_inliers or flag_current_model_with_less_error
        # (*). if new model is the best then record it as such:
        if flag_current_model_is_the_best_so_far:
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                                                     total_number_of_samples,
                                                     min_samples_for_model,
                                                     stop_probability)

            ### Test whether we've reached a point where the model is good enough: ###
            if (best_inlier_num.float() >= stop_sample_num
                    or best_inlier_residuals_sum <= stop_residuals_sum
                    or num_trials >= dynamic_max_trials):
                break

    ### estimate final model using all inliers: ###
    if best_inliers is not None:
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        best_model.estimate(*data_inliers)

    return best_model, best_inliers





def Estimate_Line_From_Event_Indices_Using_RANSAC_And_LineFit_Drone_Torch(res_points, params, H, W):
    ### Get variables from params dict: ###
    DroneLineEst_RANSAC_D = params['DroneLineEst_RANSAC_D']
    DroneLineEst_polyfit_degree = params['DroneLineEst_polyfit_degree']
    DroneLineEst_RANSAC_max_trials = params['DroneLineEst_RANSAC_max_trials']
    minimum_number_of_samples_after_polyfit = params['DroneLineEst_minimum_number_of_samples_after_polyfit']
    minimum_number_of_samples_before_polyfit = params['DroneLineEst_minimum_number_of_samples_before_polyfit']
    ROI_allocated_around_suspect = params['DroneLineEst_ROI_allocated_around_suspect']

    ### Perform RANSAC to estimate line: ###
    # (*). Remember, this accepts ALL the residual points coordinates and outputs a line. shouldn't we use something like clustering???
    model_robust, indices_within_distance = ransac_Torch(res_points, LineModelND_Torch, min_samples_for_model=2,
                                                         residual_threshold=DroneLineEst_RANSAC_D, max_trials=DroneLineEst_RANSAC_max_trials)
    holding_point = model_robust.params[0]
    direction_vec = model_robust.params[1]
    valid_trajectory_points = res_points[indices_within_distance]

    ### Get points within the predefined distance and those above it: ###
    points_off_line_found = res_points[~indices_within_distance]
    points_on_line_found = valid_trajectory_points

    ### if there are enough valid points then do a polynomial fit for their trajectory: ###
    if valid_trajectory_points.shape[0] > minimum_number_of_samples_before_polyfit:
        ### Get Linearly increasing and "full" t_vec: ###
        t_vec = torch.arange(valid_trajectory_points[:, 0][0], valid_trajectory_points[:, 0][-1]).to(res_points.device)

        ########################################################################
        ### Pytorch Polyfit: ###
        coefficients_x, prediction_x, residuals_x = polyfit_torch(valid_trajectory_points[:, 0], valid_trajectory_points[:, 1], DroneLineEst_polyfit_degree)
        coefficients_y, prediction_y, residuals_y = polyfit_torch(valid_trajectory_points[:, 0], valid_trajectory_points[:, 2], DroneLineEst_polyfit_degree)
        ### Pytorch polyval over full t_vec between the first and last time elements of the trajectory: ###
        trajectory_smoothed_polynom_X = polyval_torch(coefficients_x, t_vec)
        trajectory_smoothed_polynom_Y = polyval_torch(coefficients_y, t_vec)
        ########################################################################

        # ########################################################################
        # ### Get (t,x,y) points vec: ###
        # smoothed_trajectory_over_time_vec = torch.cat( (t_vec.unsqueeze(-1),
        #                                                 trajectory_smoothed_polynom_X.unsqueeze(-1),
        #                                                 trajectory_smoothed_polynom_Y.unsqueeze(-1)), -1)
        # ########################################################################

        # ########################################################################
        # ### Test if the x&y trajectories are within certain boundaries: ###
        # #TODO: is this really necessary? there are a lot of calculations being done for nothing! simply check the first and last elements!!!
        # #TODO: i can get rid of this in the first place when accepting outliers probably but probably doesn't take a long time anyway
        # indices_where_x_trajectory_within_frame_boundaries = (trajectory_smoothed_polynom_X >= ROI_allocated_around_suspect / 2 + 1) &\
        #                                                      (trajectory_smoothed_polynom_X <= W - ROI_allocated_around_suspect / 2 - 1)
        # indices_where_y_trajectory_within_frame_boundaries = (trajectory_smoothed_polynom_Y >= ROI_allocated_around_suspect / 2 + 1) &\
        #                                                      (trajectory_smoothed_polynom_Y <= H - ROI_allocated_around_suspect / 2 - 1)
        #
        # ### Get indices_where_trajectory_is_within_frame_boundaries: ###
        # indices_1 = (indices_where_x_trajectory_within_frame_boundaries & indices_where_y_trajectory_within_frame_boundaries)
        #
        # ### Get indices where_trajectory_is_close_enough_to_line_estimate: ###
        # #TODO: is it really necessary to do this again? i mean, the smoothed line will probably contain the same indices as before!!!
        # # it's pretty much a straight line!!! no need to get over our heads!!!!
        # indices_2 = get_Distance_From_Points_To_DirectionVec_Torch(smoothed_trajectory_over_time_vec, direction_vec, holding_point) < (DroneLineEst_RANSAC_D)
        #
        # ### Final valid indices are those which satisfy both conditions: ###
        # t_vec_valid = indices_1 * indices_2
        #
        # ### Get valid parts of the trajectory: ###
        # t_vec = t_vec[t_vec_valid]
        # trajectory_smoothed_polynom_X = trajectory_smoothed_polynom_X[t_vec_valid]
        # trajectory_smoothed_polynom_Y = trajectory_smoothed_polynom_Y[t_vec_valid]
        # ########################################################################

    else:
        ### Get valid parts of the trajectory: ###
        t_vec = []
        trajectory_smoothed_polynom_X = []
        trajectory_smoothed_polynom_Y = []

    ### make sure there are enough valid points: ###
    if len(t_vec) > minimum_number_of_samples_after_polyfit:
        flag_enough_valid_samples = True
    else:
        flag_enough_valid_samples = False

    return direction_vec, holding_point, t_vec,\
           trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, \
           points_off_line_found, points_on_line_found, flag_enough_valid_samples



#############################################################################################


#############################################################################################
### Fit 2D: ###

#############################################################################################


#############################################################################################
### Point Cloud Functions ###


#############################################################################################



#############################################################################################
### Contours: ###

#############################################################################################