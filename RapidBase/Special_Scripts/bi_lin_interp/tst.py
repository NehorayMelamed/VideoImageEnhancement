import torch
import ecc_bilinear_interpolation
import time
import sys

import warnings
warnings.filterwarnings("ignore")

### Example Stuff: ###
T = 25
H = 540
W = 8192
N = 25000
# T = 1
# H = 25
# W = 30
# N = 1

# make a homography matirx
H_matrix = torch.zeros((T, 3, 3)).cuda()
# setg the (2,2) elemnet to 1
H_matrix[:, 2, 2] = 1
# set the diagoanl to 0.75 so that the image is scaled down by 25% plus some small random noise
H_matrix[:, 0, 0] = 0.75 + torch.rand(T).cuda()*0.05
H_matrix[:, 1, 1] = 0.75 + torch.rand(T).cuda()*0.05
# set the (0,2) and (1,2) elements to some small random noise to add a small translation
H_matrix[:, 0, 2] = torch.rand(T).cuda()*0.05
H_matrix[:, 1, 2] = torch.rand(T).cuda()*0.05
# set the (0,1) and (1,0) elements to some small random noise to add a small rotation
H_matrix[:, 0, 1] = torch.rand(T).cuda()*0.05
H_matrix[:, 1, 0] = torch.rand(T).cuda()*0.05


#input_image = torch.rand(T, 1, H, W).cuda()
# vx = torch.rand(T, 1, H, W).cuda()
# vy = torch.rand(T, 1, H, W).cuda()
# for testing we will make vx and vy a grid of size (T, 1, H, W) where the values are betweeen 0 to W-1 for vx and 0 to H-1 for vy
vy, vx = torch.meshgrid(torch.arange(1, H+1), torch.arange(1, W+1))
vx = vx.unsqueeze(0).unsqueeze(1).cuda()
# replicate the values for T
vx = vx.repeat(T, 1, 1, 1)
# cast vx to float
vx = vx.float()

# replicate the values for T
vy = vy.unsqueeze(0).unsqueeze(1).cuda()
vy = vy.repeat(T, 1, 1, 1)
vy = vy.float()

input_image = (vx*vy)

X_mat_chosen_values = (torch.rand(T, N).cuda())*(W-1)
Y_mat_chosen_values = (torch.rand(T, N).cuda())*(H-1)


def bilinear_interpolation_yuri(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values):

    ### Calculations: ###
    T = input_image.shape[0]
    H = input_image.shape[2]
    W = input_image.shape[3]

    H_matrix_corrected = H_matrix.unsqueeze(-1)

    X_mat_chosen_values_corrected = X_mat_chosen_values.unsqueeze(
        -1).unsqueeze(-1)
    Y_mat_chosen_values_corrected = Y_mat_chosen_values.unsqueeze(
        -1).unsqueeze(-1)
    denom = (H_matrix_corrected[:, 2:3, 0:1] * X_mat_chosen_values_corrected +
             H_matrix_corrected[:, 2:3, 1:2] * Y_mat_chosen_values_corrected +
             H_matrix_corrected[:, 2:3, 2:3])
    xx_new = 2 * (H_matrix_corrected[:, 0:1, 0:1] * X_mat_chosen_values_corrected +
                  H_matrix_corrected[:, 0:1, 1:2] * Y_mat_chosen_values_corrected +
                  H_matrix_corrected[:, 0:1, 2:3]) / denom / max(W - 1, 1) - 1
    yy_new = 2 * (H_matrix_corrected[:, 1:2, 0:1] * X_mat_chosen_values_corrected +
                  H_matrix_corrected[:, 1:2, 1:2] * Y_mat_chosen_values_corrected +
                  H_matrix_corrected[:, 1:2, 2:3]) / denom / max(H - 1, 1) - 1

    ### Subpixel Interpolation 2: ###
    ### Subpixel Interpolation 2: ###
    bilinear_grid = torch.cat([xx_new, yy_new], dim=3)
    input_image_warped = torch.nn.functional.grid_sample(
        input_image, bilinear_grid, mode='bilinear')  # [out] = [1,1,N,1]
    vx_warped = torch.nn.functional.grid_sample(
        vx, bilinear_grid, mode='bilinear')
    vy_warped = torch.nn.functional.grid_sample(
        vy, bilinear_grid, mode='bilinear')

    # print("x " , ((xx_new[0]+1)/2*(W-1)).item(), " y ", ((yy_new[0]+1)/2*(H-1)).item())
    # print("imag " , (((xx_new[0]+1)/2*(W-1)*(xx_new[0]+1)/2*(W-1) + ((yy_new[0]+1)/2*(H-1))* ((yy_new[0]+1)/2*(H-1)))/2 ).item())

    # remove singleton dimensions
    input_image_warped = input_image_warped.squeeze()
    vx_warped = vx_warped.squeeze()
    vy_warped = vy_warped.squeeze()

    return input_image_warped, vx_warped, vy_warped


# a = bilinear_interpolation_yuri(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
# a = bilinear_interpolation_yuri(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
# torch.cuda.synchronize()
# start_time = time.time()
# a = bilinear_interpolation_yuri(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
# torch.cuda.synchronize()
# end_time = time.time()
# t1 = end_time-start_time
# print("original ", t1)


# b = ecc_bilinear_interpolation.ecc_bilinear_interpolation(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
# b = ecc_bilinear_interpolation.ecc_bilinear_interpolation(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)

# torch.cuda.synchronize()
# start_time = time.time()
# b = ecc_bilinear_interpolation.ecc_bilinear_interpolation(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
# torch.cuda.synchronize()
# end_time = time.time()
# t2 = end_time-start_time
# print("new ", t2)
# print("speedup by", t1/t2)

# c = ecc_bilinear_interpolation.ecc_bilinear_interpolation_no_grad(input_image, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
# c = ecc_bilinear_interpolation.ecc_bilinear_interpolation_no_grad(input_image, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)


# torch.cuda.synchronize()
# start_time = time.time()
# c = ecc_bilinear_interpolation.ecc_bilinear_interpolation_no_grad(input_image, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
# torch.cuda.synchronize()
# end_time = time.time()
# t3 = end_time-start_time
# print("new_no_grad ", t3)
# print("speedup by", t1/t3)

# a = bilinear_interpolation_yuri(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
# b = ecc_bilinear_interpolation.ecc_bilinear_interpolation(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
# c = ecc_bilinear_interpolation.ecc_bilinear_interpolation_no_grad(input_image, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)

# print()
# print("dudi: ", a[0].item(), " yuri: ",b[0].item(), " yuri_no_grd: ",c[0].item() )
# print("dudi: ", a[1].item(), " yuri: ",b[1].item(), " yuri_no_grd: ",c[1].item() )
# print("dudi: ", a[2].item(), " yuri: ",b[2].item(), " yuri_no_grd: ",c[2].item() )

# a = bilinear_interpolation_yuri(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
# b = ecc_bilinear_interpolation.ecc_bilinear_interpolation(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
# c = ecc_bilinear_interpolation.ecc_bilinear_interpolation_no_grad(input_image, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)


# def bilinear_interpolation_test(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values):

#     ### Calculations: ###
#     T = input_image.shape[0]
#     H = input_image.shape[2]
#     W = input_image.shape[3]

#     H_matrix_corrected = H_matrix.unsqueeze(-1)

#     X_mat_chosen_values_corrected = X_mat_chosen_values.unsqueeze(
#         -1).unsqueeze(-1)
#     Y_mat_chosen_values_corrected = Y_mat_chosen_values.unsqueeze(
#         -1).unsqueeze(-1)
#     denom = (H_matrix_corrected[:, 2:3, 0:1] * X_mat_chosen_values_corrected +
#              H_matrix_corrected[:, 2:3, 1:2] * Y_mat_chosen_values_corrected +
#              H_matrix_corrected[:, 2:3, 2:3])
#     xx_new = 2 * (H_matrix_corrected[:, 0:1, 0:1] * X_mat_chosen_values_corrected +
#                   H_matrix_corrected[:, 0:1, 1:2] * Y_mat_chosen_values_corrected +
#                   H_matrix_corrected[:, 0:1, 2:3]) / denom
#     yy_new = 2 * (H_matrix_corrected[:, 1:2, 0:1] * X_mat_chosen_values_corrected +
#                   H_matrix_corrected[:, 1:2, 1:2] * Y_mat_chosen_values_corrected +
#                   H_matrix_corrected[:, 1:2, 2:3]) / denom

#     # to test the correctness of the calculations we will make the bilinear interpolation on the original image manually
#     #bilinear_grid = torch.cat([xx_new, yy_new], dim=3)
#     sample_inds_x = xx_new.floor()
#     sample_inds_y = yy_new.floor()
#     distance_from_sample_x = sample_inds_x - xx_new
#     distance_from_sample_y = sample_inds_y - yy_new

#     # bilinear interpolation on the original image
#     input_image_warped =


#     # print("x " , ((xx_new[0]+1)/2*(W-1)).item(), " y ", ((yy_new[0]+1)/2*(H-1)).item())
#     # print("imag " , (((xx_new[0]+1)/2*(W-1)*(xx_new[0]+1)/2*(W-1) + ((yy_new[0]+1)/2*(H-1))* ((yy_new[0]+1)/2*(H-1)))/2 ).item())
#     return input_image_warped, vx_warped, vy_warped


def bilinear_interpolation_test(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values):

    ### Calculations: ###
    T = input_image.shape[0]
    H = input_image.shape[2]
    W = input_image.shape[3]

    H_matrix_corrected = H_matrix.unsqueeze(-1)

    X_mat_chosen_values_corrected = X_mat_chosen_values.unsqueeze(
        -1).unsqueeze(-1)
    Y_mat_chosen_values_corrected = Y_mat_chosen_values.unsqueeze(
        -1).unsqueeze(-1)
    denom = (H_matrix_corrected[:, 2:3, 0:1] * X_mat_chosen_values_corrected +
             H_matrix_corrected[:, 2:3, 1:2] * Y_mat_chosen_values_corrected +
             H_matrix_corrected[:, 2:3, 2:3])
    xx_new =  (H_matrix_corrected[:, 0:1, 0:1] * X_mat_chosen_values_corrected +
                  H_matrix_corrected[:, 0:1, 1:2] * Y_mat_chosen_values_corrected +
                  H_matrix_corrected[:, 0:1, 2:3]) / denom
    yy_new =  (H_matrix_corrected[:, 1:2, 0:1] * X_mat_chosen_values_corrected +
                  H_matrix_corrected[:, 1:2, 1:2] * Y_mat_chosen_values_corrected +
                  H_matrix_corrected[:, 1:2, 2:3]) / denom

    # to test the correctness of the calculations we will make the bilinear interpolation on the original image manually
    #bilinear_grid = torch.cat([xx_new, yy_new], dim=3)
    # sample_inds_x = xx_new.floor()
    # sample_inds_y = yy_new.floor()
    # distance_from_sample_x = sample_inds_x - xx_new
    # distance_from_sample_y = sample_inds_y - yy_new

    # bilinear interpolation on the original image
    # remove singletone dimensions
    xx_new = xx_new.squeeze()+1
    yy_new = yy_new.squeeze()+1

    func_vals = xx_new*yy_new

    # print("x " , ((xx_new[0]+1)/2*(W-1)).item(), " y ", ((yy_new[0]+1)/2*(H-1)).item())
    # print("imag " , (((xx_new[0]+1)/2*(W-1)*(xx_new[0]+1)/2*(W-1) + ((yy_new[0]+1)/2*(H-1))* ((yy_new[0]+1)/2*(H-1)))/2 ).item())
    return func_vals, xx_new, yy_new


a = bilinear_interpolation_yuri(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
b = ecc_bilinear_interpolation.ecc_bilinear_interpolation(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
c = ecc_bilinear_interpolation.ecc_bilinear_interpolation_no_grad(input_image, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)
d = bilinear_interpolation_test(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)


# print("dudi: ", (b[0]-d[0]).abs().max().item()  )
# print("yuri: ", (b[0]).abs().max().item())
# print("yuri_no_grd: ", (c[0]-d[0]).abs().max().item()  )

# print("dudi: ", a[0].item(), " yuri: ",b[0].item(), " yuri_no_grd: ",c[0].item(), " true: ", d[0].item() )
# print("dudi: ", a[1].item(), " yuri: ",b[1].item(), " yuri_no_grd: ",c[1].item(), " true: ", d[1].item() )
# print("dudi: ", a[2].item(), " yuri: ",b[2].item(), " yuri_no_grd: ",c[2].item(), " true: ", d[2].item() )

print("dudi: ", (2.0*(a[0]-d[0])/(a[0]+d[0])).abs().max().item(), " yuri: ", (2.0*(b[0]-d[0])/(b[0]+d[0])).abs().max().item() , " yuri_no_grd: ", (2.0*(c[0]-d[0])/(c[0]+d[0])).abs().max().item()  )
print("dudi: ", (2.0*(a[1]-d[1])/(a[1]+d[1])).abs().max().item(), " yuri: ", (2.0*(b[1]-d[1])/(b[1]+d[1])).abs().max().item() , " yuri_no_grd: ", (2.0*(c[1]-d[1])/(c[1]+d[1])).abs().max().item()  )
print("dudi: ", (2.0*(a[2]-d[2])/(a[2]+d[2])).abs().max().item(), " yuri: ", (2.0*(b[2]-d[2])/(b[2]+d[2])).abs().max().item() , " yuri_no_grd: ", (2.0*(c[2]-d[2])/(c[2]+d[2])).abs().max().item()  )

