import torch
### Example Stuff: ###
T = 25
H = 540
W = 8192
N = 25000

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




input_image = torch.rand(T, 1, H, W).cuda()
vx = torch.rand(T, 1, H, W).cuda()
vy = torch.rand(T, 1, H, W).cuda()
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

    return input_image_warped, vx_warped, vy_warped


a= bilinear_interpolation_yuri(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values)