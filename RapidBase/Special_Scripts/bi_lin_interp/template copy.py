### Example Stuff: ###
T = 25
H = 540
W = 8192
N = 25000
H_matrix = torch.randn((T, 3, 3)).cuda()
input_image = torch.randn(T, 1, H, W).cuda()
vx = torch.randn(T, 1, H, W).cuda()
vy = torch.randn(T, 1, H, W).cuda()
X_mat_chosen_values = torch.randn(T, N).cuda()
Y_mat_chosen_values = torch.randn(T, N).cuda()


def bilinear_interpolation_yuri(input_image, vx, vy, H_matrix, X_mat_chosen_values, Y_mat_chosen_values):

    ### Calculations: ###


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
