
import torch
import calc_delta_p
import calc_delta_p_v3
import time

# Example Values: ###  #TODO: delete!!!
T = 25
N = 25000
H_matrix = torch.randn((T, 3, 3)).cuda()
H_matrix[:, 2, 2] = 1
gx_chosen_values = torch.randn((T, N)).cuda()
gy_chosen_values = torch.randn((T, N)).cuda()
Jx_chosen_values = torch.randn((T, N)).cuda()
Jy_chosen_values = torch.randn((T, N)).cuda()
current_level_reference_tensor_zero_mean = (torch.randn(
    (1, N)).cuda()).repeat(T, 1)  # this should be (1 ,N)
current_level_input_tensor_warped = torch.randn((T, N)).cuda()


def images_and_gradients_to_delta_p_yuri(H_matrix,
                                         current_level_reference_tensor_zero_mean,
                                         current_level_input_tensor_warped,
                                         Jx_chosen_values, Jy_chosen_values,
                                         gx_chosen_values, gy_chosen_values):
    ### [H_matrix] = [T,3,3]
    ### [Jx_chosen_values] = [T,N]
    ### [Jy_chosen_values] = [T,N]
    ### [gx_chosen_values] = [T,N]
    ### [gy_chosen_values] = [T,N]
    ### [current_level_reference_tensor_zero_mean] = [T,N]
    ### [current_level_input_tensor_warped] = [T,N]

    ### Correct dimensions for pytorch arithmatic: ###
    Jx_chosen_values = Jx_chosen_values.unsqueeze(
        1).unsqueeze(1)  # -> [T,N,1,1]
    Jy_chosen_values = Jy_chosen_values.unsqueeze(
        1).unsqueeze(1)  # -> [T,N,1,1]
    H_matrix_corrected = H_matrix.unsqueeze(-1)  # -> [T,3,3,1]
    ### Calculate den once: ###

    den = (H_matrix_corrected[:, 2:3, 0:1] * Jx_chosen_values +
           H_matrix_corrected[:, 2:3, 1:2] * Jy_chosen_values +
           H_matrix_corrected[:, 2:3, 2:3])
    denom_inverse = 1 / den

    ### H Transform xy_prime values: ###
    xy_prime_reshaped_X = (H_matrix_corrected[:, 0:1, 0:1] * Jx_chosen_values +
                           H_matrix_corrected[:, 0:1, 1:2] * Jy_chosen_values +
                           H_matrix_corrected[:, 0:1, 2:3]) * denom_inverse
    xy_prime_reshaped_Y = (H_matrix_corrected[:, 1:2, 0:1] * Jx_chosen_values +
                           H_matrix_corrected[:, 1:2, 1:2] * Jy_chosen_values +
                           H_matrix_corrected[:, 1:2, 2:3]) * denom_inverse

    ### Correct Jx,Jy values: ###
    Jx_chosen_values = Jx_chosen_values * denom_inverse  # element-wise
    Jy_chosen_values = Jy_chosen_values * denom_inverse  # element-wise

    #### Get final Jxx,Jxy,Jyy,Jyx values: ####
    Jxx_prime = Jx_chosen_values * xy_prime_reshaped_X  # element-wise.
    Jyx_prime = Jy_chosen_values * xy_prime_reshaped_X
    Jxy_prime = Jx_chosen_values * xy_prime_reshaped_Y  # element-wise
    Jyy_prime = Jy_chosen_values * xy_prime_reshaped_Y

    # ### Get final jacobian of the H_matrix with respect to the different parameters: ###
    # J_list = [Jx_chosen_values, Jy_chosen_values, Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime]

    ### Yuri calculations: ###
    current_level_reference_tensor_zero_mean = current_level_reference_tensor_zero_mean.unsqueeze(
        1).unsqueeze(1)  # ->[T,1,1,N]
    current_level_input_tensor_warped = current_level_input_tensor_warped.unsqueeze(
        1).unsqueeze(1)  # ->[T,1,1,N]
    gx_chosen_values = gx_chosen_values.unsqueeze(
        1).unsqueeze(1)  # ->[T,1,1,N]
    gy_chosen_values = gy_chosen_values.unsqueeze(
        1).unsqueeze(1)  # ->[T,1,1,N]

    delta_p = calc_delta_p.ecc_calc_delta_p(gx_chosen_values, gy_chosen_values,
                                            Jx_chosen_values, Jy_chosen_values,
                                            Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime,
                                            current_level_reference_tensor_zero_mean,
                                            current_level_input_tensor_warped)
    return delta_p


delta_p1 = images_and_gradients_to_delta_p_yuri(H_matrix,
                                                current_level_reference_tensor_zero_mean,
                                                current_level_input_tensor_warped,
                                                Jx_chosen_values, Jy_chosen_values,
                                                gx_chosen_values, gy_chosen_values)


delta_p1 = images_and_gradients_to_delta_p_yuri(H_matrix,
                                                current_level_reference_tensor_zero_mean,
                                                current_level_input_tensor_warped,
                                                Jx_chosen_values, Jy_chosen_values,
                                                gx_chosen_values, gy_chosen_values)

torch.cuda.synchronize()
start_time = time.time()
delta_p1 = images_and_gradients_to_delta_p_yuri(H_matrix,
                                                current_level_reference_tensor_zero_mean,
                                                current_level_input_tensor_warped,
                                                Jx_chosen_values, Jy_chosen_values,
                                                gx_chosen_values, gy_chosen_values)

delta_p1 = images_and_gradients_to_delta_p_yuri(H_matrix,
                                                current_level_reference_tensor_zero_mean,
                                                current_level_input_tensor_warped,
                                                Jx_chosen_values, Jy_chosen_values,
                                                gx_chosen_values, gy_chosen_values)

end_time = time.time()
t1 = end_time-start_time
print("original ", t1)


delta_p2 = calc_delta_p_v3.ecc_calc_delta_p(H_matrix,
                                            current_level_reference_tensor_zero_mean,
                                            current_level_input_tensor_warped,
                                            Jx_chosen_values, Jy_chosen_values,
                                            gx_chosen_values, gy_chosen_values)

delta_p2 = calc_delta_p_v3.ecc_calc_delta_p(H_matrix,
                                            current_level_reference_tensor_zero_mean,
                                            current_level_input_tensor_warped,
                                            Jx_chosen_values, Jy_chosen_values,
                                            gx_chosen_values, gy_chosen_values)


torch.cuda.synchronize()
start_time = time.time()
delta_p2 = calc_delta_p_v3.ecc_calc_delta_p(H_matrix,
                                            current_level_reference_tensor_zero_mean,
                                            current_level_input_tensor_warped,
                                            Jx_chosen_values, Jy_chosen_values,
                                            gx_chosen_values, gy_chosen_values)

delta_p2 = calc_delta_p_v3.ecc_calc_delta_p(H_matrix,
                                            current_level_reference_tensor_zero_mean,
                                            current_level_input_tensor_warped,
                                            Jx_chosen_values, Jy_chosen_values,
                                            gx_chosen_values, gy_chosen_values)

end_time = time.time()
t2 = end_time-start_time
print("new ", t2)
print("speedup by", t1/t2)

# print(delta_p1)
# print(delta_p2)
# print ( (0.5*(delta_p2-delta_p1)/ (delta_p2.abs()+delta_p1.abs())).abs())


print("foo")
