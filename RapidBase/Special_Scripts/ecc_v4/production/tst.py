
import torch
import calc_delta_p
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



delta_p2 = calc_delta_p.ecc_calc_delta_p(H_matrix,
                                            current_level_reference_tensor_zero_mean,
                                            current_level_input_tensor_warped,
                                            Jx_chosen_values, Jy_chosen_values,
                                            gx_chosen_values, gy_chosen_values)

delta_p2 = calc_delta_p.ecc_calc_delta_p(H_matrix,
                                            current_level_reference_tensor_zero_mean,
                                            current_level_input_tensor_warped,
                                            Jx_chosen_values, Jy_chosen_values,
                                            gx_chosen_values, gy_chosen_values)


torch.cuda.synchronize()
start_time = time.time()
delta_p2 = calc_delta_p.ecc_calc_delta_p(H_matrix,
                                            current_level_reference_tensor_zero_mean,
                                            current_level_input_tensor_warped,
                                            Jx_chosen_values, Jy_chosen_values,
                                            gx_chosen_values, gy_chosen_values)

delta_p2 = calc_delta_p.ecc_calc_delta_p(H_matrix,
                                            current_level_reference_tensor_zero_mean,
                                            current_level_input_tensor_warped,
                                            Jx_chosen_values, Jy_chosen_values,
                                            gx_chosen_values, gy_chosen_values)
torch.cuda.synchronize()
end_time = time.time()
t2 = (end_time-start_time)/2
print("time ", t2)


# print(delta_p1)
# print(delta_p2)
# print ( (0.5*(delta_p2-delta_p1)/ (delta_p2.abs()+delta_p1.abs())).abs())

# print(delta_p2)
# print("foo")
