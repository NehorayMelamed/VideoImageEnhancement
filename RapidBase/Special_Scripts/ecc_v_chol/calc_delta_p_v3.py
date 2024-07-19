import torch
import sys
import time
# sys.path.append()
import ecc_reduction_v3


def ecc_calc_delta_p(H_matrix,
                     current_level_reference_tensor_zero_mean,
                     current_level_input_tensor_warped,
                     Jx_chosen_values, Jy_chosen_values,
                     gx_chosen_values, gy_chosen_values):

    G, Gt, Gw, C = ecc_reduction_v3.ecc_reduction(H_matrix,
                                                  current_level_reference_tensor_zero_mean,
                                                  current_level_input_tensor_warped,
                                                  Jx_chosen_values, Jy_chosen_values,
                                                  gx_chosen_values, gy_chosen_values)

    # print(C[:,::9])
    i_C = torch.linalg.inv(C)

    current_level_reference_tensor_zero_mean = current_level_reference_tensor_zero_mean.unsqueeze(
        1).unsqueeze(1)  # ->[T,1,1,N]
    current_level_input_tensor_warped = current_level_input_tensor_warped.unsqueeze(
        1).unsqueeze(1)  # ->[T,1,1,N]

    num = (torch.linalg.norm(current_level_input_tensor_warped, dim=(-1, -2))
           ).unsqueeze(-1) ** 2 - torch.transpose(Gw, -1, -2) @ i_C @ Gw
    den = (current_level_input_tensor_warped * current_level_reference_tensor_zero_mean).sum(
        [-1, 2]).unsqueeze(-1) - torch.transpose(Gt, -1, -2) @ i_C @ Gw
    lambda_correction = (num / den).unsqueeze(-1)

    # (2). compute error vector:
    imerror = lambda_correction * current_level_reference_tensor_zero_mean - \
        current_level_input_tensor_warped

    Ge = (G * imerror.squeeze().unsqueeze(-1)).sum([-2])
    delta_p = torch.matmul(i_C, Ge.unsqueeze(-1))

    # (torch.linalg.norm(current_level_input_tensor_warped, dim=(-1, -2))).unsqueeze(-1) ** 2
    # torch.cuda.synchronize()
    # start_time = time.time()
    # (torch.linalg.norm(current_level_input_tensor_warped, dim=(-1, -2))).unsqueeze(-1) ** 2
    # torch.cuda.synchronize()
    # end_time = time.time()
    # print("Time taken for redu: ", end_time - start_time)

    # torch.transpose(Gw, -1, -2) @ i_C @ Gw
    # torch.cuda.synchronize()
    # start_time = time.time()
    # torch.transpose(Gw, -1, -2) @ i_C @ Gw
    # torch.cuda.synchronize()
    # end_time = time.time()
    # print("Time taken for mat prod: ", end_time - start_time)

    # lambda_correction * current_level_reference_tensor_zero_mean - current_level_input_tensor_warped
    # torch.cuda.synchronize()
    # start_time = time.time()
    # lambda_correction * current_level_reference_tensor_zero_mean - current_level_input_tensor_warped
    # torch.cuda.synchronize()
    # end_time = time.time()
    # print("Time taken for lambda corr: ", end_time - start_time)

    return G, Gt, Gw, C ,delta_p