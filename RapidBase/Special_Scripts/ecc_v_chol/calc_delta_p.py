import torch, sys
# sys.path.append()
import ecc_reduction


def ecc_calc_delta_p(gx_chosen_values, gy_chosen_values, Jx, Jy, Jxx_prime,
                     Jxy_prime, Jyx_prime, Jyy_prime, current_level_reference_tensor_zero_mean,
                     current_level_input_tensor_warped):


    G, Gt, Gw, C = ecc_reduction.ecc_reduction(gx_chosen_values, gy_chosen_values, Jx, Jy, Jxx_prime,
                                            Jxy_prime, Jyx_prime, Jyy_prime, current_level_reference_tensor_zero_mean,
                                            current_level_input_tensor_warped)
    # print(C[:,::9])
    # print(G) 
    i_C = torch.linalg.inv(C)

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

    return G, Gt, Gw, C ,delta_p