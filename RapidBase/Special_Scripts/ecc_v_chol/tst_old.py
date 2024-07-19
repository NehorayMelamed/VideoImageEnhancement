import calc_delta_p
import time
import torch
import ecc_reduction

T = 25
In_sz = 25000
gx = torch.randn(T, 1, 1, In_sz, dtype=torch.float32).cuda()
gy_chosen_values = torch.randn(T, 1, 1, In_sz, dtype=torch.float32).cuda()
gx_chosen_values = torch.randn(T, 1, 1, In_sz, dtype=torch.float32).cuda()
Jx = torch.randn(T, 1, 1, In_sz, dtype=torch.float32).cuda()
Jy = torch.randn(T, 1, 1, In_sz, dtype=torch.float32).cuda()
# It faster to have these as tensors g, g_chosen J, J_jacob, such that the last index is the x,y etc. to achive localaity
Jxx_prime = torch.randn(T, 1, 1, In_sz, dtype=torch.float32).cuda()
Jxy_prime = torch.randn(T, 1, 1, In_sz, dtype=torch.float32).cuda()
Jyx_prime = torch.randn(T, 1, 1, In_sz, dtype=torch.float32).cuda()
Jyy_prime = torch.randn(T, 1, 1, In_sz, dtype=torch.float32).cuda()
current_level_reference_tensor_zero_mean = torch.randn(
    1, 1, 1, In_sz, dtype=torch.float32).cuda()
current_level_input_tensor_warped = torch.randn(
    T, 1, 1, In_sz, dtype=torch.float32).cuda()


Gt = torch.zeros((T, 8, 1)).to(current_level_reference_tensor_zero_mean.device)
Gw = torch.zeros((T, 8, 1)).to(current_level_reference_tensor_zero_mean.device)
C = torch.zeros((T, 8, 8)).to(gx_chosen_values.device)


def calculations(gx_chosen_values, gy_chosen_values, Jx, Jy,
                 Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime,
                 current_level_reference_tensor_zero_mean, current_level_input_tensor_warped, Gt, Gw, C):

    G0 = gx_chosen_values * Jx
    G1 = gy_chosen_values * Jx
    G2 = -gx_chosen_values * Jxx_prime - gy_chosen_values * Jxy_prime
    G3 = gx_chosen_values * Jy
    G4 = gy_chosen_values * Jy
    G5 = -gx_chosen_values * Jyx_prime - gy_chosen_values * Jyy_prime
    G6 = gx_chosen_values
    G7 = gy_chosen_values

    # TODO: understand if making a list here takes time?!??
    G_list = [G0,
              G1,
              G2,
              G3,
              G4,
              G5,
              G6,
              G7]

    Gt[:, 0] = (G0 * current_level_reference_tensor_zero_mean).sum([-1, -2])
    Gt[:, 1] = (G1 * current_level_reference_tensor_zero_mean).sum([-1, -2])
    Gt[:, 2] = (G2 * current_level_reference_tensor_zero_mean).sum([-1, -2])
    Gt[:, 3] = (G3 * current_level_reference_tensor_zero_mean).sum([-1, -2])
    Gt[:, 4] = (G4 * current_level_reference_tensor_zero_mean).sum([-1, -2])
    Gt[:, 5] = (G5 * current_level_reference_tensor_zero_mean).sum([-1, -2])
    Gt[:, 6] = (G6 * current_level_reference_tensor_zero_mean).sum([-1, -2])
    Gt[:, 7] = (G7 * current_level_reference_tensor_zero_mean).sum([-1, -2])
    # (*). Calculate Gw:
    Gw[:, 0] = (G0 * current_level_input_tensor_warped).sum([-1, -2])
    Gw[:, 1] = (G1 * current_level_input_tensor_warped).sum([-1, -2])
    Gw[:, 2] = (G2 * current_level_input_tensor_warped).sum([-1, -2])
    Gw[:, 3] = (G3 * current_level_input_tensor_warped).sum([-1, -2])
    Gw[:, 4] = (G4 * current_level_input_tensor_warped).sum([-1, -2])
    Gw[:, 5] = (G5 * current_level_input_tensor_warped).sum([-1, -2])
    Gw[:, 6] = (G6 * current_level_input_tensor_warped).sum([-1, -2])
    Gw[:, 7] = (G7 * current_level_input_tensor_warped).sum([-1, -2])

    ### PreCalculate C=(Gt*G): ###
    # TODO: would be smart to combine everything here together in the same memory run
    # TODO: make this batch-operations

    C[:, 0, 0] = (G0 * G0).sum([-1, -2, -3])
    C[:, 0, 1] = (G0 * G1).sum([-1, -2, -3])
    C[:, 0, 2] = (G0 * G2).sum([-1, -2, -3])
    C[:, 0, 3] = (G0 * G3).sum([-1, -2, -3])
    C[:, 0, 4] = (G0 * G4).sum([-1, -2, -3])
    C[:, 0, 5] = (G0 * G5).sum([-1, -2, -3])
    C[:, 0, 6] = (G0 * G6).sum([-1, -2, -3])
    C[:, 0, 7] = (G0 * G7).sum([-1, -2, -3])
    #
    C[:, 1, 0] = (G1 * G0).sum([-1, -2, -3])
    C[:, 1, 1] = (G1 * G1).sum([-1, -2, -3])
    C[:, 1, 2] = (G1 * G2).sum([-1, -2, -3])
    C[:, 1, 3] = (G1 * G3).sum([-1, -2, -3])
    C[:, 1, 4] = (G1 * G4).sum([-1, -2, -3])
    C[:, 1, 5] = (G1 * G5).sum([-1, -2, -3])
    C[:, 1, 6] = (G1 * G6).sum([-1, -2, -3])
    C[:, 1, 7] = (G1 * G7).sum([-1, -2, -3])
    #
    C[:, 2, 0] = (G2 * G0).sum([-1, -2, -3])
    C[:, 2, 1] = (G2 * G1).sum([-1, -2, -3])
    C[:, 2, 2] = (G2 * G2).sum([-1, -2, -3])
    C[:, 2, 3] = (G2 * G3).sum([-1, -2, -3])
    C[:, 2, 4] = (G2 * G4).sum([-1, -2, -3])
    C[:, 2, 5] = (G2 * G5).sum([-1, -2, -3])
    C[:, 2, 6] = (G2 * G6).sum([-1, -2, -3])
    C[:, 2, 7] = (G2 * G7).sum([-1, -2, -3])
    #
    C[:, 3, 0] = (G3 * G0).sum([-1, -2, -3])
    C[:, 3, 1] = (G3 * G1).sum([-1, -2, -3])
    C[:, 3, 2] = (G3 * G2).sum([-1, -2, -3])
    C[:, 3, 3] = (G3 * G3).sum([-1, -2, -3])
    C[:, 3, 4] = (G3 * G4).sum([-1, -2, -3])
    C[:, 3, 5] = (G3 * G5).sum([-1, -2, -3])
    C[:, 3, 6] = (G3 * G6).sum([-1, -2, -3])
    C[:, 3, 7] = (G3 * G7).sum([-1, -2, -3])
    #
    C[:, 4, 0] = (G4 * G0).sum([-1, -2, -3])
    C[:, 4, 1] = (G4 * G1).sum([-1, -2, -3])
    C[:, 4, 2] = (G4 * G2).sum([-1, -2, -3])
    C[:, 4, 3] = (G4 * G3).sum([-1, -2, -3])
    C[:, 4, 4] = (G4 * G4).sum([-1, -2, -3])
    C[:, 4, 5] = (G4 * G5).sum([-1, -2, -3])
    C[:, 4, 6] = (G4 * G6).sum([-1, -2, -3])
    C[:, 4, 7] = (G4 * G7).sum([-1, -2, -3])
    #
    C[:, 5, 0] = (G5 * G0).sum([-1, -2, -3])
    C[:, 5, 1] = (G5 * G1).sum([-1, -2, -3])
    C[:, 5, 2] = (G5 * G2).sum([-1, -2, -3])
    C[:, 5, 3] = (G5 * G3).sum([-1, -2, -3])
    C[:, 5, 4] = (G5 * G4).sum([-1, -2, -3])
    C[:, 5, 5] = (G5 * G5).sum([-1, -2, -3])
    C[:, 5, 6] = (G5 * G6).sum([-1, -2, -3])
    C[:, 5, 7] = (G5 * G7).sum([-1, -2, -3])
    #
    C[:, 6, 0] = (G6 * G0).sum([-1, -2, -3])
    C[:, 6, 1] = (G6 * G1).sum([-1, -2, -3])
    C[:, 6, 2] = (G6 * G2).sum([-1, -2, -3])
    C[:, 6, 3] = (G6 * G3).sum([-1, -2, -3])
    C[:, 6, 4] = (G6 * G4).sum([-1, -2, -3])
    C[:, 6, 5] = (G6 * G5).sum([-1, -2, -3])
    C[:, 6, 6] = (G6 * G6).sum([-1, -2, -3])
    C[:, 6, 7] = (G6 * G7).sum([-1, -2, -3])
    #
    C[:, 7, 0] = (G7 * G0).sum([-1, -2, -3])
    C[:, 7, 1] = (G7 * G1).sum([-1, -2, -3])
    C[:, 7, 2] = (G7 * G2).sum([-1, -2, -3])
    C[:, 7, 3] = (G7 * G3).sum([-1, -2, -3])
    C[:, 7, 4] = (G7 * G4).sum([-1, -2, -3])
    C[:, 7, 5] = (G7 * G5).sum([-1, -2, -3])
    C[:, 7, 6] = (G7 * G6).sum([-1, -2, -3])
    C[:, 7, 7] = (G7 * G7).sum([-1, -2, -3])

    ### New Code: ###
    ### Coompute Hessian and its inverse: ###
    i_C = torch.linalg.inv(C)

    ### ECC Closed Form Solution: ###
    # (1). compute lambda parameter:
    num = (torch.linalg.norm(current_level_input_tensor_warped, dim=(-1, -2))
           ).unsqueeze(-1) ** 2 - torch.transpose(Gw, -1, -2) @ i_C @ Gw
    den = (current_level_input_tensor_warped * current_level_reference_tensor_zero_mean).sum(
        [-1, 2]).unsqueeze(-1) - torch.transpose(Gt, -1, -2) @ i_C @ Gw
    lambda_correction = (num / den).unsqueeze(-1)

    # (2). compute error vector:
    imerror = lambda_correction * current_level_reference_tensor_zero_mean - \
        current_level_input_tensor_warped

    # (3). compute the projection of error vector into Jacobian G:
    Ge = torch.empty([T, 8], dtype=torch.float32).cuda()

    Ge[:, 0] = (G0 * imerror).sum([-1, -2]).squeeze()
    Ge[:, 1] = (G1 * imerror).sum([-1, -2]).squeeze()
    Ge[:, 2] = (G2 * imerror).sum([-1, -2]).squeeze()
    Ge[:, 3] = (G3 * imerror).sum([-1, -2]).squeeze()
    Ge[:, 4] = (G4 * imerror).sum([-1, -2]).squeeze()
    Ge[:, 5] = (G5 * imerror).sum([-1, -2]).squeeze()
    Ge[:, 6] = (G6 * imerror).sum([-1, -2]).squeeze()
    Ge[:, 7] = (G7 * imerror).sum([-1, -2]).squeeze()

    # (4). compute the optimum parameter correction vector:
    delta_p = torch.matmul(i_C, Ge.unsqueeze(-1))

    return delta_p


#####################################
delta_p1 = calculations(gx_chosen_values, gy_chosen_values, Jx, Jy, Jxx_prime,
                        Jxy_prime, Jyx_prime, Jyy_prime, current_level_reference_tensor_zero_mean,
                        current_level_input_tensor_warped, Gt, Gw, C)


#print(Gto.shape,Gwo.shape, Co.shape)
torch.cuda.synchronize()
start_time = time.time()
delta_p1 = calculations(gx_chosen_values, gy_chosen_values, Jx, Jy, Jxx_prime,
                        Jxy_prime, Jyx_prime, Jyy_prime, current_level_reference_tensor_zero_mean,
                        current_level_input_tensor_warped, Gt, Gw, C)


torch.cuda.synchronize()
end_time = time.time()
t1 = end_time-start_time
print("original ", t1)


#####################################
delta_p = calc_delta_p.ecc_calc_delta_p(gx_chosen_values, gy_chosen_values, Jx, Jy, Jxx_prime,
                                        Jxy_prime, Jyx_prime, Jyy_prime, current_level_reference_tensor_zero_mean,
                                        current_level_input_tensor_warped)


torch.cuda.synchronize()
start_time = time.time()
# G, Gt, Gw, C = ecc_reduction.ecc_reduction(gx_chosen_values, gy_chosen_values, Jx, Jy, Jxx_prime,
#                                            Jxy_prime, Jyx_prime, Jyy_prime, current_level_reference_tensor_zero_mean,
#                                            current_level_input_tensor_warped)

# i_C = torch.linalg.inv(C)

# num = (torch.linalg.norm(current_level_input_tensor_warped, dim=(-1, -2))).unsqueeze(-1) ** 2 - torch.transpose(Gw, -1, -2) @ i_C @ Gw
# den = (current_level_input_tensor_warped * current_level_reference_tensor_zero_mean).sum([-1, 2]).unsqueeze(-1) - torch.transpose(Gt, -1, -2) @ i_C @ Gw
# lambda_correction = (num / den).unsqueeze(-1)

# # (2). compute error vector:
# imerror = lambda_correction * current_level_reference_tensor_zero_mean - \
#     current_level_input_tensor_warped

# Ge = (G * imerror.squeeze().unsqueeze(-1)).sum([-2])
# delta_p = torch.matmul(i_C, Ge.unsqueeze(-1))


delta_p = calc_delta_p.ecc_calc_delta_p(gx_chosen_values, gy_chosen_values, Jx, Jy, Jxx_prime,
                                        Jxy_prime, Jyx_prime, Jyy_prime, current_level_reference_tensor_zero_mean,
                                        current_level_input_tensor_warped)

torch.cuda.synchronize()
end_time = time.time()
t2 = end_time-start_time
print("new ", t2)

#####################################


print("speedup by", t1/t2)

print(((delta_p1-delta_p)/delta_p).abs().max())

#print( (Co-d[2])/d[2] )
# print((Gto-d[0].reshape([T, 8, 1])).abs().max())
# print((Gwo-d[1].reshape([T, 8, 1])).abs().max())
# print((Co-d[2]).abs().max())

# print(d[0])


# print(d[0].shape,d[1].shape,d[2].shape)
# print(Gw.shape)
#print(Gw[1, 1, 1])
# print(Co[0,0,0])
# print(d[0][0,0]) #,Gto[0,0,0]
# print(d[1][0,0])
# print(d[2][0,0])
