import torch
import timeit
import numpy as np
import time
import ecc_reduction

T = 500
In_sz = 20000
gx = torch.randn(T, 1, 1, In_sz, dtype=torch.float).cuda()
gy_chosen_values = torch.randn(T, 1, 1, In_sz, dtype=torch.float).cuda()
gx_chosen_values = torch.randn(T, 1, 1, In_sz, dtype=torch.float).cuda()
Jx = torch.randn(T, 1, 1, In_sz, dtype=torch.float).cuda()
Jy = torch.randn(T, 1, 1, In_sz, dtype=torch.float).cuda()
# It faster to have these as tensors g, g_c J, J_jacob, such that the last index is the x,y etc. to achive localaity
Jxx_prime = torch.randn(T, 1, 1, In_sz, dtype=torch.float).cuda()
Jxy_prime = torch.randn(T, 1, 1, In_sz, dtype=torch.float).cuda()
Jyx_prime = torch.randn(T, 1, 1, In_sz, dtype=torch.float).cuda()
Jyy_prime = torch.randn(T, 1, 1, In_sz, dtype=torch.float).cuda()
current_level_reference_tensor_zero_mean = torch.randn(
    T, 1, 1, In_sz, dtype=torch.float).cuda()
current_level_input_tensor_warped = torch.randn(
    T, 1, 1, In_sz, dtype=torch.float).cuda()
Gt = torch.zeros((T, 8, 1), dtype=torch.float).to(
    current_level_reference_tensor_zero_mean.device)
Gw = torch.zeros((T, 8, 1), dtype=torch.float).to(
    current_level_reference_tensor_zero_mean.device)
C = torch.zeros((T, 8, 8), dtype=torch.float).to(gx_chosen_values.device)


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

    return C, Gt, Gw


NN = 10
# print(timeit.timeit("a,b,c = calculations(gx_chosen_values, gy_chosen_values, Jx, Jy, Jxx_prime,\
#                      Jxy_prime, Jyx_prime, Jyy_prime, current_level_reference_tensor_zero_mean,\
#                      current_level_input_tensor_warped, Gt, Gw, C)", globals=globals(), number=NN)/NN)

start = time.time()
for x in np.arange(1, NN):
    C, Gt, Gw = calculations(gx_chosen_values, gy_chosen_values, Jx, Jy, Jxx_prime,
                         Jxy_prime, Jyx_prime, Jyy_prime, current_level_reference_tensor_zero_mean,
                         current_level_input_tensor_warped, Gt, Gw, C)
end = time.time()
print( (end - start)/NN)
####
# print(current_level_reference_tensor_zero_mean.data_ptr())
# print(current_level_input_tensor_warped.data_ptr())

start = time.time()
for x in np.arange(1, NN):
    d = ecc_reduction.ecc_reduction(gx_chosen_values, gy_chosen_values,
                                    Jx, Jy,
                                    Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime,
                                    current_level_reference_tensor_zero_mean,
                                    current_level_input_tensor_warped)

end = time.time()
print( (end - start)/NN)
# print(d[0].size()," ",d[1].size()," ",d[2].size())
# print(Gt[0])

# print((gx_chosen_values.max(), gy_chosen_values.max(), Jx.max(), Jy.max(), Jxx_prime.max(),\
#                      Jxy_prime.max(), Jyx_prime.max(), Jyy_prime.max(), current_level_reference_tensor_zero_mean.max(),\
#                      current_level_input_tensor_warped.max()))
# print(a.size(),b.size(),c.size())
# print(d[0].size(),d[1].size(),d[2].size())

# print(torch.abs(Gt).max(),torch.abs(d[0]).max(),torch.abs(Gt-d[0]).max())
# print(torch.abs(Gw).max(),torch.abs(d[1]).max(),torch.abs(Gw-d[1]).max())
# print(torch.abs(C).max(),torch.abs(d[2]).max(),torch.abs(C-d[2]).max())
# print(torch.abs(c).max(),torch.abs(d[0]).max(),torch.abs(c-d[0]).max())
# print(torch.abs(b-d[0]).max())
# print(timeit.timeit("d1 = ecc_reduction.ecc_reduction(gx_chosen_values, gy_chosen_values, Jx, Jy, Jxx_prime,\
#                     Jxy_prime, Jyx_prime, Jyy_prime, current_level_reference_tensor_zero_mean,\
#                     current_level_input_tensor_warped, Gt, Gw, C)", globals=globals(), number=NN)/NN)

# C, Gt, Gw = calculations(gx_chosen_values, gy_chosen_values, Jx, Jy,
#                  Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime,
#                  current_level_reference_tensor_zero_mean, current_level_input_tensor_warped, Gt, Gw, C)

C, Gt, Gw = calculations(gx_chosen_values, gy_chosen_values, Jx, Jy, Jxx_prime,
                         Jxy_prime, Jyx_prime, Jyy_prime, current_level_reference_tensor_zero_mean,
                         current_level_input_tensor_warped, Gt, Gw, C)

d = ecc_reduction.ecc_reduction(gx_chosen_values, gy_chosen_values,
                                Jx, Jy,
                                Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime,
                                current_level_reference_tensor_zero_mean,
                                current_level_input_tensor_warped)

print(torch.abs(Gt).max(), torch.abs(d[0]).max(), torch.abs(Gt-d[0].cuda()).max())
print(torch.abs(Gw).max(), torch.abs(d[1]).max(), torch.abs(Gw-d[1].cuda()).max())
print(torch.abs(C).max(), torch.abs(d[2]).max(), torch.abs(C-d[2].cuda()).max())
