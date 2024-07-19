import torch, sys
import numpy as np
import cv2
import torchvision
import time


with open("/home/mafat/Documents/tst/outliers.bin", 'rb') as outliers_file:
    full_arr = np.fromfile(outliers_file, dtype=np.ubyte).reshape((500, 500, 8000))
outliers = torch.from_numpy(full_arr).cuda()



outliers_cpu = outliers.cpu()
outliers_np = (outliers_cpu.numpy())

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(outliers_np[0,:,:], 8, cv2.CV_16U)

start_time = time.time()
for ind in range(outliers_np.shape[0]):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(outliers_np[ind,:,:], 8, cv2.CV_16U)

end_time = time.time()
print("regular ",end_time-start_time)


start_time = time.time()
for ind in range(outliers_np.shape[0]):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(outliers_np[ind,:,:], 8, cv2.CV_16U,cv2.CCL_DEFAULT)
end_time = time.time()
print("DEFAULT ",end_time-start_time)

start_time = time.time()
for ind in range(outliers_np.shape[0]):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(outliers_np[ind,:,:], 8, cv2.CV_16U,cv2.CCL_WU)
end_time = time.time()
print("WU ",end_time-start_time)

start_time = time.time()
for ind in range(outliers_np.shape[0]):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(outliers_np[ind,:,:], 8, cv2.CV_16U,cv2.CCL_GRANA)
end_time = time.time()
print("GRANA ",end_time-start_time)


start_time = time.time()
for ind in range(outliers_np.shape[0]):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(outliers_np[ind,:,:], 8, cv2.CV_16U,cv2.CCL_BOLELLI)
end_time = time.time()
print("GRANA ",end_time-start_time)

# input = torch.ones([500,500,8000],device='cuda',dtype=torch.uint8)


# input[:,::3,:]=0
# input[:,:,::3]=0 
# input[0,3,0]=1 

# #print(input)
# #CCL.sup_dude()
# #CCL.dummy(input)
# import time
# start_time = time.time()
# output=torch.ops.CCL.CCL(input)
# output=torch.ops.CCL.CCL(input)
# output=torch.ops.CCL.CCL(input)
# output=torch.ops.CCL.CCL(input)
# output=torch.ops.CCL.CCL(input)
# torch.cuda.synchronize()
# end_time = time.time()
# print(end_time-start_time)

# #print(output)



