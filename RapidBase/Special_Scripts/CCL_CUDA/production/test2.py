import torch, sys
import numpy as np
sys.path.append("build/lib.linux-x86_64-cpython-39")
import CCL


#inputs_raw = torch.rand([1,10,10],device='cuda',dtype=float)
print("loding file")
inputs_raw = np.fromfile("outliers.bin", dtype=bool).reshape([500,500,8000])
inputs = torch.from_numpy(inputs_raw).to("cuda")
print("done loding file")

blob_lables =CCL.CCL(inputs)

ind = 150
blobs=CCL.CC_Blob_Centers_Formatted(blob_lables,0.9)

ind2 = 1 
unique_lables= torch.unique(blob_lables[ind,:,:])
print(unique_lables)
# print(unique_lables.shape)
x,y= torch.where(blobs == unique_lables[ind2])

print(x.shape, x.sum(),y.sum())

cur_blob_rows=blobs[:,0]==ind
cur_blob=blobs[cur_blob_rows,:]
print(cur_blob[0,:])
# print(cur_blobs_rows.sum())

# import time
# start_time = time.time()

# output1=CCL.CCL(inputs)
# output2=CCL.CC_Blob_Centers(output1)
# torch.cuda.synchronize()
# mid_time = time.time()
# output3=CCL.CC_Blob_Centers_Formatted(output1,0.9)

# torch.cuda.synchronize()
# end_time = time.time()
# print(mid_time-start_time)
# print(end_time-mid_time)
# print(end_time-start_time)

#print(inputs)
#print(output2)
#print(output3)
# #print(output)



