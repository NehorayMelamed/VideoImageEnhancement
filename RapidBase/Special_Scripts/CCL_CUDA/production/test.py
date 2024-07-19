import torch, sys
sys.path.append("build/lib.linux-x86_64-cpython-39")
import CCL

#input = torch.ones([500,500,8000],device='cuda',dtype=bool)

#inputs_raw = torch.ones([2,5,5],device='cuda',dtype=bool)

p=0.3
#inputs_raw = torch.rand([1,10,10],device='cuda',dtype=float)
inputs_raw = torch.rand([500,500,8000],device='cuda',dtype=float)
inputs = (inputs_raw<p).to(bool)

#inputs[0,::3,:]=0
#inputs[0,:,::3]=0
#inputs[1,::2,:]=0
#inputs[1,:,::2]=0 
# # input[0,3,0]=1 

# print(input.is_contiguous())

# output1=CCL.CCL(input)
# output2=CCL.CC_Blob_Centers(output1)

# print(input)
# print(output1)
# print(output2)

#CCL.sup_dude()
#CCL.dummy(input)


import time
start_time = time.time()

output1=CCL.CCL(inputs)
output2=CCL.CC_Blob_Centers(output1)
torch.cuda.synchronize()
mid_time = time.time()
output3=CCL.CC_Blob_Centers_Formatted(output1,0.9)

torch.cuda.synchronize()
end_time = time.time()
print(mid_time-start_time)
print(end_time-mid_time)
print(end_time-start_time)

#print(inputs)
#print(output2)
#print(output3)
# #print(output)



