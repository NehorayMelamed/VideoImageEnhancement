import torch, sys
#sys.path.append("build/lib.linux-x86_64-cpython-39")
torch.ops.load_library("build/libCCL.so")


input = torch.ones([500,500,8000],device='cuda',dtype=torch.uint8)


input[:,::3,:]=0
input[:,:,::3]=0 
input[0,3,0]=1 

#print(input)
#CCL.sup_dude()
#CCL.dummy(input)
import time
start_time = time.time()
output=torch.ops.CCL.CCL(input)
output=torch.ops.CCL.CCL(input)
output=torch.ops.CCL.CCL(input)
output=torch.ops.CCL.CCL(input)
output=torch.ops.CCL.CCL(input)
torch.cuda.synchronize()
end_time = time.time()
print(end_time-start_time)

#print(output)



