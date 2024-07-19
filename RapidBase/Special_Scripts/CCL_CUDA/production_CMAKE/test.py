import torch, sys
#sys.path.append("build/lib.linux-x86_64-cpython-39")
torch.ops.load_library("build/libCCL.so")


input = torch.ones([3,10,12],device='cuda',dtype=bool)


input[:,::3,:]=0
input[:,:,::3]=0 
input[0,3,0]=1 

print(input.is_contiguous())

#CCL.sup_dude()
#CCL.dummy(input)
output=torch.ops.CCL.CCL(input)
torch.cuda.synchronize()
print(output)



