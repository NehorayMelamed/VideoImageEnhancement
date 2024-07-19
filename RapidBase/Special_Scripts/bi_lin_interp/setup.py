import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-11.8"


from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME)

setup(name='ecc_bilinear_interpolation',
      ext_modules=[
          CUDAExtension(
              'ecc_bilinear_interpolation',
              ['ecc_bind.cu'],
              extra_compile_args={'cxx': ['-O3', '-m64'],
                                  'nvcc': ['-O3',
                                           '-std=c++17',
                                           '--use_fast_math',
                                           '--expt-relaxed-constexpr',
                                           '--expt-extended-lambda',
                                           '-lineinfo',
                                           '-m64']}
          )],
      cmdclass={'build_ext': BuildExtension},
      )


# setup(name='CCL',
#       ext_modules=[CUDAExtension('CCL', ['regions_bind.cu'])],
#       cmdclass={'build_ext': BuildExtension},
# )

# extra_compile_args = {'cxx': ['-O3', '-std=c++17']
#                         ,'nvcc':['-O3','-std=c++17',
#                         '--use_fast_math',
#                         '--expt-relaxed-constexpr',
#                         '-lineinfo',
#                         '-m64']}