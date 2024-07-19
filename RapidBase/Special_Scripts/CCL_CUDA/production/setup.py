from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='CCL',
      ext_modules=[
          CUDAExtension(
              'CCL',
              ['regions_bind.cu'],
              extra_compile_args={'cxx': ['-O3', '-m64'],
                                  'nvcc': ['-O3',
                                           '-std=c++17',
                                           '--use_fast_math',
                                           '--expt-relaxed-constexpr',
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
