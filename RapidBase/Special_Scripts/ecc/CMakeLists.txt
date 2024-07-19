cmake_minimum_required (VERSION 3.9)

project(ecc_reduce LANGUAGES CXX CUDA)

#execute_process(COMMAND python3 -c 'import torch;print(torch.utils.cmake_prefix_path)' OUTPUT_VARIABLE FOO CMAKE_PREFIX_PATH)

set(CMAKE_PREFIX_PATH "/home/yuri/lib64/python3.9/site-packages/torch/share/cmake")
#set(OpenCV_DIR "/home/yuri/software/OpenCV_4/lib64/cmake/opencv4")
#set(CMAKE_PREFIX_PATH /home/mafat/.local/lib/python3.8/site-packages/torch/share/cmake)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)
#find_package(OpenCV REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

add_library(ecc_reduce SHARED
  ecc_bind.cu
)
target_compile_features(ecc_reduce PRIVATE cxx_std_17)
target_link_libraries(ecc_reduce PRIVATE ${TORCH_LIBRARIES} Python::Python)
target_include_directories(ecc_reduce PRIVATE ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_definitions(-DGPU_OPENCV_ENABLE)
# Use if the default GCC version gives issues
target_compile_options(ecc_reduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ccbin g++ -O3 -m64 -std=c++17 --use_fast_math --expt-relaxed-constexpr ${TORCH_CXX_FLAGS}>)


# Use a variant of this if you're on an earlier cmake than 3.18
# target_compile_options(pytorch_cmake_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)
