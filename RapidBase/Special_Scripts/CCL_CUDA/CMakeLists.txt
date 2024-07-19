cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(array_fire_regions LANGUAGES CXX CUDA)

add_executable(array_fire_regions "regions_test.cu")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++17 -m64")
set(CMAKE_CXX_STANDARD 17)
#option(USE_CUDA "Use CUDA" ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -std=c++17 --use_fast_math --expt-relaxed-constexpr -lineinfo -m64")
enable_language("CUDA")

#cmake_policy(SET CMP0057 NEW)
#set(CMAKE_C_COMPILER gcc)
#set(CMAKE_CXX_COMPILER g++)

#set(Python_ROOT_DIR "/usr/lib/python3.9/")
#find_package(Python COMPONENTS Interpreter Development)


#set(pybind11_DIR "/home/yuri/.local/lib/python3.9/site-packages/pybind11/share/cmake/pybind11/")
#FIND_PACKAGE(pybind11 CONFIG REQUIRED)

#execute_process(COMMAND python3 -c 'import torch;print(torch.utils.cmake_prefix_path)' OUTPUT_VARIABLE TORCH_DIR)
#set(ArrayFire_DIR "/home/mafat/arrayfire/ArrayFire-3.8.2-Linux/share/ArrayFire/cmake")

#set(Torch_DIR "/home/yuri/.local/lib/python3.9/site-packages/torch/share/cmake/Torch/")
set(Torch_DIR "/home/mafat/.local/lib/python3.8/site-packages/torch/share/cmake/Torch/")

find_package(Torch REQUIRED)

#enable_language("CUDA")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set_property(TARGET array_fire_regions PROPERTY CXX_STANDARD 17)
target_link_libraries(array_fire_regions PRIVATE "${TORCH_LIBRARIES}")

#pybind11_add_module(TST TST.cpp)

#target_link_libraries(example-app "${TORCH_LIBRARIES}")



