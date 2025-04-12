cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(libtorch_test)
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")

find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED PATHS "/home/handsome/lib/libtorch2.6.0/libtorch")
add_executable(main main.cpp)
target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(main PUBLIC ${TORCH_LIBRARIES})