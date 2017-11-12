### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -x cu
    )

# Specify include directories
include_directories(
    headers
    )

# Specify library paths
#link_directories(
#    )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    matrix_main
    matrix_main.cu
    matrixOps.cu
    utils.cu
    vectorOps.cu
    )

# For linking ...
# Specify target & libraries to link it with
#target_link_libraries(
#    )
