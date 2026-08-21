cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 23)

set(PROJECT_NAME OpenCL_Learning)

project(${PROJECT_NAME} LANGUAGES CXX)

set(OPENCL_SDK_BUILD_SAMPLES OFF CACHE BOOL "")
set(OPENCL_SDK_BUILD_TESTS OFF CACHE BOOL "")

add_subdirectory(dependencies/OpenCL-SDK)
add_subdirectory(dependencies/OpenCL-CPP)

add_subdirectory(kernels)

add_executable(${PROJECT_NAME} main.cpp)

add_dependencies(${PROJECT_NAME} Kernels OpenCL-CPP)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Kernels
    OpenCL-CPP
)