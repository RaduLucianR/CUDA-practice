cmake_minimum_required(VERSION 3.18)  # Ensure a recent CMake version for better CUDA support
project(MyCudaProject LANGUAGES CXX CUDA)

# Optionally, specify the CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Create an executable from the main.cu file
add_executable(hello src/hello.cu)

add_executable(matrix_mul src/matrix_mul.cu)

add_executable(testc src/check.cpp)

add_executable(generate_inputs src/generate_inputs.cpp)

add_executable(mmul src/mmul.cu)

add_executable(mmul_simple src/mmul_simple.cu)