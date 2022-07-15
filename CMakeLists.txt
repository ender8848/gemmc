set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
cmake_minimum_required(VERSION 3.22)
project(CutlassTest CUDA)

set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CUDA_ARCHITECTURES 61)

add_executable(CutlassTest main.cu Interval.cuh IntervalTest.cuh gemmGPU.cuh gemmGPUCTest.cuh gemmGPUPyTest.cuh)

set_target_properties(CutlassTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

