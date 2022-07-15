//
// Created by hao on 07/07/22.
//

#ifndef CUTLASSTEST_GEMMGPUCTEST_CUH
#define CUTLASSTEST_GEMMGPUCTEST_CUH

#include "../src/gemmGPU.cuh"
#include "../cutlass/numeric_types.h"
#include "../cutlass/gemm/device/gemm.h"
#include "../cutlass/util/host_tensor.h"
#include "../src/Interval.cuh"

template<typename T>
__global__ void printMatrix(T* M_dev, int row, int col);

template<>
__global__ void printMatrix(float * M_dev, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("%.2f ", M_dev[i*col + j]);
        }
        printf("\n");
    }
}


template<>
__global__ void printMatrix(double * M_dev, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("%.2f ", M_dev[i*col + j]);
        }
        printf("\n");
    }
}

// implementing template within template
template<>
__global__ void printMatrix(Interval<float>* M_dev, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("(%.2f, %.2f) ", M_dev[i*col + j].lower, M_dev[i*col + j].upper);
        }
        printf("\n");
    }
}

template<>
__global__ void printMatrix(Interval<double>* M_dev, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("(%.2f, %.2f) ", M_dev[i*col + j].lower, M_dev[i*col + j].upper);
        }
        printf("\n");
    }
}

template<typename T>
__global__ void initMatrix(T* M_dev, int row, int col, T val) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            M_dev[i*col + j] = val;
        }
    }
}

template<typename T>
void GemmCalculatesCorrectly() {
    printf("Testing type -- %s\n", typeid(T).name());

    // Define the problem size and params
    int M = 2;
    int N = 4;
    int K = 4;

    // Allocate device memory
    cutlass::HostTensor<T, cutlass::layout::ColumnMajor> A({M, K});
    cutlass::HostTensor<T, cutlass::layout::ColumnMajor> B({K, N});
    cutlass::HostTensor<T, cutlass::layout::ColumnMajor> C({M, N});

    T *A_dev = A.device_data();
    T *B_dev = B.device_data();
    T *C_dev = C.device_data(); // use C_dev as D_dev

    // init will make dest zero, which is same as pyTorch init
    float a = 1.f;
    float b = 2.f;

    initMatrix<T><<<1, 1>>>(A_dev, M, K, {a});
    cudaDeviceSynchronize();
    printMatrix<T><<<1, 1>>>(A_dev, M, K);
    cudaDeviceSynchronize();
    printf("times...\n");
    initMatrix<T><<<1, 1>>>(B_dev, K, N, b);
    cudaDeviceSynchronize();
    printMatrix<T><<<1, 1>>>(B_dev, K, N);
    cudaDeviceSynchronize();

    // test the API
    gemmGPUCUsingGPUPtr<T>(A_dev, B_dev, C_dev, M, N, K, C_dev);

    printf("equals to:\n");
    printMatrix<T><<<1, 1>>>(C_dev, M, N);
    cudaDeviceSynchronize();
    printf("\n");
}


#endif //CUTLASSTEST_GEMMGPUCTEST_CUH
