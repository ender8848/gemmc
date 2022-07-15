//
// Created by hao on 07/07/22.
//

#ifndef CUTLASSTEST_GEMMGPU_CUH
#define CUTLASSTEST_GEMMGPU_CUH
#include "../cutlass/numeric_types.h"
#include "../cutlass/gemm/device/gemm.h"
#include "../cutlass/util/host_tensor.h"
#include "Interval.cuh"


enum datatype {
    FLOAT = 0,
    DOUBLE = 1,
    INTV_FLOAT = 2,
    INTV_DOUBLE = 3
};

/*
 * C interface to do gemm on GPU
 * Performs dest_dev = A_dev @ B_dev + bias_dev if bias_dev is not NULL
 * Performs dest_dev = A_dev @ B_dev + dest_dev if bias_dev is NULL
 * @param A_dev: pointer to the device memory of matrix A
 * @param B_dev: pointer to the device memory of matrix B
 * @param dest_dev: pointer to the device memory of matrix dest, which is the output matrix
 * @param M: number of rows of A and dest
 * @param N: number of columns of B and dest
 * @param K: number of columns of A and rows of B
 * @param bias_dev: pointer to the device memory of bias matrix, default to NULL
 */
template<typename T>
void gemmGPUCUsingGPUPtr(T* A_dev, T* B_dev, T* dest_dev, int M, int N, int K, T* bias_dev = nullptr) {
    using Gemm = cutlass::gemm::device::Gemm<
            T,                                    // ElementA, namely type of A
            cutlass::layout::ColumnMajor,         // LayoutA, column major means colum of A is contiguous in memory
            T,                                    // ElementB
            cutlass::layout::ColumnMajor,         // LayoutB
            T,                                    // ElementOutput
            cutlass::layout::ColumnMajor,         // LayoutOutput
            T,                                    // ElementAccumulator
            cutlass::arch::OpClassSimt,           // tag indicating Tensor Cores, architecture-dependent
            cutlass::arch::Sm61                   // tag indicating target GPU opcode class, architecture-dependent (61 for GTX 1060)
    >;
    Gemm gemm;
    cutlass::Status status;

    T alpha = T(1.);    // Define alpha and beta, this controls dest = alpha * A @ B + beta * bias
    T beta = T(1.);     // use 1 here to get dest = A @ B + bias
    int lda = M;        // leading dimension of A, namely the number of rows of A
    int ldb = K;        // leading dimension of B, namely the number of rows of B
    int ld_dest = M;    // leading dimension of dest, namely the number of rows of dest
    int ld_bias = M;    // leading dimension of bias, namely the number of rows of bias

    status = gemm({
        {M,        N, K},
        {A_dev,    lda},            // TensorRef to A device tensor
        {B_dev,    ldb},            // TensorRef to B device tensor
        {bias_dev, ld_dest},        // TensorRef to C device tensor
        {dest_dev, ld_bias},        // TensorRef to D device tensor - may be the same as C (depending on passed value)
        {alpha,    beta}            // epilogue operation arguments
    });

    if (status != cutlass::Status::kSuccess) {
        printf("GEMM failed\n");
        printf("status: %d\n", status);
        printf("\n");
    }
}

/*
 * C interface to do gemm on GPU, but receives host tensor as input
 * Performs dest_host = A_host @ B_host + bias_host if bias_host is not NULL
 * Performs dest_host = A_host @ B_host + dest_host if bias_host is NULL
 * This function is just an adapter which create device tensor from host tensor and call gemmGPUCUsingGPUPtr
 * @param A_host: pointer to the host memory of matrix A
 * @param B_host: pointer to the host memory of matrix B
 * @param dest_host: pointer to the host memory of matrix dest, which is the output matrix
 * @param M: number of rows of A and dest
 * @param N: number of columns of B and dest
 * @param K: number of columns of A and rows of B
 * @param bias_host: pointer to the host memory of bias matrix, default to NULL
 */
template<typename T>
void gemmGPUCUsingCPUPtr(T* A_host, T* B_host, T* dest_host, int M, int N, int K, T* bias_host = nullptr) {
    // create device tensor in GPU with same size as host tensor
    T * A_dev = nullptr;
    T * B_dev = nullptr;
    T * dest_dev = nullptr;
    cudaMalloc((void**)&A_dev, sizeof(T)*M*K);
    cudaMalloc((void**)&B_dev, sizeof(T)*K*N);
    cudaMalloc((void**)&dest_dev, sizeof(T)*M*N);
    // copy host tensor to device tensor
    cudaMemcpy(A_dev, A_host, sizeof(T)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, sizeof(T)*K*N, cudaMemcpyHostToDevice);

    // delegate to gemmGPUCUsingGPUPtr
    gemmGPUCUsingGPUPtr(A_dev, B_dev, dest_dev, M, N, K, bias_host);

    // copy dest tensor back to host tensor
    cudaMemcpy(dest_host, dest_dev, sizeof(T)*M*N, cudaMemcpyDeviceToHost);
}

/// need to change this place, because new function gemmGPUCUsingCPUPtr has been added
/// also add more comments to explain this function
extern "C" {
// perform float gemm D = AB+C
void gemmGPUPy(void* A_dev, void* B_dev, void* dest_dev, int M, int N, int K, int datatype, void* bias_dev = nullptr) {
    switch (datatype) {
        case datatype::FLOAT:
            gemmGPUCUsingGPUPtr<float>(static_cast<float *>(A_dev),
                                       static_cast<float *>(B_dev),
                                       static_cast<float *>(dest_dev),
                                       M, N, K,
                                       static_cast<float *>(bias_dev));
            break;
        case datatype::DOUBLE:
            gemmGPUCUsingGPUPtr<double>(static_cast<double *>(A_dev),
                                        static_cast<double *>(B_dev),
                                        static_cast<double *>(dest_dev),
                                        M, N, K,
                                        static_cast<double *>(bias_dev));
            break;
        case datatype::INTV_FLOAT:
            gemmGPUCUsingGPUPtr<Interval<float>>(static_cast<Interval<float> *>(A_dev),
                                                 static_cast<Interval<float> *>(B_dev),
                                                 static_cast<Interval<float> *>(dest_dev),
                                                 M, N, K,
                                                 static_cast<Interval<float> *>(bias_dev));
            break;
        case datatype::INTV_DOUBLE:
            gemmGPUCUsingGPUPtr<Interval<double>>(static_cast<Interval<double> *>(A_dev),
                                                  static_cast<Interval<double> *>(B_dev),
                                                  static_cast<Interval<double> *>(dest_dev),
                                                  M, N, K,
                                                  static_cast<Interval<double> *>(bias_dev));
            break;
        default:
            break;
    }
}
};



#endif //CUTLASSTEST_GEMMGPU_CUH
