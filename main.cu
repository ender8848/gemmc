#include "test/IntervalTest.cuh"
#include "test/gemmGPUCTest.cuh"
#include "test/gemmGPUPyTest.cuh"
#include "test/mmaGPUCTest.cuh"
#include "src/mma.cuh"



int main() {
    /// Interval test
    printf("--------Testing Interval--------\n");
    hasSaferMultiplication<<<1, 1>>>();
    cudaDeviceSynchronize();
    hasSaferAddition<<<1,1>>>();
    cudaDeviceSynchronize();

    /// gemmGPUCTest
    printf("--------Testing gemmGPUCUsingGPUPtr API--------\n");
    GemmCalculatesCorrectly<float>();
    GemmCalculatesCorrectly<double>();
    GemmCalculatesCorrectly<Interval<float>>();
    // do not why, but this does not work:
    GemmCalculatesCorrectly<Interval<double>>();

    /// gemmGPUPyTest
    printf("--------Testing gemmGPUPy API--------\n");
    canCallGemmGPUC<float>();
    canCallGemmGPUC<double>();
    canCallGemmGPUC<Interval<float>>();
    canCallGemmGPUC<Interval<double>>();

    /// mmaGPUCTest
    printf("--------Testing gemmGPUCUsingGPUPtr API--------\n");
    mmaGPUCalculatesCorrectly<float>();
    mmaGPUCalculatesCorrectly<double>();
    mmaGPUCalculatesCorrectly<Interval<float>>();
    // do not why, but this does not work:
    mmaGPUCalculatesCorrectly<Interval<double>>();
    return 0;
}