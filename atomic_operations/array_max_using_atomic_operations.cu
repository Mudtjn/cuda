#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__device__ float atomicMinFloat(float *addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) : __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
    return old;
}

// this function only callable by device
__device__ float atomicEvalMax(float *address, float value)
{
    float old; 
    old = (value >= 0) ? __int_as_float(atomicMax((int*) address, __float_as_int(value))) : 
                __int_as_float(atomicMin((int*) address, __float_as_int(value))); 

    // atomicCAS to replace value at address
    int *address_as_i = (int*) address; 
    int old_val = *address_as_i, assumed; 

    // do while is needed since CAS fails when a thread is concurrently modifying it
    do {
        assumed = old_val; 
        old_val = atomicCAS(address_as_i, assumed, __float_as_int(old));     
    } while(assumed != old_val); 

    return __int_as_float(old_val); 
}


__global__ void atomicMaxKernel(const float *input, float *result, int N)
{
    int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if (indx < N)
    {
        atomicEvalMax(result, input[indx]); 
    }
}

int main()
{

    constexpr auto N = 1 << 20;
    constexpr size_t size = N * sizeof(float);

    // host memory
    auto *h_input = (float *)malloc(size);
    auto h_result = -FLT_MAX;

    for (auto i = 0; i < N; i++)
        h_input[i] = i < N / 2 ? 1.1 : 2.2;

    // device memory
    float *d_input, *d_result;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_result, sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    constexpr auto blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    atomicMaxKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_input, d_result, N); 
    cudaDeviceSynchronize(); 

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input); 
    cudaFree(d_result);
    free(h_input); 

    std::cout<<"Max of array is: "<<h_result<<std::endl; 

    return 0;
}
