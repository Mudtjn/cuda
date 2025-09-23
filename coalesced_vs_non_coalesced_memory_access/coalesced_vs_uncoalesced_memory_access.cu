#include<bits/stdc++.h>

#define TRIAL_RUNS 1000
#define THREADS_PER_BLOCK 1024
#define STRIDE 2

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)


// coalesced memory access kernel
__global__ void coalesced_kernel(float *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // this ensures contiguous memory access
    if (idx < n) { 
        a[idx] += 1.0f;
    }
}

// non coalesced memory access kernel
__global__ void non_coalesced_kernel(float *a, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // the multiplication ensures that threads access memory locations that are not contiguous
        int index = (idx * stride) % n; // wrap around if index exceeds n
        a[index] = a[index] + 1.0f;
    }
}



int main() {
    size_t N = 1<<20; 
    float *h_a, *h_b; // host memory
    float *d_a, *d_b; // device memory

    dim3 gridSize((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    size_t arr_size = N * sizeof(float);
    h_a = (float*)malloc(arr_size);
    h_b = (float*)malloc(arr_size);
    CHECK_CUDA(cudaMalloc(&d_a, arr_size));
    CHECK_CUDA(cudaMalloc(&d_b, arr_size));

    for (auto i = 0; i < N; i++) {
        h_a[i] = 1.0f, h_b[i] = 1.0f;
    }


    std::cout << "Benchmarking Coalesced Memory Access" << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto total_coalesced_time = 0.0f;
    for(int i = 0; i<TRIAL_RUNS; i++) {
        CHECK_CUDA(cudaMemcpy(d_a, h_a, arr_size, cudaMemcpyHostToDevice)); // reallocating memory to avoid caching effects

        cudaEventRecord(start);

        coalesced_kernel<<<gridSize, THREADS_PER_BLOCK>>>(d_a, N);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    
        cudaEventRecord(stop); 
        cudaEventSynchronize(stop); 
        auto coalesced_naive = 0.0f; 
        cudaEventElapsedTime(&coalesced_naive, start, stop);
        total_coalesced_time += coalesced_naive;
    }
    auto coalesced_avg_time = total_coalesced_time/ (TRIAL_RUNS * 1.0);
    std::cout << "Average time for coalesced memory access kernel: " << coalesced_avg_time << " ms" << std::endl;


    std::cout << "Benchmarking Non Coalesced Memory Access" << std::endl;
    
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    auto total_non_coalesced_time = 0.0f;
    for(int i = 0; i<TRIAL_RUNS ; i++) {
        CHECK_CUDA(cudaMemcpy(d_b, h_b, arr_size, cudaMemcpyHostToDevice));

        cudaEventRecord(start1);

        non_coalesced_kernel<<<gridSize, THREADS_PER_BLOCK>>>(d_b, N, 100);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEventRecord(stop1); 
        cudaEventSynchronize(stop1); 
        auto non_coalesced_naive = 0.0f; 
        cudaEventElapsedTime(&non_coalesced_naive, start1, stop1);
        total_non_coalesced_time += non_coalesced_naive;
    }
    auto non_coalesced_avg_time = total_non_coalesced_time/ (TRIAL_RUNS * 1.0);
    std::cout << "Average time for non coalesced memory access kernel: " << non_coalesced_avg_time << " ms" << std::endl;
 
    cudaFree(d_a);
    free(h_a);
    cudaFree(d_b);
    free(h_b);

    return 0; 
}