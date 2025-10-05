#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 32
#define STRIDE_SIZE 100
#define NUM_ITERATIONS 100000
#define DATA_SIZE (TILE_WIDTH * TILE_WIDTH)

// Kernel that intentionally causes bank conflicts
__global__ void bankConflictKernel(float *input, float *output) {
    __shared__ float sharedData[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int index = ty * TILE_WIDTH + tx;

    // Access pattern causing bank conflicts
    int conflictIndex = (tx * STRIDE_SIZE) % TILE_WIDTH;  // Artificial non-optimal pattern
    sharedData[ty][conflictIndex] = input[index];

    __syncthreads();

    output[index] = sharedData[ty][conflictIndex];
}

// Kernel that avoids bank conflicts by using a contiguous access pattern
__global__ void optimizedKernel(float *input, float *output) {
    __shared__ float sharedData[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int index = ty * TILE_WIDTH + tx;

    // Coalesced access pattern
    sharedData[ty][tx] = input[index];

    __syncthreads();

    output[index] = sharedData[ty][tx];
}

int main() {
    size_t size = DATA_SIZE * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < DATA_SIZE; i++) {
        h_input[i] = (float)(i);
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Measure bankConflictKernel execution time
    auto total_time = 0.0f;
    for(auto i = 0 ; i<NUM_ITERATIONS ; i++) {
        cudaEventRecord(start);
        bankConflictKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float timeConflict;
        cudaEventElapsedTime(&timeConflict, start, stop);
        total_time += timeConflict;
    }
    double timeConflict = total_time / NUM_ITERATIONS;

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    auto timeOptimized = 0.0f;
    // Measure optimizedKernel execution time
    for(auto i = 0; i<NUM_ITERATIONS ; i++) {
        cudaEventRecord(start1);
        optimizedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);
        float tempTime;
        cudaEventElapsedTime(&tempTime, start1, stop1);
        timeOptimized += tempTime;
    }
    timeOptimized = timeOptimized / NUM_ITERATIONS;

    printf("Bank Conflict Kernel Time:    %f ms\n", timeConflict);
    printf("Optimized Kernel Time:        %f ms\n", timeOptimized);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}