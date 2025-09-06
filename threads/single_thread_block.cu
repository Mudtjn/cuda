#include<iostream>
#include<math.h>
 
// gridDim.x = number of blocks
__global__
void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int stride = blockDim.x * gridDim.x; // stride = 256
    for (int i = index; i < n; i+=stride)
        y[i] = x[i] + y[i];
}

int main(void) {
    int N = 1<<20; 
    float *x, *y; 

    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for(int i =0 ; i<N ; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blocksize = 256; 
    int numBlocks = (N+ blocksize - 1)/ blocksize;

    cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
    cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);

    add<<<numBlocks,blocksize>>>(N, x, y);

    cudaDeviceSynchronize(); 
    float maxError = 0.0f; 
    for(int i = 0 ; i<N ; i++) {
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }

    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
    return 0;
}

