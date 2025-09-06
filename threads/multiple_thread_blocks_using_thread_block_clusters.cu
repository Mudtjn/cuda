#include<bits/stdc++.h>
using namespace std; 

#define N 256

// kernel defn
// Compile time cluster size 2 in x dimension, 1 in y and z dimension
// not supported in RTX 40 series 
__global__ void __cluster_dims__(2,1,1) MatAdd( float *A, float *B, float *C ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int j = blockIdx.y * blockDim.y + threadIdx.y; // column

    if(i*N + j < N*N) {
        C[i*N + j] = A[i*N + j] + B[i*N + j];
    }
} 

__global__ void  MatAddWithRuntimeConfig( float *A, float *B, float *C ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int j = blockIdx.y * blockDim.y + threadIdx.y; // column

    if(i*N + j < N*N) {
        C[i*N + j] = A[i*N + j] + B[i*N + j];
    }
}

int main() {
    float*A, *B, *C;
    
    cudaMallocManaged(&A, N*N*sizeof(float));
    cudaMallocManaged(&B, N*N*sizeof(float));
    cudaMallocManaged(&C, N*N*sizeof(float));

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            A[i*N+j] = 1.0f; 
            B[i*N+j] = 2.0f;
        }
    }

    dim3 threadsPerBlock(16, 16);
    // N/threadsPerBlock.x in x direction and N/threadsPerBlock.y in y direction
    dim3 numBlocks( N/threadsPerBlock.x, N/threadsPerBlock.y); 
 
 
    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);

    cudaDeviceSynchronize(); 

    // kernel invocation with runtime cluster configuration
    {
        cudaLaunchConfig_t config = {0}; 
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using number of blocks.
        // The grid dimension must be a multiple of cluster size.
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1]; 
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, MatAddWithRuntimeConfig, A, B, C);
    }

    float maxError = 0.0f; 
    for(int i = 0 ; i<N ; i++) {
        for(int j = 0 ; j<N ; j++) {
            float expected = 3.0f; 
            maxError = fmax(maxError, fabs(C[i*N+j]-expected));            
        }
    }

    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C); 
    return 0;
}