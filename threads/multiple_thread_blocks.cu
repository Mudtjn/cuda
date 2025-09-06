#include<bits/stdc++.h>
using namespace std; 

#define N 256

__global__ 
void MatAdd( float *A, float *B, float *C ) {
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
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);

    cudaDeviceSynchronize(); 
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