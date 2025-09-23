#include <bits/stdc++.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
using namespace std; 

#define M 4096  // Number of rows in A and C
#define K 8192   // Number of columns in A and rows in B
#define N 4096  // Number of columns in B and C
// keeping tile = warp size for better coalescing
#define BLOCK_SIZE 32

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)


void matmul_cpu(float* A, float* B, float *C, int m, int k, int n) {
    for(int i =0 ; i<m ; i++) {
        for(int j = 0; j<n ; j++) {
            float sum = 0.0f; 
            for(int p = 0 ; p<k ; p++) {
                sum += A[i*k + p] * B[p*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

__global__ 
void matmul_gpu(float* A, float* B, float * C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n) {
        float sum = 0.0f ;
        for(int i = 0 ; i<k ; i++) {
            sum += A[row * k + i] * B[i * n + col]; 
        }
        C[row * n + col] = sum;
    }
}

__global__
void matmul_gpu_tiling(const float* A, const float* B, float* C, int m, int k, int n) {
    __device__ __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __device__ __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Conventional mapping: threadIdx.x -> column within tile, threadIdx.y -> row within tile
    int tileRow = blockIdx.y;      // which tile-row of C
    int tileCol = blockIdx.x;      // which tile-col of C
    int localRow = threadIdx.y;    // 0..BLOCK_SIZE-1
    int localCol = threadIdx.x;    // 0..BLOCK_SIZE-1
 
    unsigned int row = tileRow * BLOCK_SIZE + localRow; // global row index in C
    unsigned int col = tileCol * BLOCK_SIZE + localCol; // global col index in C

    float sum = 0.0f; // this is each thread memory 

    int numTiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Load tile of A: rows = row, cols = t*BLOCK_SIZE + localCol
        int aCol = t * BLOCK_SIZE + localCol;
        if (row < m && aCol < k)
            As[localRow][localCol] = A[row * k + aCol];
        else
            As[localRow][localCol] = 0.0f;

        // Load tile of B: rows = t*BLOCK_SIZE + localRow, cols = col
        int bRow = t * BLOCK_SIZE + localRow;
        if (bRow < k && col < n)
            Bs[localRow][localCol] = B[bRow * n + col];
        else
            Bs[localRow][localCol] = 0.0f;

        __syncthreads();

        // Multiply accumulate for this tile
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            sum = __fadd_rd(sum, __fmul_rd(As[localRow][e], Bs[e][localCol])); 
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum; // write once after accumulating all tiles
    }
}


// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {

    float *h_A, *h_B, *h_C_cpu;
    float *d_A, *d_B, *d_C, *d_C_tiling;
    
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);

    srand(time(NULL)); 
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    CHECK_CUDA(cudaMalloc(&d_C_tiling, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); // 1024 threads per block
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE); // grid => 8 * 8
    // 64 thread blocks
    // 256 * 256 / 8*8 = 1024 operations per block
    // 1024 / 1024 = 1 operation per thread

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        // matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_gpu<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        matmul_gpu_tiling<<<gridSize, blockSize>>>(d_A, d_B, d_C_tiling, M, K, N); 
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Benchmark CPU implementation
    // printf("Benchmarking CPU implementation...\n");
    // double cpu_total_time = 0.0;
    // for (int i = 0; i < 20; i++) {
    //     double start_time = get_time();
    //     matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    //     double end_time = get_time();
    //     cpu_total_time += end_time - start_time;
    // }
    // double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    /**
     * @brief This is cuda time recording 
     * measuring with get_time also includes 
     * kernel launch overhead
     * driver scheduling
     * OS jitter
     * PCIe synchronization costs
     * 
     * This will cause computed time for gpu to be all over the place as other are variable
     * cudaEventCreate help calculate raw cost of computation
     */
    cudaEvent_t start, stop; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); 
    for(int i = 0 ; i<20 ; i++) {    
        matmul_gpu<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    float matmul_naive = 0.0f; 
    cudaEventElapsedTime(&matmul_naive, start, stop);
    double gpu_avg_time = matmul_naive/ 20.0;

    // Benchmarking GPU tiling
    printf("Benchmarking GPU TILING implementation...\n");
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); 
    for(int i = 0 ; i<20 ; i++) {
        matmul_gpu_tiling<<<gridSize, blockSize>>>(d_A, d_B, d_C_tiling, M, K, N);
    }
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    float matmul_tiling = 0.0f; 
    cudaEventElapsedTime(&matmul_tiling, start, stop); 
    double gpu_avg_time_tiling = matmul_tiling / 20.0;


    cudaEventDestroy(start); 
    cudaEventDestroy(stop); 

    // Print results
    // printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e3f));
    printf("GPU average time tiling: %f microseconds\n", (gpu_avg_time_tiling * 1e3f));
    // printf("Speedup: %fx\n", cpu_avg_time * 1e3f / gpu_avg_time);
    printf("Speed gpu vs gpu_tiling: %fx\n", gpu_avg_time / gpu_avg_time_tiling); 

    free(h_A); 
    free(h_B);
    free(h_C_cpu);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));  
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_tiling));

    return 0; 
}

// Performing warm-up runs...
// Benchmarking GPU implementation...
// Benchmarking GPU TILING implementation...
// GPU average time: 408633.544922 microseconds
// GPU average time tiling: 281615.869141 microseconds
// Speed gpu vs gpu_tiling: 1.451032x