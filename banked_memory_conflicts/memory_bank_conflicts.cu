#include<bits/stdc++.h>
using namespace std; 

#define TILE_WIDTH 32

__global__ 
void non_conflicting_bank_access(float*in, float* out) {
    __shared__ float data[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int index = ty * TILE_WIDTH + tx;

    data[ty][tx] = in[index];
    __syncthreads();

    out[index] = data[ty][tx];
}

__global__
void conflicting_bank_access(float*in, float* out) {
    
    __shared__ float data[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int index = ty * TILE_WIDTH + tx;
    int conflict_index = (tx * 2) % TILE_WIDTH;

    data[ty][conflict_index] = in[index];
    __syncthreads();

    out[index] = data[ty][conflict_index];

}

int main() {

    float *in, *out, *d_in, *d_out;
    int size = TILE_WIDTH * TILE_WIDTH * sizeof(float);

    in = (float*)malloc(size);
    out = (float*)malloc(size);

       
}