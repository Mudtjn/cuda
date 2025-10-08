#include "constants.cuh"

namespace GPUOPERATIONS
{
    __global__ void lightenImage(float *d_image, float *d_result_image, int N)
    {
        // indx being pointed to
        int idx = (blockDim.x * blockIdx.x + threadIdx.x);
        if (idx < N)
        {
            d_result_image[idx] = d_image[idx] * LIGHTNING_FACTOR;
        }
    }
}