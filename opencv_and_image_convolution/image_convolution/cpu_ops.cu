#include <iostream>
#include "constants.cuh"

namespace CPUOPERATIONS
{
    float getValAtIndex(float *d_image, int x, int y, int ht, int width) {
        if(x < 0 || y < 0 || x >= ht || y >= width) return 0.0f; 
        auto indx = x * width + y; 
        return d_image[indx]; 
    }

    void ImageConvolution(float *d_image, float *d_result_image, float *kernel, int ht, int width)
    {
        // assuming kernel already flipped
        for(auto i = 0; i<ht; i++) {
            for(auto j =0 ; j<width ; j++) {
                // FIX: forgot to account for 3 channels
                // they are in form of BGRBGR......
                auto row_start = i - (KERNEL_SIZE / 2), row_end = i + (KERNEL_SIZE / 2), kernel_iterator_i = 0;
                auto col_start = j - (KERNEL_SIZE / 2), col_end = j + (KERNEL_SIZE / 2), kernel_iterator_j = 0;
                auto sum = 0.0f; 
                for(auto st = row_start ; st<=row_end; st++, kernel_iterator_i++) {
                    for(auto en = col_start ; en <= col_end ; en++, kernel_iterator_j++) {
                        sum += getValAtIndex(d_image, st, en, ht, width) * kernel[kernel_iterator_i * KERNEL_SIZE + kernel_iterator_j]; 
                    }
                }
                d_result_image[i*width + j] = sum;
                std::cout<<"d_result_image(i,j) = " << sum << std::endl; 
            }
        }
    }

    void lightenImage(float *d_image, float *d_result_image, int N)
    {
        for (auto idx = 0; idx < N; idx++)
        {
            d_result_image[idx] = d_image[idx] * LIGHTNING_FACTOR;
        }
    }

}