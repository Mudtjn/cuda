#include <opencv2/opencv.hpp>
#define THREADS_PER_BLOCK 1024
#define KERNEL_SIZE 31

__global__
void lightenImage(float *d_image, float *d_result_image, int N) {
    // indx being pointed to
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) ; 
    if(idx < N) {
        d_result_image[idx] = d_image[idx] * 3.0;
    }
}

void displayImage( cv::Mat &result_image) {
    cv::Mat display; 
    result_image.convertTo(display, CV_8UC3, 255.0); 
    cv::imshow("Output", display);
    cv::waitKey(0);
}

int main() {

    cv::Mat image = cv::imread("mario.png", cv::IMREAD_COLOR); 
    if(image.empty()) {
        std::cerr << "Couldn't load sample \n"; 
        return -1; 
    }

    // this is a 1-D array
    cv::Mat imgFloat; 
    image.convertTo(imgFloat, CV_32FC3, 1.0/255.0); 

    std::cout << "Image size: " << imgFloat.rows << " x " << imgFloat.cols << std::endl;

    float *initial_image, *result_image;
    size_t num_elements = imgFloat.total() * imgFloat.channels(); 
    size_t image_size = num_elements * sizeof(float); 
    


    cudaMallocManaged(&initial_image, image_size); 
    cudaMallocManaged(&result_image, image_size); 
    cudaMemcpy(initial_image, imgFloat.ptr(), image_size, cudaMemcpyHostToDevice); 

    auto BLOCKS_PER_GRID = (num_elements + THREADS_PER_BLOCK - 1) /   THREADS_PER_BLOCK; 

    lightenImage<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>> (initial_image, result_image, image_size);
    cudaDeviceSynchronize(); 
    
    cv::Mat final_image(imgFloat.rows, imgFloat.cols, CV_32FC3, result_image); 
    displayImage(final_image); 

    return 0; 
}