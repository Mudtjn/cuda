#include <opencv2/opencv.hpp>
#include "constants.cuh" 
#include "cpu_ops.cu"
#include "gpu_ops.cu" 

void displayImage( cv::Mat &result_image) {
    cv::Mat display; 
    result_image.convertTo(display, CV_8UC3, 255.0); 
    cv::imshow("Output", display);
    cv::waitKey(0);
}

int main() {

    cv::Mat image = cv::imread("../../images/mario.png", cv::IMREAD_COLOR); 
    if(image.empty()) {
        std::cerr << "Couldn't load sample \n"; 
        return -1; 
    }

    //////////////////// GAUSSIAN_KERNEL ////////////////
    cv::Mat gaussianKernel = cv::getGaussianKernel(KERNEL_SIZE, GAUSSIAN_STANDARD_DEVIATION, CV_32F); 
    float *gaussian_kernel; 
    auto kernelElements = KERNEL_SIZE * KERNEL_SIZE;
    auto kernelSize = kernelElements * sizeof(float);  
    cudaMallocManaged(&gaussian_kernel, kernelSize);
    cudaMemcpy(gaussian_kernel, gaussianKernel.ptr(), kernelSize, cudaMemcpyHostToHost);  
    /////////////////////////////////////////////////////


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

    /////////////////////////// LIGHTEN IMAGE ///////////////////////////////
    // GPUOPERATIONS::lightenImage<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>> (initial_image, result_image, image_size);
    // cudaDeviceSynchronize(); 
    /////////////////////////////////////////////////////////////////////////

    //////////////////////////// IMAGE_CONVOLUTION CPU //////////////////////
    CPUOPERATIONS::ImageConvolution(initial_image, result_image, gaussian_kernel, imgFloat.rows, imgFloat.cols); 
    /////////////////////////////////////////////////////////////////////////
    std::cout<<"Image convolution complete"<<std::endl; 
    
    cv::Mat final_image(imgFloat.rows, imgFloat.cols, CV_32FC3, result_image); 
    displayImage(final_image); 

    return 0; 
}