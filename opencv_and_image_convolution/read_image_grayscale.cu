#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("skull-american-flag.png", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Couldn't load image\n";
        return -1;
    }

    cv::Mat imgFloat;
    image.convertTo(imgFloat, CV_32FC3, 1.0 / 255.0);

    std::cout << "Image size: " << imgFloat.rows << " x " << imgFloat.cols << std::endl;

    for (int i = 200; i < 220; ++i) {
        for (int j = 200; j < 220; ++j) {
            cv::Vec3f pixel = imgFloat.at<cv::Vec3f>(i, j); // B, G, R
            std::cout << "(" << pixel[0] << ", " << pixel[1] << ", " << pixel[2] << ") ";
        }
        std::cout << std::endl;
    }

    // Step 5: Display the image (convert back to 8-bit for display)
    cv::Mat displayImg;
    imgFloat.convertTo(displayImg, CV_8UC3, 255.0); // back to 0-255
    // cv::imshow("Display window", displayImg);
    // cv::waitKey(0);

    return 0;
}
