#include <opencv2/opencv.hpp>

int main() {
    // 读取图像
    cv::Mat image = cv::imread("kzd.png", cv::IMREAD_COLOR);
    if (image.empty()) 
    {
        std::cerr << "Could not read the image." << std::endl;
        return -1;
    }

    // 转换为灰度图像
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 使用直线段检测
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(gray, lines, 1, CV_PI / 180, 50, 50, 10);

    // 绘制检测到的直线段
    cv::Mat result = image.clone();
    for (size_t i = 0; i < lines.size(); ++i) {
        cv::Vec4i line = lines[i];
        cv::line(result, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // 显示结果
    cv::imshow("Detected Lines", result);
    cv::waitKey(0);

    return 0;
}
