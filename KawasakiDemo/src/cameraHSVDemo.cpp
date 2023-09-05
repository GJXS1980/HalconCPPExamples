#include <iostream>
#include "MechEyeApi.h"
#include "SampleUtil.h"
#include "OpenCVUtil.h"

#include <opencv2/opencv.hpp>

#include <exception>
#include <csignal> // 包含信号处理的头文件

bool status_isOK = false;

// SIGINT 信号处理函数(Ctrl+C断开相机连接并退出程序)
void signalHandler(int signum) 
{
    status_isOK = false;
    std::cout << "Ctrl+C pressed. Exiting..." << std::endl;
    // 断开相机连接
    mmind::api::MechEyeDevice device;
    device.disconnect();
    std::cout << "Disconnected from the Mech-Eye device successfully." << std::endl;

    //  退出程序
    exit(signum);
}


int main() 
{
    mmind::api::MechEyeDevice device;
    mmind::api::ColorMap color;
    std::string colorFile;
    cv::Mat img;

    // 注册信号处理函数
    signal(SIGINT, signalHandler);

    if (!findAndConnect(device))
    {
        return -1;
    }

    //  相机连接成功
    status_isOK = true;

    try 
    {
        //  循环采集照片并做实例分割
        while (status_isOK)
        {
            try 
            {
                //  采集照片
                showError(device.captureColorMap(color));
                img = cv::Mat(color.height(), color.width(), CV_8UC3, color.data());
                colorFile = "boxes.png";
                
            }
            catch(const std::exception& e) 
            {
                std::cout << "Error occurred: " << e.what() << std::endl; // 打印错误信息
                device.disconnect();
                std::cout << "Disconnected from the Mech-Eye device successfully." << std::endl;
                break;
            }

            if (img.empty()) 
            {
                std::cerr << "无法读取图像" << std::endl;
                return -1;
            }
            

            // 将图像转换为HSV颜色空间
            cv::Mat hsvImage;
            cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);

            // 定义绿色的HSV范围
            cv::Scalar lowerGreen = cv::Scalar(43, 31, 32); // HSV下界
            cv::Scalar upperGreen = cv::Scalar(100, 154, 254); // HSV上界

            // 对图像进行颜色阈值分割，提取绿色区域
            cv::Mat greenMask;
            cv::inRange(hsvImage, lowerGreen, upperGreen, greenMask);

            // 对二值图像进行腐蚀和膨胀操作，以去除噪点
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::morphologyEx(greenMask, greenMask, cv::MORPH_OPEN, kernel);

            // 使用闭操作来补全断开的绿色带部分
            cv::morphologyEx(greenMask, greenMask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);   // 增加迭代次数

            // 使用霍夫直线变换检测直线
            std::vector<cv::Vec4i> lines;
            cv::HoughLinesP(greenMask, lines, 1, CV_PI / 180, 100, 50, 10);

            // 设置阈值，只绘制长度大于该值的直线
            int minLineLength = 0;

            // 在原始图像上绘制检测到的直线
            cv::Mat resultImage = img.clone();
            for (size_t i = 0; i < lines.size(); ++i) 
            {
                cv::Vec4i line = lines[i];
                // cv::line(resultImage, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2);

                // 计算直线长度
                double lineLength = cv::norm(cv::Point(line[0], line[1]) - cv::Point(line[2], line[3]));

                // 仅绘制长度大于阈值的直线
                if (lineLength >= minLineLength) 
                {
                    cv::line(resultImage, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2);
                }

            }

            // 显示结果
            cv::imshow("原始图像", img);
            cv::imshow("绿色直线检测结果", resultImage);
            cv::imshow("绿色掩膜图像", greenMask);

            cv::waitKey(1);
 
        }
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
    }

    return 0;
}
