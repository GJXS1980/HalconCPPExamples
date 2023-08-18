#include <iostream>
#include "MechEyeApi.h"
#include "SampleUtil.h"
#include "OpenCVUtil.h"

#include "HalconCpp.h"
#include "HDevThread.h"
#include <opencv2/opencv.hpp>

#include "PclUtil.h"
#include <pcl/io/ply_io.h>

#include "yolov8.h"
#include "cmd_line_util.h"
#include <exception>
#include <csignal> // 包含信号处理的头文件
bool status_isOK = false;

using namespace HalconCpp;

// SIGINT 信号处理函数(Ctrl+C退出)
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

int main(int argc, char *argv[]) 
{
    mmind::api::MechEyeDevice device;
    // 注册信号处理函数
    signal(SIGINT, signalHandler);

    // 创建 VideoWriter 对象
    cv::VideoWriter videoWriter("output.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), 30, cv::Size(1280, 1024));

    if (!findAndConnect(device))
    {
        return -1;
    }

    status_isOK = true;
    mmind::api::ColorMap color;
    YoloV8Config config;
    std::string onnxModelPath;
    std::string colorFile;
    
    cv::Mat img;
    cv::Mat color8UC3;
    int count = 0;
    try 
    {
        while (status_isOK)
        {
            try 
            {
                //  采集照片
                showError(device.captureColorMap(color));
                img = cv::Mat(color.height(), color.width(), CV_8UC3, color.data());
                colorFile = "boxes.png";
                // saveMap(color, colorFile);
                // Read the input image
                // img = cv::imread(color8UC3);
                
            }
            catch(const std::exception& e) 
            {
                std::cout << "Error occurred: " << e.what() << std::endl; // 打印错误信息
                device.disconnect();
                std::cout << "Disconnected from the Mech-Eye device successfully." << std::endl;
                break;
            }

            // Parse the command line arguments
            if (!parseArguments(argc, argv, config, onnxModelPath, colorFile)) 
            {
                return -1;
            }

            // Create the YoloV8 engine
            YoloV8 yoloV8(onnxModelPath, config);
            
            if (img.empty()) {
                std::cout << "Error: Unable to read image at path '" << colorFile << "'" << std::endl;
                return -1;
            }

            // Run inference
            const auto objects = yoloV8.detectObjects(img);

            // Draw the bounding boxes on the image
            yoloV8.drawObjectLabels(img, objects);

            // Save the image to disk
            const auto outputName = colorFile.substr(0, colorFile.find_last_of('.')) + "_annotated.jpg";

            //  保存照片
            cv::imwrite(outputName, img);

            cv::imshow("实例分割Demo", img);
            // 等待用户按下按键
            cv::waitKey(1);
 
        }
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
    }

    videoWriter.release();

    return 0;
}