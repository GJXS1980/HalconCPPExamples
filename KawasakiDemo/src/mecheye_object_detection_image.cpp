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
#include <pcl/point_types.h>
#include <typeinfo>

using namespace HalconCpp;

// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) 
{
    mmind::api::MechEyeDevice device;
    if (!findAndConnect(device))
        return -1;

    //  采集彩色点云数据
    mmind::api::PointXYZBGRMap pointXYZBGRMap;
    showError(device.capturePointXYZBGRMap(pointXYZBGRMap));
    const std::string pointCloudColorPath = "pointCloudColor.ply";
    savePLY(pointXYZBGRMap, pointCloudColorPath);

    //  采集彩色图
    mmind::api::ColorMap color;
    showError(device.captureColorMap(color));
    std::string colorFile = "boxes.png";
    saveMap(color, colorFile);

    //  相机关闭连接
    device.disconnect();
    std::cout << "Disconnected from the Mech-Eye device successfully." << std::endl;

    YoloV8Config config;
    std::string onnxModelPath;

    // 从命令行输入参数
	if (!parseArguments(argc, argv, config, onnxModelPath, colorFile)) 
    {
		return -1;
    }

     // 创建yolov8配置参数
    YoloV8 yoloV8(onnxModelPath, config);

    // 读取输入文件
    auto img = cv::imread(colorFile);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << colorFile << "'" << std::endl;
        return -1;
    }

    // 运行识别
    const auto objects = yoloV8.detectObjects(img);

    //在图片上面画框
    yoloV8.drawObjectLabels(img, objects);

    // 保存照片
    const auto outputName = colorFile.substr(0, colorFile.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;

    //  点云处理
    int count = 0;
    for (const auto& object: objects) 
    {
        if (object.label == 1)
        {
            cv::Mat grayImage;
            // 将彩色图转成灰度图
            cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);
            cv::Mat mask = grayImage.clone();

            // 将像素值为255的像素转换为像素值为0
            cv::threshold(grayImage, mask, 254, 0, cv::THRESH_BINARY);

            //  设置像素值为255
            cv::Scalar color = cv::Scalar(1, 1, 1);
            //  在图片上画出掩膜区域
            mask(object.rect).setTo(color * 255, object.boxMask);
            cv::addWeighted(grayImage, 0.5, mask, 0.8, 1, grayImage);

            //  保存分割后的灰度图
            const auto maskName = "mask" + std::to_string(count) + ".png";
            cv::imwrite(maskName, grayImage);
            
            // 读取PLY点云
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::PLYReader reader;
            reader.read("pointCloudColor.ply", *cloud);

            // 点云分割
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmentedCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

            // 遍历每个点云，并根据掩膜提取掩膜区域的点云
            for (int y = 0; y < grayImage.rows; ++y) {
                for (int x = 0; x < grayImage.cols; ++x) 
                {
                    // 如果掩膜像素值等于255
                    if (grayImage.at<uchar>(y, x) == 255) 
                    { 
                        const pcl::PointXYZRGB& point = cloud->at(x, y); // 提取对应点云中的点
                        segmentedCloud->push_back(point); // 添加到分割后的点云中
                    }
                }
            }

            // 保存分割后的点云
            pcl::PLYWriter Writer;
            const auto plyName = "segmented_point_cloud" + std::to_string(count) + ".ply";
            Writer.write(plyName, *segmentedCloud);
        }

        count ++;
    }


    return 0;
}