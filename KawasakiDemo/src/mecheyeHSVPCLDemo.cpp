#include <iostream>
#include "MechEyeApi.h"
#include "SampleUtil.h"
#include "OpenCVUtil.h"

#include <opencv2/opencv.hpp>

#include <exception>
#include <csignal> // 包含信号处理的头文件

#include <typeinfo>
#include <cmath>

#include "PclUtil.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

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
    // 注册信号处理函数
    signal(SIGINT, signalHandler);

    if (!findAndConnect(device))
    {
        return -1;
    }


    //  采集彩色点云数据
    mmind::api::PointXYZBGRMap pointXYZBGRMap;
    showError(device.capturePointXYZBGRMap(pointXYZBGRMap));
    const std::string pointCloudColorPath = "pointCloudColor.ply";
    savePLY(pointXYZBGRMap, pointCloudColorPath);

    //  采集彩色图
    mmind::api::ColorMap color;
    showError(device.captureColorMap(color));
    std::string colorFile = "color.png";
    saveMap(color, colorFile);

    //  相机关闭连接
    device.disconnect();
    std::cout << "Disconnected from the Mech-Eye device successfully." << std::endl;

    // 读取输入文件
    auto color_img = cv::imread(colorFile);
    if (color_img.empty()) 
    {
        std::cout << "Error: Unable to read image at path '" << colorFile << "'" << std::endl;
        return -1;
    }

    // 将图像转换为HSV颜色空间
    cv::Mat hsvImage;
    cv::cvtColor(color_img, hsvImage, cv::COLOR_BGR2HSV);

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

    //  保存分割后的灰度图掩膜
    const auto maskName = "mask.png";
    cv::imwrite(maskName, greenMask);

    // // 读取目标掩膜图像
    // // cv::Mat greenMask = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
    // // 定义每个目标的增加宽度
    // int addedWidth = 5;
    // // 创建一个新的图像，用于存储增加宽度后的目标掩膜
    // greenMask = cv::Mat::zeros(greenMask.rows + 2 * addedWidth, greenMask.cols + 2 * addedWidth, CV_8U);

    // // 将目标掩膜复制到新图像的中心区域
    // greenMask.copyTo(greenMask(cv::Rect(addedWidth, addedWidth, greenMask.cols, greenMask.rows)));
    // //  保存分割后的灰度图掩膜
    // const auto maskAddName = "maskAdd.png";
    // cv::imwrite(maskAddName, greenMask);

    // 使用霍夫直线变换检测直线
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(greenMask, lines, 1, CV_PI / 180, 100, 50, 10);

    // 设置阈值，只绘制长度大于该值的直线
    int minLineLength = 0;

    // 在原始图像上绘制检测到的直线
    cv::Mat resultImage = color_img.clone();
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

    //  保存分割后的灰度图掩膜
    const auto linesName = "lines.png";
    cv::imwrite(linesName, resultImage);


    // 读取PLY点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PLYReader reader;
    reader.read("pointCloudColor.ply", *cloud);

    // 点云分割
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmentedCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // 遍历每个点云，并根据掩膜提取掩膜区域的点云
    for (int y = 0; y < greenMask.rows; ++y) 
    {
        for (int x = 0; x < greenMask.cols; ++x) 
        {
            // 如果掩膜像素值等于255
            if (greenMask.at<uchar>(y, x) == 255) 
            { 
                const pcl::PointXYZRGB& point = cloud->at(x, y); // 提取对应点云中的点
                segmentedCloud->push_back(point); // 添加到分割后的点云中
            }
        }
    }

    // 保存分割后的点云
    pcl::PLYWriter Writer;
    const auto plyName = "segmented_point_cloud.ply";
    Writer.write(plyName, *segmentedCloud);

    // 创建彩色点云可视化
    pcl::visualization::PCLVisualizer::Ptr viewerColor(new pcl::visualization::PCLVisualizer("Color Point Cloud"));

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PLYReader colorreader;
    colorreader.read("pointCloudColor.ply", *colorCloud);

    // 可视化点云
    viewerColor->addPointCloud(colorCloud);

    // 创建分割点云可视化
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Segmented Point Cloud"));
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr segcolorCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PLYReader segreader;
    segreader.read("segmented_point_cloud.ply", *segcolorCloud);

    // 可视化点云
    viewer->addPointCloud(segcolorCloud);

    // Spin the viewer
    viewer->spin();
    // Spin the viewer
    viewerColor->spin();

    return 0;
}
