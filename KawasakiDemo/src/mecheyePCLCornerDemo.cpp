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
#include <pcl/common/common.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

bool status_isOK = false;

// 定义全局变量，用于存储鼠标框选的矩形区域
cv::Rect roiRect;
bool isDrawing = false;

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

// 鼠标事件回调函数
void onMouse(int event, int x, int y, int flags, void* userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN) 
    {
        isDrawing = true;
        roiRect = cv::Rect(x, y, 0, 0);
    } 
    else if (event == cv::EVENT_MOUSEMOVE && isDrawing) 
    {
        roiRect.width = x - roiRect.x;
        roiRect.height = y - roiRect.y;
    } 
    else if (event == cv::EVENT_LBUTTONUP) 
    {
        isDrawing = false;
        roiRect.width = x - roiRect.x;
        roiRect.height = y - roiRect.y;
    }
}


int main() 
{
    // mmind::api::MechEyeDevice device;
    // 注册信号处理函数
    signal(SIGINT, signalHandler);

    // if (!findAndConnect(device))
    // {
    //     return -1;
    // }


    // //  采集彩色点云数据
    // mmind::api::PointXYZBGRMap pointXYZBGRMap;
    // showError(device.capturePointXYZBGRMap(pointXYZBGRMap));
    // const std::string pointCloudColorPath = "pointCloudColor.ply";
    // savePLY(pointXYZBGRMap, pointCloudColorPath);

    // //  采集彩色图
    // mmind::api::ColorMap color;
    // showError(device.captureColorMap(color));
    // std::string colorFile = "color.png";
    // saveMap(color, colorFile);

    // //  相机关闭连接
    // device.disconnect();
    // std::cout << "Disconnected from the Mech-Eye device successfully." << std::endl;


    // 加载彩色图像和点云数据
    cv::Mat colorNoMaskImage = cv::imread("boxes_01.png");   // 没有掩膜

    cv::Mat colorImage = cv::imread("maskColor.jpg");   //  有掩膜

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PLYReader reader;
    reader.read("pointCloudColor.ply", *pointCloud);

    //  创建点云可视化接口
    pcl::visualization::PCLVisualizer::Ptr viewerColor(new pcl::visualization::PCLVisualizer("Segmented Point Cloud"));

    // 创建一个空白的掩膜图像
    cv::Mat grayImage;
    // 将彩色图转成灰度图
    cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat mask = grayImage.clone();

    // 使用Canny边缘检测
    cv::Mat edges;
    cv::Canny(grayImage, edges, 50, 150, 3);  // 调整阈值和内核大小以适应您的图像


    // 在图像中找到墙体角点（优化：增加深度学习方法，获取识别区域掩膜再计算掩膜角点）
    std::vector<cv::Point2f> cornerPoints;
    cv::goodFeaturesToTrack(edges, cornerPoints, 1, 0.01, 10);

    // cv::Mat resultImage;
    for (const cv::Point2f& cornerPoint : cornerPoints) 
    {
        // 在结果图像上绘制红色圆圈
        cv::circle(colorNoMaskImage, cornerPoint, 5, cv::Scalar(0, 0, 255), -1); // -1 表示实心圆
        cv::circle(colorImage, cornerPoint, 5, cv::Scalar(0, 0, 255), -1); // -1 表示实心圆

    }

    //  保存掩膜图像
    const auto cornerName = "cornerimg.png";
    cv::imwrite(cornerName, colorImage);

    const auto cornerNoMaskName = "cornerimgNoMask.png";
    cv::imwrite(cornerNoMaskName, colorNoMaskImage);


    // 将像素值为255的像素转换为像素值为0
    cv::threshold(grayImage, mask, 254, 0, cv::THRESH_BINARY);

    // 在掩膜上绘制角点
    for (const cv::Point2f& point : cornerPoints) 
    {
        cv::circle(mask, point, 5, cv::Scalar(255), -1);
    }

    //  保存掩膜图像
    const auto maskName = "mask.png";
    cv::imwrite(maskName, mask);

    // 点云分割
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmentedCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // 遍历每个点云，并根据掩膜提取掩膜区域的点云
    for (int y = 0; y < mask.rows; ++y) 
    {
        for (int x = 0; x < mask.cols; ++x) 
        {
            // 如果掩膜像素值等于255
            if (mask.at<uchar>(y, x) == 255) 
            { 
                const pcl::PointXYZRGB& pointData = pointCloud->at(x, y); // 提取对应点云中的点
                segmentedCloud->push_back(pointData); // 添加到分割后的点云中
                std::cout << "角点坐标: " << pointData << endl;
            }
        }
    }


    // 保存分割后的点云
    pcl::PLYWriter Writer;
    const auto plyName = "segmented_point_cloud.ply";
    Writer.write(plyName, *segmentedCloud);

    // 创建分割点云可视化
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Segmented Point Cloud"));
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr segcolorCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PLYReader segreader;
    segreader.read("segmented_point_cloud.ply", *segcolorCloud);

    // 可视化点云
    viewer->addPointCloud(segcolorCloud);
    viewerColor->addPointCloud(pointCloud);

    // Spin the viewer
    viewer->spin();
    // // Spin the viewer
    viewerColor->spin();

    return 0;
}
