#include <iostream>
#include "MechEyeApi.h"
#include "SampleUtil.h"

#include "HalconCpp.h"
#include "HDevThread.h"
#include <opencv2/opencv.hpp>

#include "PclUtil.h"
#include <pcl/io/ply_io.h>

#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>

using namespace HalconCpp;
using namespace std;

int main() {
    // 读取.ply文件
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPLYFile("pointcloud.ply", *cloud);
    //  点云为空，则退出程序
    if (cloud->empty()) {
        std::cerr << "Point cloud is empty!" << std::endl;
        return 0;
        // Handle the error or return from the function
    }
    // 读取掩膜图像
    cv::Mat maskImage = cv::imread("boxes_mask_01.png", cv::IMREAD_GRAYSCALE);

    // 提取掩膜区域的点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr maskedCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (int y = 548; y < 629; y++) {
        for (int x = 290; x < 392; x++) {
            if (maskImage.at<uchar>(y, x) > 0) {
                maskedCloud->push_back(cloud->at(x, y));
            }
        }
    }

    // 保存提取的点云
    pcl::io::savePLYFile("MaskPointCloud.ply", *maskedCloud);
    // 点云聚类
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(maskedCloud);
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(0.02); // 设置聚类的容差
    ec.setMinClusterSize(0.1);    // 设置最小聚类点数
    ec.setMaxClusterSize(2500);  // 设置最大聚类点数
    ec.setSearchMethod(tree);
    ec.setInputCloud(maskedCloud);
    ec.extract(clusterIndices);

    // 点云上表面拟合和位姿获取
    for (const auto &indices : clusterIndices) 
    {
        // 创建一个用于存储聚类点云的PointCloud对象
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        // 将属于当前聚类的点添加到clusterCloud中
        for (const auto &index : indices.indices) {
            clusterCloud->push_back(maskedCloud->at(index));
        }

        // 创建对象用于存储拟合结果的系数
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
         // 创建对象用于存储内点索引
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
         // 创建一个SACSegmentation对象用于平面拟合
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        seg.setOptimizeCoefficients(true);   // 优化拟合系数
        seg.setModelType(pcl::SACMODEL_PLANE);  // 设置拟合模型为平面模型
        seg.setMethodType(pcl::SAC_RANSAC); // 使用RANSAC算法进行拟合
        seg.setMaxIterations(100);  // 设置最大迭代次数
        seg.setInputCloud(clusterCloud);    // 设置输入点云
        seg.segment(*inliers, *coefficients);   // 进行平面拟合

        // 判断是否找到了拟合模型（内点数量大于0）
        if (inliers->indices.size() > 0) 
        {
            // 从拟合系数中提取平面的法向量
            Eigen::Vector3f plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
            // 获取聚类点云的原点，用于确定平面上的点
            Eigen::Vector3f plane_centroid(clusterCloud->sensor_origin_[0], clusterCloud->sensor_origin_[1], clusterCloud->sensor_origin_[2]);

            // 在这里进行位姿计算和保存
            // ...

            // 输出拟合结果
            std::cout << "Surface normal: " << plane_normal << std::endl;
            std::cout << "Centroid: " << plane_centroid << std::endl;
        }
    }

    return 0;
}














