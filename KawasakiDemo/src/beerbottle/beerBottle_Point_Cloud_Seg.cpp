#include <iostream>
#include "MechEyeApi.h"
#include "SampleUtil.h"
#include "OpenCVUtil.h"
#include <opencv2/opencv.hpp>

// #include "boxesYolov8.h"
// #include "cmd_line_utilBoxes.h"
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

// /**
//  * @brief 将旋转矩阵转换为 ZYZ 欧拉角（角度制）
//  * 
//  * @param rotationMatrix 旋转矩阵
//  * @return Eigen::Vector3d ZYZ 欧拉角（角度制），顺序为 phi, theta, psi
//  */
// Eigen::Vector3d rotationMatrixToZYZEulerAngles(const Eigen::Matrix3d& rotationMatrix) 
// {
//     double phi, theta, psi;

//     // 计算 theta 的范围在 [0, pi]
//     theta = atan2(sqrt(rotationMatrix(2, 0) * rotationMatrix(2, 0) + rotationMatrix(2, 1) * rotationMatrix(2, 1)), rotationMatrix(2, 2));

//     // 处理 theta 为 0 或 pi 的情况
//     if (theta < 1e-6) 
//     {
//         phi = atan2(rotationMatrix(0, 1), rotationMatrix(0, 0));
//         psi = 0.0;
//     } 
//     else if (theta > M_PI - 1e-6) 
//     {
//         phi = -atan2(rotationMatrix(0, 1), rotationMatrix(0, 0));
//         psi = 0.0;
//     } 
//     else 
//     {
//         phi = atan2(rotationMatrix(2, 1) / sin(theta), rotationMatrix(2, 0) / sin(theta));
//         psi = atan2(rotationMatrix(1, 2) / sin(theta), -rotationMatrix(0, 2) / sin(theta));
//     }

//     return Eigen::Vector3d(phi*180.00/M_PI, theta*180.00/M_PI, psi*180.00/M_PI);
// }


// /**
//  * @brief 将位姿pose数据转换成齐次变换矩阵
//  * 
//  * @param double x, double y, double z, double qw, double qx, double qy, double qz  位姿xyz和四元数的值
//  * @return transform_cam_to_base.matrix() 转换的齐次变换矩阵
//  */
// Eigen::Matrix4d transformMatrixFromPose(double x, double y, double z, double qw, double qx, double qy, double qz)
// {
//     // 创建平移向量
//     Eigen::Vector3d translation(x, y, z);
//     // 创建四元数
//     Eigen::Quaterniond quaternion(qw, qx, qy, qz);
//     // 创建四元数
//     quaternion.normalize();
//     // 创建齐次变换矩阵
//     Eigen::Affine3d transform = Eigen::Translation3d(translation) * quaternion;
//     // 打印变换矩阵
//     // std::cout << "生成齐次变换矩阵:" << std::endl;
//     std::cout << transform.matrix() << std::endl;
//     return transform.matrix();
// }


// Runs object detection on an input image then saves the annotated image to disk.
int main() 
{
    std::string colorFile = "mask_image_white1.jpg";

    // 读取输入文件
    auto img = cv::imread(colorFile);
    if (img.empty()) 
    {
        std::cout << "Error: Unable to read image at path '" << colorFile << "'" << std::endl;
        return -1;
    }

    cv::Mat grayImage;
    // 将彩色图转成灰度图
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat mask = grayImage.clone();

    // 读取PLY点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PLYReader reader;
    reader.read("pointCloudColor.ply", *cloud);

    // 点云分割
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmentedCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // 遍历每个点云，并根据掩膜提取掩膜区域的点云
    for (int y = 0; y < grayImage.rows; ++y) 
    {
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
    const auto plyName = "segmented_point_cloud.ply";
    Writer.write(plyName, *segmentedCloud);

    // 点云聚类

    /*              求解平面中心点坐标和法向量        */
    // 创建点云指针，用于存储加载的点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr segCloud(new pcl::PointCloud<pcl::PointXYZ>);

        // 从PLY文件中加载点云数据到segCloud
    pcl::io::loadPLYFile<pcl::PointXYZ>(plyName, *segCloud);

    // 创建法线点云指针，用于存储计算得到的法线信息
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    // 创建法线估计对象
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;

    // 设置法线估计对象的输入点云数据
    normalEstimation.setInputCloud(segCloud);

    // 创建KdTree用于法线估计的搜索
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normalEstimation.setSearchMethod(tree);
    normalEstimation.setRadiusSearch(0.003); // 根据点云密度调整搜索半径(0.03)
    // 计算法线估计并存储到normals中
    normalEstimation.compute(*normals);

    // 创建欧几里德聚类提取对象
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> clusterExtractor;
    clusterExtractor.setClusterTolerance(0.01);  // 设置聚类的欧几里德距离阈值(3.0)
    clusterExtractor.setMinClusterSize(100);     // 设置最小聚类点数(800)
    clusterExtractor.setMaxClusterSize(3000000);    // 设置最大聚类点数
    clusterExtractor.setInputCloud(segCloud);    // 设置输入点云数据

    // 存储提取的聚类索引
    std::vector<pcl::PointIndices> clusters;
    // 执行聚类提取
    clusterExtractor.extract(clusters);


    // 创建点云可视化
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));

    // 可视化点云
    viewer->addPointCloud(cloud);


    // 创建分割点云可视化
    pcl::visualization::PCLVisualizer::Ptr viewerSeg(new pcl::visualization::PCLVisualizer("Seg Point Cloud Viewer"));
    // 可视化分割点云
    viewerSeg->addPointCloud(segCloud);


    // 遍历每个聚类
    int clusterNum = 1;
    for (const auto& indices : clusters) 
    {
        // 创建一个新的点云指针，用于存储当前聚类的点云数据
        pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZ>);

        // 将当前聚类中的点索引对应的点数据存入clusterCloud
        for (const auto& index : indices.indices)
        {
            clusterCloud->push_back(segCloud->points[index]);
        }

        // 计算当前聚类的质心坐标
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*clusterCloud, centroid);

        // 计算当前聚类的协方差矩阵
        Eigen::Matrix3f covariance_matrix;
        Eigen::Vector4f centroid4f;
        pcl::computeCovarianceMatrixNormalized(*clusterCloud, centroid4f, covariance_matrix);

        // 使用SelfAdjointEigenSolver计算协方差矩阵的特征向量
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix);
        Eigen::Vector3f normal = eigen_solver.eigenvectors().col(0);

        // 输出当前聚类的中心坐标和法向量
        std::cout << "Cluster " << clusterNum << " Center: " << centroid << std::endl;
        std::cout << "Cluster " << clusterNum << " Normal: " << normal << std::endl;

        /*                  可视化抓取点        */

        // 可视化质心xyz坐标值
        pcl::PointXYZ centroid_point;
        centroid_point.x = centroid[0];
        centroid_point.y = centroid[1];
        centroid_point.z = centroid[2];

        // Define coordinate system colors
        Eigen::Vector3i colors[3];
        colors[0] = Eigen::Vector3i(255, 0, 0); // X轴为红色
        colors[1] = Eigen::Vector3i(0, 255, 0); // Y轴为绿色
        colors[2] = Eigen::Vector3i(0, 0, 255); // Z轴为蓝色

        // 在三个坐标轴（X、Y、Z）上分别进行循环
        for (int i = 0; i < 3; ++i) 
        {
            // 创建一个仿射变换矩阵，初始化为单位矩阵
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();

            // 设置变换矩阵的平移部分，即将坐标原点放置在点云的质心位置
            transform.translation() << centroid[0],centroid[1],centroid[2];

            switch (clusterNum) 
            {
                case 1:
                {
                    // 创建一个零向量，表示坐标轴的方向
                    Eigen::Vector3f axis1 = Eigen::Vector3f::Zero();

                    // 将当前坐标轴的分量设置为1，表示该坐标轴方向上有值
                    axis1[i] = 1.0;
                    // 创建两个点 p1 和 p2 用于绘制坐标轴线段
                    pcl::PointXYZ p11, p12;
                    // 将 p1 设置为质心位置
                    p11.getVector3fMap() = centroid.head<3>();
                    // 将 p2 设置为质心位置加上坐标轴方向的一点，构成线段
                    p12.getVector3fMap() = centroid.head<3>() + axis1 * 0.1;
                    // 在可视化器中添加线段，用于绘制坐标轴
                    viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(p11, p12, colors[i][0], colors[i][1], colors[i][2], "axis1_" + std::to_string(i), 0);
                    std::cout << "p11: " << p11 << std::endl;
                    std::cout << "p12: " << p12 << std::endl;
                    break;
                }

                case 2:
                {
                    // 创建一个零向量，表示坐标轴的方向
                    Eigen::Vector3f axis2 = Eigen::Vector3f::Zero();

                    // 将当前坐标轴的分量设置为1，表示该坐标轴方向上有值
                    axis2[i] = 1.0;
                    // 创建两个点 p1 和 p2 用于绘制坐标轴线段
                    pcl::PointXYZ p21, p22;
                    // 将 p1 设置为质心位置
                    p21.getVector3fMap() = centroid.head<3>();
                    // 将 p2 设置为质心位置加上坐标轴方向的一点，构成线段
                    p22.getVector3fMap() = centroid.head<3>() + axis2 * 0.1;
                    // 在可视化器中添加线段，用于绘制坐标轴
                    viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(p21, p22, colors[i][0], colors[i][1], colors[i][2], "axis2_" + std::to_string(i), 0);
                    std::cout << "p21: " << p21 << std::endl;
                    std::cout << "p22: " << p22 << std::endl;
                    break;
                }
                case 3:
                {
                    // 创建一个零向量，表示坐标轴的方向
                    Eigen::Vector3f axis3 = Eigen::Vector3f::Zero();

                    // 将当前坐标轴的分量设置为1，表示该坐标轴方向上有值
                    axis3[i] = 1.0;
                    // 创建两个点 p1 和 p2 用于绘制坐标轴线段
                    pcl::PointXYZ p31, p32;
                    // 将 p1 设置为质心位置
                    p31.getVector3fMap() = centroid.head<3>();
                    // 将 p2 设置为质心位置加上坐标轴方向的一点，构成线段
                    p32.getVector3fMap() = centroid.head<3>() + axis3 * 0.1;
                    // 在可视化器中添加线段，用于绘制坐标轴
                    viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(p31, p32, colors[i][0], colors[i][1], colors[i][2], "axis3_" + std::to_string(i), 0);
                    std::cout << "p31: " << p31 << std::endl;
                    std::cout << "p32: " << p32 << std::endl;
                    break;
                }
                default:
                    break;

            }
        }

        clusterNum++;
    }



    // Spin the viewer
    viewer->spin();
    viewerSeg->spin();


    // 点云形状检测器

    // 获取最高层点云


    // 计算平面点云的位姿和尺寸




    return 0;
}