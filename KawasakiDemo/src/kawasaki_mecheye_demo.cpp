#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>

#include "MechEyeApi.h"
#include "SampleUtil.h"

#include "HalconCpp.h"
#include "HDevThread.h"
#include <opencv2/opencv.hpp>

#include "PclUtil.h"
#include <pcl/io/ply_io.h>
#include <pcl/filters/passthrough.h>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>


/**
 * @brief 将旋转矩阵转换为 ZYZ 欧拉角（角度制）
 * 
 * @param rotationMatrix 旋转矩阵
 * @return Eigen::Vector3d ZYZ 欧拉角（角度制），顺序为 phi, theta, psi
 */
Eigen::Vector3d rotationMatrixToZYZEulerAngles(const Eigen::Matrix3d& rotationMatrix) 
{
    double phi, theta, psi;

    // 计算 theta 的范围在 [0, pi]
    theta = atan2(sqrt(rotationMatrix(2, 0) * rotationMatrix(2, 0) + rotationMatrix(2, 1) * rotationMatrix(2, 1)), rotationMatrix(2, 2));

    // 处理 theta 为 0 或 pi 的情况
    if (theta < 1e-6) {
        phi = atan2(rotationMatrix(0, 1), rotationMatrix(0, 0));
        psi = 0.0;
    } else if (theta > M_PI - 1e-6) {
        phi = -atan2(rotationMatrix(0, 1), rotationMatrix(0, 0));
        psi = 0.0;
    } else {
        phi = atan2(rotationMatrix(2, 1) / sin(theta), rotationMatrix(2, 0) / sin(theta));
        psi = atan2(rotationMatrix(1, 2) / sin(theta), -rotationMatrix(0, 2) / sin(theta));
    }

    return Eigen::Vector3d(phi*180.00/M_PI, theta*180.00/M_PI, psi*180.00/M_PI);
}


/**
 * @brief 将位姿pose数据转换成齐次变换矩阵
 * 
 * @param double x, double y, double z, double qw, double qx, double qy, double qz  位姿xyz和四元数的值
 * @return transform_cam_to_base.matrix() 转换的齐次变换矩阵
 */
Eigen::Matrix4d transformMatrixFromPose(double x, double y, double z, double qw, double qx, double qy, double qz)
{
    // 创建平移向量
    Eigen::Vector3d translation(x, y, z);
    // 创建四元数
    Eigen::Quaterniond quaternion(qw, qx, qy, qz);
    // 创建四元数
    quaternion.normalize();
    // 创建齐次变换矩阵
    Eigen::Affine3d transform = Eigen::Translation3d(translation) * quaternion;
    // 打印变换矩阵
    std::cout << "生成齐次变换矩阵:" << std::endl;
    std::cout << transform.matrix() << std::endl;
    return transform.matrix();
}


int main() 
{

    double x = 0.0, y = 0.0, z = 0.0;
    double qw = 0.0, qx = 0.0, qy = 0.0, qz = 0.0;
    double normal_x = 0.0, normal_y = 0.0, normal_z = 0.0;
    double count = 0.0;

    // 读取点云文件
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPLYFile("pointCloudColor.ply", *cloud);

    // 定义掩膜范围
    int x_min = 296; // 填入实际的x_min值
    int x_max = 400; // 填入实际的x_max值
    int y_min = 527; // 填入实际的y_min值
    int y_max = 615; // 填入实际的y_max值

 // 提取掩膜区域的点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr maskedCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (int y = y_min; y <= y_max; y++) {
        for (int x = x_min; x <= x_max; x++) {
            maskedCloud->push_back(cloud->at(x, y));
        }
    }

    // 保存提取的点云
    pcl::io::savePLYFile("MaskPointCloud.ply", *maskedCloud);

    // 创建法向量估计对象
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud(maskedCloud);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    ne.setSearchMethod(tree);

    // 设置法向量估计的参数
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(500);  // 设置每个点的最近邻数
    ne.compute(*cloud_normals);

    // 点云聚类
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr cluster_tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    cluster_tree->setInputCloud(maskedCloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(0.065); // 聚类容差
    ec.setMinClusterSize(5);     // 最小聚类点数
    ec.setMaxClusterSize(1000);  // 最大聚类点数
    ec.setSearchMethod(cluster_tree);
    ec.setInputCloud(maskedCloud);
    ec.extract(cluster_indices);

    // 对每个聚类进行处理
    for (const auto &indices : cluster_indices) 
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (const auto &index : indices.indices) {
            clusterCloud->push_back(maskedCloud->at(index));
        }

        // 计算法向量和中心点坐标
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne_cluster;
        ne_cluster.setInputCloud(clusterCloud);
        ne_cluster.setSearchMethod(tree);
        ne_cluster.setKSearch(50);
        pcl::PointCloud<pcl::Normal>::Ptr cluster_normals(new pcl::PointCloud<pcl::Normal>);
        ne_cluster.compute(*cluster_normals);

        // 计算聚类点云的平均中心点和法向量
        Eigen::Vector3f avg_centroid(0.0, 0.0, 0.0);
        Eigen::Vector3f avg_normal(0.0, 0.0, 0.0);
        size_t num_points = cluster_normals->size();


        // 点云上表面拟合和位姿获取
        for (size_t i = 0; i < cloud_normals->size(); ++i) 
        {
            if (!pcl::isFinite<pcl::Normal>((*cloud_normals)[i])) {
                continue;  // 跳过无效法向量
            }

            // 获取法向量和点的坐标
            Eigen::Vector3f normal = cloud_normals->at(i).getNormalVector3fMap();
            pcl::PointXYZRGB point = maskedCloud->at(i);

            // 计算平面中心坐标
            Eigen::Vector3f plane_centroid(point.x, point.y, point.z);

            //  对聚类的xyz坐标进行累计
            x += plane_centroid.x();
            y += plane_centroid.y();
            z += plane_centroid.z();

            //  对聚类的法向量进行累计
            normal_x += normal.x();
            normal_y += normal.y();
            normal_z += normal.z();
            count += 1;

        }

        x /= count;
        y /= count;
        z /= count;
        normal_x /= count;
        normal_y /= count;
        normal_z /= count;
        std::cout << "平面中心点(x, y, z): " << "(" << x << "," << y << "," << z << ")" << std::endl;
        std::cout << "平面的法向量: " << "(" << normal_x << "," << normal_y << "," << normal_z << ")" << std::endl;

        // 法向量
        Eigen::Vector3f normal(normal_x, normal_y, normal_z); 

        // 计算旋转矩阵
        Eigen::Vector3f up(0, 0, 1); // 参考向上的向量
        Eigen::Vector3f axis = normal.cross(up);
        float angle = std::acos(normal.dot(up));
        Eigen::Matrix3f rotation;
        rotation = Eigen::AngleAxisf(angle, axis);

        // 构造旋转四元数(在Eigen库中，四元数的构造方式默认使用的是XYZW的顺序)
        Eigen::Quaternionf quaternion;
        quaternion = Eigen::AngleAxisf(angle, axis);
        qw = quaternion.w();
        qx = quaternion.x();
        qy = quaternion.y();
        qz = quaternion.z();

        // 输出旋转四元数的系数
        std::cout << "四元数(qx, qy, qz, qw)为: " << std::endl << quaternion.coeffs() << std::endl;
        std::cout << "四元数(qx, qy, qz, qw)为: " << std::endl << quaternion.w() << std::endl;

        // 提取欧拉角 (ZYZ顺序)
        Eigen::Vector3f euler_angles = rotation.eulerAngles(2, 1, 2); // ZYZ顺序的欧拉角

        // 输出欧拉角(弧度)
        std::cout << "First Rotation (Z): " << euler_angles[0] << " radians" << std::endl;
        std::cout << "Second Rotation (Y): " << euler_angles[1] << " radians" << std::endl;
        std::cout << "Third Rotation (Z): " << euler_angles[2] << " radians" << std::endl;

        // 输出欧拉角(弧度)
        std::cout << "First Rotation (Z): " << euler_angles[0] * 180 / M_PI << " ° " << std::endl;
        std::cout << "Second Rotation (Y): " << euler_angles[1] * 180 / M_PI << " ° " << std::endl;
        std::cout << "Third Rotation (Z): " << euler_angles[2] * 180 / M_PI << " ° " << std::endl;

    }

    //  相机相对于机器人基座的齐次变换矩阵
    Eigen::Matrix4d transform_cam_to_base = transformMatrixFromPose(-0.329298, 1.03579, 1.15312, 0.0496425, 0.00719909, 0.998676, -0.0113733);
    //  识别到物体相对于相机的齐次变换矩阵
    Eigen::Matrix4d transform_obj_to_cam = transformMatrixFromPose(x, y, z, qw, qx, qy, qz);

    // 计算物体相对于机器人基座的位姿
    Eigen::Matrix4d result = transform_cam_to_base * transform_obj_to_cam.matrix();
    result(2,3) = result(2, 3) + 0.179; // 末端执行器的长度

    // 输出结果
    std::cout << "Resulting Homogeneous Transformation Matrix:" << std::endl;
    std::cout << result << std::endl;

    // 提取平移向量
    Eigen::Vector3d translation = result.block<3, 1>(0, 3);

    // 提取旋转矩阵
    Eigen::Matrix3d rotationMatrix = result.block<3, 3>(0, 0);

    // 将旋转矩阵转换为 ZYZ 欧拉角
    Eigen::Vector3d euler_angles = rotationMatrixToZYZEulerAngles(rotationMatrix);

    // 输出平移向量和欧拉角
    std::cout << "Translation vector: " << translation.transpose() << std::endl;
    std::cout << "ZYZ Euler angles (phi, theta, psi): " << euler_angles.transpose() << std::endl;

    return 0;
}
