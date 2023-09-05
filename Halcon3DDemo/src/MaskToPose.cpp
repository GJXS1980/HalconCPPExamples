#include <iostream>
#include <cmath>

#include "MechEyeApi.h"
#include "SampleUtil.h"

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

using namespace std;

int main() 
{
    // 读取点云文件
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPLYFile("../pointCloudColor.ply", *cloud);

    // 定义掩膜范围
    int x_min = 673; // 填入实际的x_min值
    int x_max = 773; // 填入实际的x_max值
    int y_min = 577; // 填入实际的y_min值
    int y_max = 667; // 填入实际的y_max值

 // 提取掩膜区域的点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr maskedCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (int y = y_min; y <= y_max; y++) {
        for (int x = x_min; x <= x_max; x++) {
            maskedCloud->push_back(cloud->at(x, y));
        }
    }

    // 保存提取的点云
    pcl::io::savePLYFile("../MaskPointCloud.ply", *maskedCloud);

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
        double x = 0.0, y = 0.0, z = 0.0;
        double normal_x = 0.0, normal_y = 0.0, normal_z = 0.0;
        double count = 0.0;

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

        // 输出旋转四元数的系数
        std::cout << "四元数(qx, qy, qz, qw)为: " << std::endl << quaternion.coeffs() << std::endl;

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

    return 0;
}
