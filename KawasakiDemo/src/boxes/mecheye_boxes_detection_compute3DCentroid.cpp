#include <iostream>
#include "MechEyeApi.h"
#include "SampleUtil.h"
#include "OpenCVUtil.h"
#include <opencv2/opencv.hpp>

#include "yolov8.h"
#include "cmd_line_util.h"
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
#include <pcl/features/normal_3d.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

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

// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) 
{
    //  初始化参数
    double x = 0.0, y = 0.0, z = 0.0;
    double qw = 0.0, qx = 0.0, qy = 0.0, qz = 0.0;

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

    // 读取输入文件
    auto color_img = cv::imread(colorFile);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << colorFile << "'" << std::endl;
        return -1;
    }

    std::cout << "选择要识别的物体：" << std::endl;
    std::cout << "1. 大药盒" << std::endl;
    std::cout << "2. 中药盒" << std::endl;
    std::cout << "3. 小药盒" << std::endl;
    int choice;
    std::cout << "请输入选项的编号：";
    std::cin >> choice;

    //  点云处理
    int count = 0;
    for (const auto& object: objects) 
    {

        if (object.label == (choice - 1))
        {

            cv::Mat grayImage;
            // 将彩色图转成灰度图
            cv::cvtColor(color_img, grayImage, cv::COLOR_BGR2GRAY);
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
            const auto plyName = "segmented_point_cloud" + std::to_string(count) + ".ply";
            Writer.write(plyName, *segmentedCloud);

            /*                  计算点云质心坐标和法向量                    */
            pcl::PointCloud<pcl::PointXYZ>::Ptr segCloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::io::loadPLYFile<pcl::PointXYZ>(plyName, *segCloud);

            // 计算点云质心坐标
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*segCloud, centroid);

            // Estimate normals
            pcl::PointCloud<pcl::Normal>::Ptr segNormals(new pcl::PointCloud<pcl::Normal>);
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
            normal_estimator.setInputCloud(segCloud);
            normal_estimator.setRadiusSearch(0.03); // Adjust this value as needed
            normal_estimator.compute(*segNormals);

            // Assume Z-axis as reference direction
            Eigen::Vector3f reference_direction = Eigen::Vector3f::UnitZ();

            // Compute rotation matrix
            Eigen::Matrix3f rotation_matrix;
            Eigen::Vector3f source_normal = segNormals->at(0).getNormalVector3fMap();
            rotation_matrix = Eigen::Quaternionf::FromTwoVectors(source_normal, reference_direction).toRotationMatrix();

            // 获取xyz和四元数
            Eigen::Quaternionf quaternion(rotation_matrix);
            x = centroid[0], y = centroid[1], z = centroid[2];
            qw = quaternion.w(), qx = quaternion.x(), qy = quaternion.y(), qz = quaternion.z();
            std::cout << "Centroid: " << centroid[0] << " " << centroid[1] << " " << centroid[2] << std::endl;
            std::cout << "Quaternion: " << quaternion.w() << " " << quaternion.x() << " " << quaternion.y() << " " << quaternion.z() << std::endl;

            /*                  可视化抓取点坐标系                         */

            // 创建点云可视化
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));

            // 可视化点云
            viewer->addPointCloud(segmentedCloud);

            // 可视化质心
            pcl::PointXYZ centroid_point;
            centroid_point.x = centroid[0];
            centroid_point.y = centroid[1];
            centroid_point.z = centroid[2];

            // 定义坐标系颜色
            Eigen::Vector3i colors[3];
            colors[0] = Eigen::Vector3i(255, 0, 0); // Red for X-axis
            colors[1] = Eigen::Vector3i(0, 255, 0); // Green for Y-axis
            colors[2] = Eigen::Vector3i(0, 0, 255); // Blue for Z-axis

            // 在三个坐标轴（X、Y、Z）上分别进行循环
            for (int i = 0; i < 3; ++i) 
            {
                // 创建一个零向量，表示坐标轴的方向
                Eigen::Vector3f axis = Eigen::Vector3f::Zero();
                // 将当前坐标轴的分量设置为1，表示该坐标轴方向上有值
                axis[i] = 1.0;
                // 创建一个仿射变换矩阵，初始化为单位矩阵
                Eigen::Affine3f transform = Eigen::Affine3f::Identity();
                // 设置变换矩阵的平移部分，即将坐标原点放置在点云的质心位置
                transform.translation() << centroid[0],centroid[1],centroid[2];
                // transform.translation() << x,y,z;

                // 创建两个点 p1 和 p2 用于绘制坐标轴线段
                pcl::PointXYZ p1, p2;
                    // 将 p1 设置为质心位置
                p1.getVector3fMap() = centroid.head<3>();
                // 将 p2 设置为质心位置加上坐标轴方向的一点，构成线段
                p2.getVector3fMap() = centroid.head<3>() + axis * 0.1;
                // 在可视化器中添加线段，用于绘制坐标轴
                viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(p1, p2, colors[i][0], colors[i][1], colors[i][2], "axis_" + std::to_string(i), 0);
            }

            /*                  求解物体相对于机器人基座位姿      */
            //  相机相对于机器人基座的齐次变换矩阵
            Eigen::Matrix4d transform_cam_to_base = transformMatrixFromPose(-0.329298, 1.03579, 1.15312, 0.0496425, 0.00719909, 0.998676, -0.0113733);
            //  识别到物体相对于相机的齐次变换矩阵
            Eigen::Matrix4d transform_obj_to_cam = transformMatrixFromPose(x, y, z, qw, qx, qy, qz);

            // 计算物体相对于机器人基座的位姿
            Eigen::Matrix4d result = transform_cam_to_base * transform_obj_to_cam.matrix();
            result(2,3) = result(2, 3) + 0.195; // 增加末端执行器的长度

            // 输出结果
            std::cout << "Resulting Homogeneous Transformation Matrix:" << std::endl;
            std::cout << result << std::endl;

            // 提取平移向量
            Eigen::Vector3d translation = result.block<3, 1>(0, 3);

            // 提取旋转矩阵
            Eigen::Matrix3d rotationMatrix = result.block<3, 3>(0, 0);

            // 将旋转矩阵转换为 ZYZ 欧拉角
            Eigen::Vector3d robot_euler_angles = rotationMatrixToZYZEulerAngles(rotationMatrix);

            // 输出平移向量和欧拉角
            std::cout << "Translation vector: " << translation.transpose() << std::endl;
            std::cout << "ZYZ Euler angles (phi, theta, psi): " << robot_euler_angles.transpose() << std::endl;
            
            // 保持点云可视化窗口打开
            viewer->spin();
        }
        count ++;
    }
    return 0;
}