#include <iostream>
#include "MechEyeApi.h"
#include "SampleUtil.h"
#include "OpenCVUtil.h"
#include <opencv2/opencv.hpp>

#include "yolov8.h"
#include "cmd_line_util.h"
#include <typeinfo>
#include <cmath>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

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
    if (theta < 1e-6) 
    {
        phi = atan2(rotationMatrix(0, 1), rotationMatrix(0, 0));
        psi = 0.0;
    } 
    else if (theta > M_PI - 1e-6) 
    {
        phi = -atan2(rotationMatrix(0, 1), rotationMatrix(0, 0));
        psi = 0.0;
    } 
    else 
    {
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
    // std::cout << "生成齐次变换矩阵:" << std::endl;
    std::cout << transform.matrix() << std::endl;
    return transform.matrix();
}


// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) 
{

    // 创建套接字
    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) 
    {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }
    // 设置服务器地址
    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = inet_addr("192.168.11.55"); // 使用本地IP地址
    serverAddress.sin_port = htons(50000); // 设置端口号

    // 绑定套接字
    if (bind(serverSocket, (struct sockaddr*)&serverAddress, sizeof(serverAddress)) == -1) 
    {
        std::cerr << "Error binding socket" << std::endl;
        close(serverSocket);
        return -1;
    }

    // 监听连接
    if (listen(serverSocket, 1) == -1) 
    {
        std::cerr << "Error listening" << std::endl;
        close(serverSocket);
        return -1;
    }
    std::cout << "Waiting for robot's capture signal..." << std::endl;

    while (true) 
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

        // // 接受连接
        struct sockaddr_in clientAddress;
        socklen_t clientAddrLength = sizeof(clientAddress);
        int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddress, &clientAddrLength);
        // int clientSocket;
        // if (clientSocket == -1) 
        // {
        //     std::cerr << "Error accepting connection" << std::endl;
        //     continue;
        // }

        // // 接收数据
        char buffer[1];
        // if (recv(clientSocket, buffer, sizeof(buffer), 0) == -1) 
        // {
        //     std::cerr << "Error receiving data" << std::endl;
        //     close(clientSocket);
        //     continue;
        // }

        // // 判断是否为拍照信号
        // std::cout << "数据" << buffer[0] << std::endl;
        // if (buffer[0] == 'r') 
        if (clientSocket) 
        {
            // 进行拍照操作并获得拍照结果，这里模拟返回结果
            std::string poseResult;

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
            if (img.empty()) 
            {
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
            if (img.empty()) 
            {
                std::cout << "Error: Unable to read image at path '" << colorFile << "'" << std::endl;
                return -1;
            }

            // std::cout << "选择要识别的物体：" << std::endl;
            // std::cout << "1. 大药盒" << std::endl;
            // std::cout << "2. 中药盒" << std::endl;
            // std::cout << "3. 小药盒" << std::endl;
            int choice = 2;
            // std::cout << "请输入选项的编号：";
            // std::cin >> choice;

            //  导入点云模板
            std::string icpPlyName;
            switch (choice) 
            {
                case 1:
                    //  大药盒点云模板
                    icpPlyName = "../models/plyModel/model/large/model_large.ply";
                    break;
                case 2:
                    //  中药盒点云模板
                    icpPlyName = "../models/plyModel/model/middle/model_middle.ply";
                    break;
                case 3:
                    //  小药盒点云模板
                    icpPlyName = "../models/plyModel/model/small/model_small.ply";
                    break;
                default:
                    break;
            }

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

                    /*                  求解平面中心点坐标和法向量        */
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
                    normalEstimation.setRadiusSearch(0.03); // 根据点云密度调整搜索半径
                    // 计算法线估计并存储到normals中
                    normalEstimation.compute(*normals);

                    // 创建欧几里德聚类提取对象
                    pcl::EuclideanClusterExtraction<pcl::PointXYZ> clusterExtractor;
                    clusterExtractor.setClusterTolerance(3.0);  // 设置聚类的欧几里德距离阈值
                    clusterExtractor.setMinClusterSize(800);     // 设置最小聚类点数
                    clusterExtractor.setMaxClusterSize(3000000);    // 设置最大聚类点数
                    clusterExtractor.setInputCloud(segCloud);    // 设置输入点云数据

                    // 存储提取的聚类索引
                    std::vector<pcl::PointIndices> clusters;
                    // 执行聚类提取
                    clusterExtractor.extract(clusters);

                    // 粗略配准
                    pcl::PointCloud<pcl::PointXYZ>::Ptr icpCloud(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::io::loadPLYFile<pcl::PointXYZ>(icpPlyName, *icpCloud);

                    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
                    icp.setInputSource(icpCloud); // 定义源点云
                    icp.setMaximumIterations(50);
                    
                    // 精细配准
                    pcl::PointCloud<pcl::PointXYZ>::Ptr icpFineCloud(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::io::loadPLYFile<pcl::PointXYZ>(icpPlyName, *icpFineCloud);

                    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icpFine;
                    icpFine.setInputSource(icpFineCloud); // 定义源点云
                    icpFine.setMaximumIterations(100);

                    for (const auto& indices : clusters) 
                    {
                        pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZ>);
                        pcl::ExtractIndices<pcl::PointXYZ> extract;

                        extract.setInputCloud(segCloud);
                        extract.setIndices(boost::make_shared<const pcl::PointIndices>(indices));
                        extract.filter(*clusterCloud);

                        // 粗略配准
                        icp.setInputTarget(clusterCloud); // 设置目标点云
                        pcl::PointCloud<pcl::PointXYZ> alignedCloudCoarse;
                        icp.align(alignedCloudCoarse);

                        // 精细配准
                        icpFine.setInputTarget(alignedCloudCoarse.makeShared()); // 设置目标点云
                        pcl::PointCloud<pcl::PointXYZ> alignedCloudFine;
                        icpFine.align(alignedCloudFine);

                        // 获取配准后点云的质心和姿态矩阵
                        pcl::CentroidPoint<pcl::PointXYZ> centroid;
                        for (const auto& point : alignedCloudFine.points)
                        {
                            centroid.add(point);
                        }
                        pcl::PointXYZ centroidPoint;
                        centroid.get(centroidPoint);

                        // 计算法向量
                        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
                        normalEstimation.setInputCloud(alignedCloudFine.makeShared());
                        normalEstimation.setSearchMethod(tree);
                        normalEstimation.setRadiusSearch(0.03); // 根据点云密度调整搜索半径
                        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
                        normalEstimation.compute(*normals);

                        // 从ICP变换获取姿态矩阵
                        Eigen::Matrix4f pose = icpFine.getFinalTransformation();

                        // 打印质心和姿态矩阵
                        std::cout << "质心坐标: " << centroidPoint.x << ", " << centroidPoint.y << ", " << centroidPoint.z << std::endl;
                        std::cout << "法向量: " << normals->at(0).normal_x << ", " << normals->at(0).normal_y << ", " << normals->at(0).normal_z << std::endl;
                        std::cout << "姿态矩阵：" << std::endl << pose << std::endl;

                        //     /*                  可视化抓取点        */
                        // 创建点云可视化
                        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("物体抓取点可视界面"));

                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                        pcl::PLYReader reader;
                        reader.read("pointCloudColor.ply", *colorCloud);

                        // 可视化点云
                        viewer->addPointCloud(colorCloud);

                        // Define coordinate system colors
                        Eigen::Vector3i colors[3];
                        colors[0] = Eigen::Vector3i(255, 0, 0); // X轴为红色
                        colors[1] = Eigen::Vector3i(0, 255, 0); // Y轴为绿色
                        colors[2] = Eigen::Vector3i(0, 0, 255); // Z轴为蓝色

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
                            transform.translation() <<  centroidPoint.x,  centroidPoint.y, centroidPoint.z;

                            // 创建两个点 p1 和 p2 用于绘制坐标轴线段
                            pcl::PointXYZ p1, p2;
                            // 将 p1 设置为质心位置
                            p1.getVector3fMap() = centroidPoint.getVector3fMap();
                            // 将 p2 设置为质心位置加上坐标轴方向的一点，构成线段
                            p2.getVector3fMap() = centroidPoint.getVector3fMap() + axis * 0.1;
                            // 在可视化器中添加线段，用于绘制坐标轴
                            viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(p1, p2, colors[i][0], colors[i][1], colors[i][2], "axis_" + std::to_string(i), 0);
                        }

                        /*                  求解物体相对于机器人基座位姿      */
                        //  法向量转四元数
                        Eigen::Vector3f normal(normal.x(), normal.y(), normal.z()); 
                        // 计算旋转矩阵
                        Eigen::Vector3f up(0, 0, 1); // 参考向上的向量
                        Eigen::Vector3f axis = normal.cross(up);
                        float angle = std::acos(normal.dot(up));
                        Eigen::Matrix3f rotation;
                        rotation = Eigen::AngleAxisf(angle, axis);
                        // 构造旋转四元数(在Eigen库中，四元数的构造方式默认使用的是XYZW的顺序)
                        Eigen::Quaternionf quaternion;
                        quaternion = Eigen::AngleAxisf(angle, axis);
                        double qw = quaternion.w();
                        double qx = quaternion.x();
                        double qy = quaternion.y();
                        double qz = quaternion.z();

                        //  相机相对于机器人基座的齐次变换矩阵
                        Eigen::Matrix4d transform_cam_to_base = transformMatrixFromPose(-0.329298, 1.03579, 1.15312, 0.0496425, 0.00719909, 0.998676, -0.0113733);
                        //  识别到物体相对于相机的齐次变换矩阵
                        Eigen::Matrix4d transform_obj_to_cam = transformMatrixFromPose(centroidPoint.x, centroidPoint.y, centroidPoint.z, qw, qx, qy, qz);

                        // 计算物体相对于机器人基座的位姿
                        Eigen::Matrix4d result = transform_cam_to_base * transform_obj_to_cam.matrix();
                        result(2,3) = result(2, 3) + 0.178; // 增加末端执行器的长度

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
                        std::cout << "Translation vector: " << translation.transpose() * 1000 << std::endl;
                        std::cout << "ZYZ Euler angles (phi, theta, psi): " << robot_euler_angles.transpose() << std::endl;
                        
                        std::ostringstream pose_x, pose_y, pose_z, pose_rr, pose_rp, pose_ry;
                        pose_x << std::fixed << std::setprecision(2) << translation(0)*1000;
                        pose_y << std::fixed << std::setprecision(2) << translation(1)*1000;
                        pose_z << std::fixed << std::setprecision(2) << translation(2)*1000;
                        pose_rr << std::fixed << std::setprecision(2) << robot_euler_angles(0);
                        pose_rp << std::fixed << std::setprecision(2) << robot_euler_angles(1);
                        pose_ry << std::fixed << std::setprecision(2) << robot_euler_angles(2);

                        //  提取结果(发送数据格式存在问题)
                        poseResult = pose_x.str() + "," + pose_y.str() + "," + pose_z.str() + "," + pose_rr.str() + "," + pose_rp.str() + "," + pose_ry.str() + "," + std::to_string(1);

                        std::cout << "发送结果为: " << poseResult << std::endl;

                        // 发送结果
                        if (send(clientSocket, poseResult.c_str(), poseResult.size(), 0) == -1) 
                        {
                            std::cerr << "Error sending data" << std::endl;
                        }
                        else
                        {
                            std::cerr << "Sending data success" << std::endl;
                            std::cerr << "Sending data success" <<  poseResult.c_str() << std::endl;
                            std::cerr << "Sending data success" <<  poseResult.size() << std::endl;
                            send(clientSocket, poseResult.c_str(), poseResult.size(), 0);
                        }

                        // Spin the viewer
                        viewer->spin();
                    }
                }
                else
                {
                    //  发送结果
                    
                    poseResult = std::to_string(0.0) + "," + std::to_string(0.0) + "," + std::to_string(0.0) + "," + std::to_string(0.0) + "," + std::to_string(0.0) + "," + std::to_string(0.0) + "," + "2";

                    std::cout << "发送结果为: " << poseResult << std::endl;
                                    // 发送结果
                    if (send(clientSocket, poseResult.c_str(), poseResult.size(), 0) == -1) 
                    {
                        std::cerr << "Error sending data" << std::endl;
                    }
                    else
                    {
                        std::cerr << "Sending data success" << std::endl;
                    }

                }


                count ++;

                }

        }

        // 关闭客户端套接字
        close(clientSocket);
    }

    // 关闭服务器套接字
    close(serverSocket);   


    return 0;
}