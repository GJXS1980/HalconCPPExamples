/*******************************************************************************
 *BSD 3-Clause License
 *
 *Copyright (c) 2016-2023, Mech-Mind Robotics
 *All rights reserved.
 *
 *Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *
 *1. Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 *2. Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 *3. Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 *THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

/*
With this sample program, you can obtain and save the depth map in OpenCV format from a camera.
*/

#include <iostream>
#include "MechEyeApi.h"
#include "SampleUtil.h"
#include "OpenCVUtil.h"

#include "HalconCpp.h"
#include "HDevThread.h"
#include <opencv2/opencv.hpp>

#include "PclUtil.h"
#include <pcl/io/ply_io.h>

using namespace HalconCpp;


int main()
{
    mmind::api::MechEyeDevice device;
    if (!findAndConnect(device))
        return -1;

    //  采集彩色点云数据
    mmind::api::PointXYZBGRMap pointXYZBGRMap;
    showError(device.capturePointXYZBGRMap(pointXYZBGRMap));
    const std::string pointCloudColorPath = "pointCloudColor.ply";
    savePLY(pointXYZBGRMap, pointCloudColorPath);

    // //  将彩色点云转成640*480的点云
    // // Load PLY point cloud
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::io::loadPLYFile("pointCloudColor.ply", *cloud);

    // int newWidth = 640;
    // int newHeight = 480;

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr resizedCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // resizedCloud->width = newWidth;
    // resizedCloud->height = newHeight;
    // resizedCloud->is_dense = false;
    // resizedCloud->points.resize(newWidth * newHeight);

    // float widthScale = static_cast<float>(cloud->width) / newWidth;
    // float heightScale = static_cast<float>(cloud->height) / newHeight;

    // for (int y = 0; y < newHeight; y++) {
    //     for (int x = 0; x < newWidth; x++) {
    //         int origX = static_cast<int>(x * widthScale);
    //         int origY = static_cast<int>(y * heightScale);
    //         resizedCloud->at(x, y) = cloud->at(origX, origY);
    //     }
    // }

    // // Save the resized point cloud to PLY file
    // pcl::io::savePLYFileBinary("pointCloudColor.ply", *resizedCloud);


    //  采集1通道彩色图
    mmind::api::ColorMap color;
    showError(device.captureColorMap(color));
    const std::string colorFile = "boxes_01.png";
    saveMap(color, colorFile);
    // cv::Mat color8UC3 = cv::Mat(color.height(), color.width(), CV_8UC3, color.data());
    // cv::Mat grayImage;
    // cv::cvtColor(color8UC3, grayImage, cv::COLOR_BGR2GRAY);

    // // // 缩放图像到640x480尺寸
    // // cv::Mat resizedImage;
    // // cv::resize(grayImage, resizedImage, cv::Size(640, 480));

    // cv::imwrite(colorFile, grayImage);
    // std::cout << "Capture and save color image : " << colorFile << std::endl;


    // //  将彩色点云转成tif格式图像
    // // Load PLY point cloud
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tif(new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::io::loadPLYFile("pointCloudColor.ply", *cloud_tif);

    // // Create a 3-channel, 32-bit floating-point image
    // cv::Mat image(cloud_tif->height, cloud_tif->width, CV_32FC3);

    // // Find min and max values for scaling
    // float min_xyz = std::numeric_limits<float>::max();
    // float max_xyz = std::numeric_limits<float>::lowest();
    // for (int y = 0; y < image.rows; y++) {
    //     for (int x = 0; x < image.cols; x++) {
    //         pcl::PointXYZRGB& point = cloud_tif->at(x, y);
    //         float xyz_max = std::max({point.x, point.y, point.z});
    //         float xyz_min = std::min({point.x, point.y, point.z});
    //         min_xyz = std::min(min_xyz, xyz_min);
    //         max_xyz = std::max(max_xyz, xyz_max);
    //     }
    // }

    // // Fill the image data from the point cloud, normalize values to [0, 1]
    // for (int y = 0; y < image.rows; y++) {
    //     for (int x = 0; x < image.cols; x++) {
    //         pcl::PointXYZRGB& point = cloud_tif->at(x, y);
    //         float r = (point.x - min_xyz) / (max_xyz - min_xyz);
    //         float g = (point.y - min_xyz) / (max_xyz - min_xyz);
    //         float b = (point.z - min_xyz) / (max_xyz - min_xyz);
    //         image.at<cv::Vec3f>(y, x) = cv::Vec3f(r, g, b);
    //     }
    // }

    // // Save the image as TIFF
    // std::vector<int> compression_params;
    // compression_params.push_back(cv::IMWRITE_TIFF_COMPRESSION);
    // compression_params.push_back(cv::IMWRITE_TIFF_RESUNIT);
    // cv::imwrite("boxes_xyz_01.tif", image, compression_params);


    device.disconnect();
    std::cout << "Disconnected from the Mech-Eye device successfully." << std::endl;
    return 0;

}
