#include <iostream>
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


int main() 
{
    //  相机相对于机器人基座的齐次变换矩阵
    Eigen::Matrix4d transform_cam_to_base = transformMatrixFromPose(-0.329298, 1.03579, 1.15312, 0.0496425, 0.00719909, 0.998676, -0.0113733);
    //  识别到物体相对于相机的齐次变换矩阵
    Eigen::Matrix4d transform_obj_to_cam = transformMatrixFromPose(-0.240, 0.077, 1.395, 0.0496425, 0.00719909, 0.998676, -0.0113733);


    // 计算物体相对于机器人基座的位姿
    Eigen::Matrix4d result = transform_cam_to_base * transform_obj_to_cam.matrix();
    result(2,3) = result(2, 3) + 0.169; // 末端执行器的长度

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
