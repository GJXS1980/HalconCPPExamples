cmake_minimum_required(VERSION 3.10)
project(MecheyeConCameraACaptureImages_demo)

set(CMAKE_CXX_STANDARD 11)

# 设置HALCON的安装路径，根据你的实际安装路径进行修改
set(HALCON_INSTALL_DIR "/opt/MVTec/HALCON-23.05-Progress")

# 设置可执行文件输出目录为上一级目录的bin文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# 添加HALCON的头文件目录和库文件目录
include_directories(
    ${HALCON_INSTALL_DIR}/include/halconcpp
    ${HALCON_INSTALL_DIR}/include/

)
link_directories(${HALCON_INSTALL_DIR}/lib/x64-linux)

# 添加编译目标
add_executable(ConCameraACaptureImages_demo src/main.cpp src/camera_parameters.cpp src/dev_update_off.cpp src/disp_message.cpp src/set_display_font.cpp src/visualize_object_model_3d.cpp)

# 链接HALCON库
target_link_libraries(ConCameraACaptureImages_demo halconcpp)


