cmake_minimum_required(VERSION 3.10)
project(ShapeTempMatchExample)

set(CMAKE_CXX_STANDARD 11)

# 设置HALCON的安装路径，根据你的实际安装路径进行修改
set(HALCON_INSTALL_DIR "/opt/MVTec/HALCON-23.05-Progress")

# 添加HALCON的头文件目录和库文件目录
include_directories(${HALCON_INSTALL_DIR}/include)
link_directories(${HALCON_INSTALL_DIR}/lib/x64-linux)

# 添加编译目标
add_executable(ShapeTempMatchExample ShapeTempMatchExample_demo.cpp)

# 链接HALCON库
target_link_libraries(ShapeTempMatchExample halconcpp)
