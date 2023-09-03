cmake_minimum_required(VERSION 3.10)
project(ShapeTempMatchExample)

set(CMAKE_CXX_STANDARD 11)

# 设置HALCON的安装路径，根据你的实际安装路径进行修改
#set(HALCON_INSTALL_DIR "/opt/MVTec/HALCON-23.05-Progress")

# 设置可执行文件输出目录为上一级目录的bin文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# 添加HALCON的头文件目录和库文件目录
include_directories(
    $ENV{HALCONROOT}/include
    $ENV{HALCONROOT}/include/halconcpp
    )
link_directories($ENV{HALCONROOT}/lib/x64-linux)

# 添加编译目标
add_executable(ShapeTempMatchExample src/ShapeTempMatchExample_demo.cpp)

# 链接HALCON库
target_link_libraries(ShapeTempMatchExample halconcpp)

# 添加编译目标
add_executable(ShapeTempMatchDemo src/main.cpp)

# 链接HALCON库
target_link_libraries(ShapeTempMatchDemo halconcpp)

