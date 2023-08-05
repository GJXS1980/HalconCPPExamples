cmake_minimum_required(VERSION 3.10)
project(HandeyeDemo)

set(CMAKE_CXX_STANDARD 11)

# 设置HALCON的安装路径，根据你的实际安装路径进行修改
set(HALCON_INSTALL_DIR "/opt/MVTec/HALCON-23.05-Progress")

# 设置可执行文件输出目录为上一级目录的bin文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# 添加HALCON的头文件目录和库文件目录
include_directories(
    ${HALCON_INSTALL_DIR}/include
    ${HALCON_INSTALL_DIR}/include/halconcpp
    )
link_directories(${HALCON_INSTALL_DIR}/lib/x64-linux)

# 添加编译目标
add_executable(
    HandToEyeDemo 
    src/disp_continue_message.cpp 
    src/disp_message.cpp 
    src/main.cpp
    src/set_display_font.cpp
    )

# 链接HALCON库
target_link_libraries(HandToEyeDemo halconcpp)

# 添加编译目标
add_executable(
    GenCaltabDemo 
    src/GenCaltabDemo.cpp 
    )

# 链接HALCON库
target_link_libraries(GenCaltabDemo halconcpp)


