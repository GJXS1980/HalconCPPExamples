cmake_minimum_required(VERSION 3.10)
project(MecheyeConCameraACaptureImages_demo)

set(CMAKE_CXX_STANDARD 14)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(FATAL_ERROR "This sample is not valid in ARM platform.")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

# 设置可执行文件输出目录为上一级目录的bin文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Halcon/cmake;${CMAKE_MODULE_PATH}")
find_package(Halcon REQUIRED)
find_package(HDevEngine REQUIRED)
if(NOT HALCON_FOUND)
  message(FATAL_ERROR "Halcon not found.")
endif()

# 添加HALCON的头文件目录和库文件目录
include_directories(${HALCON_INC_DIRS})
link_directories(${HALCON_EXT_LIB_DIR})

# 添加编译目标
add_executable(ConCameraACaptureImages_demo src/main.cpp src/camera_parameters.cpp src/dev_update_off.cpp src/disp_message.cpp src/set_display_font.cpp src/visualize_object_model_3d.cpp)

# 链接HALCON库
target_link_libraries(ConCameraACaptureImages_demo ${HALCON_LIBS} ${HALCON_LIBS_XL})