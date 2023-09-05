cmake_minimum_required(VERSION 3.5)

project(HandeyeDemo)

set(CMAKE_CXX_STANDARD 14)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(FATAL_ERROR "This sample is not valid in ARM platform.")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Halcon/cmake;${CMAKE_MODULE_PATH}")
find_package(Halcon REQUIRED)
find_package(HDevEngine REQUIRED)
if(NOT HALCON_FOUND)
  message(FATAL_ERROR "Halcon not found.")
endif()

# 添加HALCON的头文件目录和库文件目录
include_directories( ${HALCON_INC_DIRS})
link_directories(${HALCON_EXT_LIB_DIR})

# 添加编译目标
add_executable(
    HandToEyeDemo 
    src/disp_continue_message.cpp 
    src/disp_message.cpp 
    src/main.cpp
    src/set_display_font.cpp
    )

# 链接HALCON库
target_link_libraries(HandToEyeDemo ${HALCON_LIBS} ${HALCON_LIBS_XL})

# 添加编译目标
add_executable(
    GenCaltabDemo 
    src/GenCaltabDemo.cpp 
    )

# 链接HALCON库
target_link_libraries(GenCaltabDemo ${HALCON_LIBS} ${HALCON_LIBS_XL})


