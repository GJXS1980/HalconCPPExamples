cmake_minimum_required(VERSION 3.10)
project(KawasakiDemo)

# 寻找Eigen库
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_STANDARD 11)

# 设置可执行文件输出目录为上一级目录的bin文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)


# 添加可执行文件
add_executable(kawasaki_transform src/kawasaki_transform.cpp)

# 链接Eigen库
target_link_libraries(kawasaki_transform Eigen3::Eigen)


