cmake_minimum_required(VERSION 3.10)
project(KawasakiDemo)

# 寻找Eigen库
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_STANDARD 11)

# 设置HALCON的安装路径，根据你的实际安装路径进行修改
set(HALCON_INSTALL_DIR "/opt/MVTec/HALCON-23.05-Progress")

set(Files ${PROJECT_SOURCE_DIR}/src/PCLDemo.cpp)


if(CMAKE_HOST_WIN32)
  find_package(MechEyeApi REQUIRED CONFIG PATHS "$ENV{MECHEYE_DIR}/API")
  if(NOT MechEyeApi_FOUND)
    message(
      FATAL_ERROR "MechEyeApi not found. Please install MechEyeApi first.")
  endif()
elseif(CMAKE_HOST_UNIX)
  find_package(PkgConfig)
  if(NOT PkgConfig_FOUND)
    message(FATAL_ERROR "PkgConfig not found.")
  else()
    pkg_check_modules(MECHEYEAPI REQUIRED MechEyeApi)
    if(NOT MECHEYEAPI_FOUND)
      message(
        FATAL_ERROR "MechEyeApi not found. Please install MechEyeApi first.")
    endif()
  endif()
endif()

if(CMAKE_HOST_WIN32)
  if(MSVC_VERSION LESS 1910)
    message(FATAL_ERROR "Please use Visual Studio 2017 or higher.")
  elseif(MSVC_VERSION LESS 1920)
    find_package(PCL 1.8.1...1.9.1 REQUIRED)
  elseif(MSVC_VERSION LESS 1930)
    find_package(PCL 1.8.1...1.12.1 REQUIRED)
  else()
    find_package(PCL 1.8.1 REQUIRED)
  endif()
elseif(CMAKE_HOST_UNIX)
  find_package(PCL REQUIRED)
endif()
if(NOT PCL_FOUND)
  message(
    FATAL_ERROR
      "Point Cloud Library (PCL) not found. Please point PCL_DIR to the directory of your PCL installation (containing the file PCLConfig.cmake)."
  )
endif()

# OpenCV_DIR: set as your OpenCV libraries directory; Uncomment next line to set
# OpenCV_DIR manually

# set(OpenCV_DIR  "path to OpenCV directory")
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(
    FATAL_ERROR
      "OpenCV not found. Please point OpenCV_DIR to the directory of your OpenCV installation (containing the file OpenCVConfig.cmake)."
  )
endif()

include_directories(${MECHEYEAPI_INCLUDE_DIRS} 
                    ${OpenCV_INCLUDE_DIRS}
                    ${HALCON_INSTALL_DIR}/include
                    ${HALCON_INSTALL_DIR}/include/halconcpp
                    ${PCL_INCLUDE_DIRS}

)
link_directories(${MECHEYEAPI_LIBRARY_DIRS} 
                ${OpenCV_LIBRARY_DIRS}
                ${HALCON_INSTALL_DIR}/lib/x64-linux
                ${PCL_INCLUDE_DIRS}
                )



# 设置可执行文件输出目录为上一级目录的bin文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)


# 添加可执行文件
add_executable(kawasaki_transform src/kawasaki_transform.cpp)

# 链接Eigen库
target_link_libraries(kawasaki_transform Eigen3::Eigen)

add_executable(KawasakiMecheyeDemo src/kawasaki_mecheye_demo.cpp)
target_link_libraries(KawasakiMecheyeDemo ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS} halconcpp ${PCL_LIBRARIES})

add_executable(CaptureDepthMap src/CaptureDepthMap.cpp)
target_link_libraries(CaptureDepthMap ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS} halconcpp ${PCL_LIBRARIES})




