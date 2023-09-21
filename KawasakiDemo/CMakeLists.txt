cmake_minimum_required(VERSION 3.10)
project(KawasakiDemo)

# 寻找Eigen库
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_STANDARD 17)

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
                    ${PCL_INCLUDE_DIRS}
                    ${CMAKE_SOURCE_DIR}/include
)
link_directories(${MECHEYEAPI_LIBRARY_DIRS} 
                ${OpenCV_LIBRARY_DIRS}
                ${PCL_INCLUDE_DIRS}
)


# TensorRT is found and linked by the tensorrt-cpp-api
add_subdirectory(libs/tensorrt-cpp-api)

# 设置可执行文件输出目录为上一级目录的bin文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# 添加可执行文件
add_executable(kawasaki_transform src/kawasaki_transform.cpp)
# 链接Eigen库
target_link_libraries(kawasaki_transform Eigen3::Eigen)

add_executable(KawasakiMecheyeDemo src/kawasaki_mecheye_demo.cpp)
target_link_libraries(KawasakiMecheyeDemo ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# 梅卡相机获取彩色图及点云
add_executable(CaptureDepthMap src/CaptureDepthMap.cpp)
target_link_libraries(CaptureDepthMap ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# Build the YoloV8  library
add_library(YoloV8_TRT SHARED src/yoloCfg/yolov8.cpp)
target_link_libraries(YoloV8_TRT PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(YoloV8_TRT PUBLIC libs/tensorrt-cpp-api/src)


# 构建药盒YoloV8检测库
add_library(boxesYoloDemo SHARED src/yoloCfg/boxesYolov8.cpp)
target_link_libraries(boxesYoloDemo PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(boxesYoloDemo PUBLIC libs/tensorrt-cpp-api/src)

# 构建角落YoloV8检测库
add_library(cornerYoloDemo SHARED src/yoloCfg/cornerYolov8.cpp)
target_link_libraries(cornerYoloDemo PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(cornerYoloDemo PUBLIC libs/tensorrt-cpp-api/src)

# 构建捆扎带和啤酒瓶YoloV8检测库
add_library(beerYoloDemo SHARED src/yoloCfg/beerBottleYolov8.cpp)
target_link_libraries(beerYoloDemo PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(beerYoloDemo PUBLIC libs/tensorrt-cpp-api/src)

# 啤酒瓶检测（实时检测）
add_executable(MechEyeDetectBeer src/beerbottle/mecheye_beerBottle_Point_Cloud_Registration.cpp)
target_link_libraries(MechEyeDetectBeer beerYoloDemo ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# 啤酒瓶点云分割
add_executable(MechEyeDetectBeerDemo src/beerbottle/beerBottle_Point_Cloud_Seg.cpp)
target_link_libraries(MechEyeDetectBeerDemo beerYoloDemo ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# 啤酒瓶和捆扎带检测（图片）
add_executable(MechEyeDetectBeerImgDemo src/beerbottle/beerBottle_Point_Cloud_Seg_yolo.cpp)
target_link_libraries(MechEyeDetectBeerImgDemo beerYoloDemo ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# Build and link the executables
add_executable(detect_object_image src/object_detection_image.cpp)
target_link_libraries(detect_object_image YoloV8_TRT)

# 药盒检测（图像检测）
add_executable(detect_object_imageBoxes src/boxes/object_detection_imageBoxes.cpp)
target_link_libraries(detect_object_imageBoxes boxesYoloDemo)

# 角点检测（图像检测）
add_executable(detect_object_imageCorner src/corner/object_detection_imageCorner.cpp)
target_link_libraries(detect_object_imageCorner cornerYoloDemo)

# 捆扎带检测（图像检测）
# add_executable(detect_object_imageStrap src/strap/object_detection_imageStrap.cpp)
# target_link_libraries(detect_object_imageStrap strapYoloDemo)

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark YoloV8_TRT)

# 视频流检测药盒
add_executable(detect_object_video src/boxes/object_detection_video_stream.cpp)
target_link_libraries(detect_object_video YoloV8_TRT)

# Build and link the executables
add_executable(MechEyeDetect src/boxes/mecheye_boxes_detection_compute3DCentroid.cpp)
target_link_libraries(MechEyeDetect YoloV8_TRT ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# 药盒检测（实时检测）
add_executable(MechEyeDetectBox src/boxes/mecheye_object_detectionBoxes.cpp)
target_link_libraries(MechEyeDetectBox boxesYoloDemo ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# 角点检测（实时检测）
add_executable(MechEyeDetectCorner src/corner/mecheye_object_detectionCorner.cpp)
target_link_libraries(MechEyeDetectCorner cornerYoloDemo ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# 捆扎带检测（实时检测）
# add_executable(MechEyeDetectStrap src/strap/mecheye_object_detectionStrap.cpp)
# target_link_libraries(MechEyeDetectStrap strapYoloDemo ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# EuclideanCluster点云聚类
add_executable(MechEyeDetectEuclideanCluster src/boxes/mecheye_boxes_detection_EuclideanCluster.cpp)
target_link_libraries(MechEyeDetectEuclideanCluster YoloV8_TRT ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# Registration点云聚类
add_executable(MechEyeDetectRegistration src/boxes/mecheye_boxes_Point_Cloud_Registration.cpp)
target_link_libraries(MechEyeDetectRegistration YoloV8_TRT ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# Registration点云聚类（图像流）
add_executable(MechEyeDetectRegistrationVideo src/boxes/mecheye_boxes_Point_Cloud_Registration_Video.cpp)
target_link_libraries(MechEyeDetectRegistrationVideo YoloV8_TRT ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# Registration点云聚类（图像流+机械臂TCP）
add_executable(MechEyeDetectRobot src/boxes/mecheye_boxes_Point_Cloud_Registration_TCP.cpp)
target_link_libraries(MechEyeDetectRobot YoloV8_TRT ${MECHEYEAPI_LIBRARIES} ${OpenCV_LIBS}  ${PCL_LIBRARIES})

# hsv颜色识别捆扎带
add_executable(cameraHSVDemo src/cameraHSVDemo.cpp)
target_link_libraries(cameraHSVDemo  ${OpenCV_LIBS} ${MECHEYEAPI_LIBRARIES})

# 测试程序
add_executable(testDemo src/test.cpp)
target_link_libraries(testDemo  ${OpenCV_LIBS})

# hsv颜色识别及点云分割捆扎带
add_executable(mecheyeHSVPCLDemo src/mecheyeHSVPCLDemo.cpp)
target_link_libraries(mecheyeHSVPCLDemo  ${OpenCV_LIBS} ${MECHEYEAPI_LIBRARIES}  ${PCL_LIBRARIES})

# 实例分割及点云分割捆扎带
add_executable(mecheyePCLCornerDemo src/corner/mecheyePCLCornerDemo.cpp)
target_link_libraries(mecheyePCLCornerDemo  ${OpenCV_LIBS} ${MECHEYEAPI_LIBRARIES}  ${PCL_LIBRARIES})

