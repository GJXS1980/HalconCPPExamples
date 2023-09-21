#pragma once
#include "engine.h"
#include <fstream>

// 检查文件是否存在
inline bool doesFileExist (const std::string& name) 
{
    std::ifstream f(name.c_str());
    return f.good();
}

//  构建物体结构体
struct Object 
{
    // 物体的类
    int label{};
    // 置信度
    float probability{};
    // The object bounding box rectangle.
    cv::Rect_<float> rect;
    // 分割掩膜
    cv::Mat boxMask;
    // 骨骼特征点
    std::vector<float> kps{};
};

// Config the behavior of the YoloV8 detector.
// Can pass these arguments as command line parameters.
// yolov8结构体
struct YoloV8Config 
{
    // The precision to be used for inference
    Precision precision = Precision::FP16;
    // Calibration data directory. Must be specified when using INT8 precision.
    std::string calibrationDataDirectory;
    // 检测物体的置信度
    float probabilityThreshold = 0.8f;
    // Non-maximum suppression threshold
    float nmsThreshold = 0.4f;
    // 最大检测物体数量
    int topK = 100;
    // 分割选项
    int segChannels = 32;
    int segH = 160;
    int segW = 160;
    float segmentationThreshold = 0.5f;
    // 姿态估计选项
    int numKPS = 17;
    float kpsThreshold = 0.5f;
    // 检测类
    std::vector<std::string> classNames = {
        "ties", "bottle"
    };
};

//  boxes yolov8类
class YoloV8 
{
public:
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    YoloV8(const std::string& onnxModelPath, const YoloV8Config& config);

    // 在图像中检测物体
    std::vector<Object> detectObjects(const cv::Mat& inputImageBGR);
    std::vector<Object> detectObjects(const cv::cuda::GpuMat& inputImageBGR);

    // 在图像中画出物体的框和标签
    void drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, unsigned int scale = 2);
private:
    // Preprocess the input
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat& gpuImg);

    // Postprocess the output
    std::vector<Object> postprocessDetect(std::vector<float>& featureVector);

    // Postprocess the output for pose model
    std::vector<Object> postprocessPose(std::vector<float>& featureVector);

    // Postprocess the output for segmentation model
    std::vector<Object> postProcessSegmentation(std::vector<std::vector<float>>& featureVectors);
    std::unique_ptr<Engine> m_trtEngine = nullptr;

    // Used for image preprocessing
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS {0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS {1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;

    // Filter thresholds
    const float PROBABILITY_THRESHOLD;
    const float NMS_THRESHOLD;
    const int TOP_K;

    // Segmentation constants
    const int SEG_CHANNELS;
    const int SEG_H;
    const int SEG_W;
    const float SEGMENTATION_THRESHOLD;

    // Object classes as strings
    const std::vector<std::string> CLASS_NAMES;

    // Pose estimation constant
    const int NUM_KPS;
    const float KPS_THRESHOLD;

    // 识别物体框的颜色
    const std::vector<std::vector<float>> COLOR_LIST = 
    {
            {0.098, 0.325, 0.850},
            {0.125, 0.694, 0.929},
            {0.556, 0.184, 0.494},
            {0.188, 0.674, 0.466}
    };

    const std::vector<std::vector<unsigned int>> KPS_COLORS = 
    {
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255}
    };

    const std::vector<std::vector<unsigned int>> SKELETON = 
    {
            {16, 14},
            {14, 12},
            {17, 15},
            {15, 13},
            {12, 13},
            {6, 12},
            {7, 13},
            {6, 7},
            {6, 8},
            {7, 9},
            {8, 10},
            {9, 11},
            {2, 3},
            {1, 2},
            {1, 3},
            {2, 4},
            {3, 5},
            {4, 6},
            {5, 7}
    };

    const std::vector<std::vector<unsigned int>> LIMB_COLORS = 
    {
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {255, 51, 255},
            {255, 51, 255},
            {255, 51, 255},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0}
    };
};