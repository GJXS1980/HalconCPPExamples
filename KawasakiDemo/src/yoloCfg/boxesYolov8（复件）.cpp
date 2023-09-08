#include <opencv2/cudaimgproc.hpp>
#include "boxesYolov8.h"

/**
 * @brief YoloV8 类的构造函数，用于根据给定的 ONNX 模型路径和配置初始化 YoloV8 实例
 * 
 * @param onnxModelPath ONNX 模型文件的路径
 * @param config YoloV8Config 配置对象，包含相关参数
 */
YoloV8::YoloV8(const std::string& onnxModelPath, const YoloV8Config& config)
        : PROBABILITY_THRESHOLD(config.probabilityThreshold)
        , NMS_THRESHOLD(config.nmsThreshold)
        , TOP_K(config.topK)
        , SEG_CHANNELS(config.segChannels)
        , SEG_H(config.segH)
        , SEG_W(config.segW)
        , SEGMENTATION_THRESHOLD(config.segmentationThreshold)
        , CLASS_NAMES(config.classNames)
        , NUM_KPS(config.numKPS)
        , KPS_THRESHOLD(config.kpsThreshold) 
{
    // 创建一个 Options 对象，用于设置 GPU 推理选项
    Options options;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;

    options.precision = config.precision;
    options.calibrationDataDirectoryPath = config.calibrationDataDirectory;

    // 如果选定的精度是 INT8，则必须提供 INT8 校准数据路径
    if (options.precision == Precision::INT8) 
    {
        if (options.calibrationDataDirectoryPath.empty()) 
        {
            throw std::runtime_error("Error: 必须提供 INT8 校准数据路径");
        }
    }

    // 创建 TensorRT 推理引擎
    m_trtEngine = std::make_unique<Engine>(options);

    // 将给定的 ONNX 模型构建为一个 TensorRT 引擎文件
    // 如果引擎文件已经存在，该函数会立即返回
    // 如果上述 Options 发生了变化，引擎文件会被重新构建
    auto succ = m_trtEngine->build(onnxModelPath, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!succ) 
    {
        const std::string errMsg = "Error: 无法构建 TensorRT 引擎。 "
                                   "请尝试将 TensorRT 日志级别增加到 kVERBOSE（位于 /libs/tensorrt-cpp-api/engine.cpp）。";
        throw std::runtime_error(errMsg);
    }

    // 加载 TensorRT 引擎文件中的网络权重
    succ = m_trtEngine->loadNetwork();
    if (!succ) 
    {
        throw std::runtime_error("Error: 无法将 TensorRT 引擎权重加载到内存中。");
    }
}


/**
 * @brief 对输入图像进行预处理，将其转换为模型输入所需的格式
 * 
 * @param gpuImg 输入的 GPU 图像
 * @return 包含预处理后的图像的嵌套 GPU 图像向量的向量
 */
std::vector<std::vector<cv::cuda::GpuMat>> YoloV8::preprocess(const cv::cuda::GpuMat &gpuImg) 
{
    // 获取输入张量的维度
    const auto& inputDims = m_trtEngine->getInputDims();

    // 将图像从 BGR 格式转换为 RGB 格式
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    auto resized = rgbMat;

    // 调整图像大小到模型所需的输入尺寸，同时保持纵横比并使用填充
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) 
    {
        // 如果尺寸不符合要求，进行调整大小以保持纵横比并填充
        resized = Engine::resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
    }

    // 转换为推理引擎所需的格式
    // 之所以使用奇怪的格式是因为它支持多输入模型以及批处理
    // 在我们的情况下，模型只有一个输入，并且我们使用批处理大小为 1。
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    std::vector<std::vector<cv::cuda::GpuMat>> inputs {std::move(input)};

    // 这些参数将在后处理阶段使用
    m_imgHeight = rgbMat.rows;
    m_imgWidth = rgbMat.cols;
    m_ratio =  1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));

    return inputs;
}

/**
 * @brief 对输入图像进行目标检测，返回检测到的目标信息
 * 
 * @param inputImageBGR 输入的 BGR 图像
 * @return 包含检测到的目标信息的对象向量
 */
std::vector<Object> YoloV8::detectObjects(const cv::cuda::GpuMat &inputImageBGR) 
{
    // 预处理输入图像
#ifdef ENABLE_BENCHMARKS
    static int numIts = 1;
    preciseStopwatch s1;
#endif
    const auto input = preprocess(inputImageBGR);
#ifdef ENABLE_BENCHMARKS
    static long long t1 = 0;
    t1 += s1.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Preprocess time: " << (t1 / numIts) / 1000.f << " ms" << std::endl;
#endif
    // 使用 TensorRT 引擎进行推理
#ifdef ENABLE_BENCHMARKS
    preciseStopwatch s2;
#endif
    std::vector<std::vector<std::vector<float>>> featureVectors;
    auto succ = m_trtEngine->runInference(input, featureVectors);
    if (!succ) {
        throw std::runtime_error("Error: Unable to run inference.");
    }
#ifdef ENABLE_BENCHMARKS
    static long long t2 = 0;
    t2 += s2.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Inference time: " << (t2 / numIts) / 1000.f << " ms" << std::endl;
    preciseStopwatch s3;
#endif
    // 检查模型是否仅支持目标检测或还支持分割
    std::vector<Object> ret;
    const auto& numOutputs = m_trtEngine->getOutputDims().size();
    if (numOutputs == 1) {
        // 目标检测或姿态估计
        // 由于批处理大小为 1，且只有一个输出，我们需要将输出从 3D 数组转换为 1D 数组。 
        std::vector<float> featureVector;
        Engine::transformOutput(featureVectors, featureVector);

        const auto& outputDims = m_trtEngine->getOutputDims();
        int numChannels = outputDims[outputDims.size() - 1].d[1];
        // TODO: 需要改进以使其更通用（不要使用魔术数）
        // 目前它适用于 Ultralytics 预训练模型。
        if (numChannels == 56) 
        {
            // 姿态估计
            ret = postprocessPose(featureVector);
        } 
        else 
        {
            // 目标检测
            ret = postprocessDetect(featureVector);
        }
    } 
    else 
    {
        // 分割
        // 由于批处理大小为 1，且有 2 个输出，我们需要将输出从 3D 数组转换为 2D 数组。
        std::vector<std::vector<float>> featureVector;
        Engine::transformOutput(featureVectors, featureVector);
        ret = postProcessSegmentation(featureVector);
    }
#ifdef ENABLE_BENCHMARKS
    static long long t3 = 0;
    t3 +=  s3.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Postprocess time: " << (t3 / numIts++) / 1000.f << " ms\n" << std::endl;
#endif
    return ret;
}


/**
 * @brief 在 YoloV8 类中执行目标检测，返回检测到的目标信息。
 * 
 * @param inputImageBGR 输入的 BGR 图像
 * @return 包含检测到的目标信息的对象向量
 */
std::vector<Object> YoloV8::detectObjects(const cv::Mat &inputImageBGR) 
{
    // 将图像上传到 GPU 内存
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);

    // 使用 GPU 图像调用 detectObjects 函数
    return detectObjects(gpuImg);
}


/**
 * @brief 对分割模型的输出进行后处理，提取分割目标信息。
 * 
 * @param featureVectors 分割模型的输出特征向量
 * @return 包含分割目标信息的对象向量
 */

std::vector<Object> YoloV8::postProcessSegmentation(std::vector<std::vector<float>>& featureVectors) 
{
    // 获取输出维度信息
    const auto& outputDims = m_trtEngine->getOutputDims();

    int numChannels = outputDims[outputDims.size() - 1].d[1];
    int numAnchors = outputDims[outputDims.size() - 1].d[2];

    const auto numClasses = numChannels - SEG_CHANNELS - 4;

    // 确保输出长度正确
    if (featureVectors[0].size() != static_cast<size_t>(SEG_CHANNELS) * SEG_H * SEG_W) 
    {
        throw std::logic_error("第 0 个输出长度不正确");
    }

    if (featureVectors[1].size() != static_cast<size_t>(numChannels) * numAnchors) 
    {
        throw std::logic_error("第 1 个输出长度不正确");
    }

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVectors[1].data());
    output = output.t();

    cv::Mat protos = cv::Mat(SEG_CHANNELS, SEG_H * SEG_W, CV_32F, featureVectors[0].data());

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> maskConfs;
    std::vector<int> indices;

    // 对每个锚框进行处理，解析出边界框和类别标签
    for (int i = 0; i < numAnchors; i++) 
    {
        // 获取当前目标的输出行指针
        auto rowPtr = output.row(i).ptr<float>();
        
        auto bboxesPtr = rowPtr;
        // 从输出行数据中获取得分信息
        auto scoresPtr = rowPtr + 4;
        // 从输出行数据中获取分割掩膜信息
        auto maskConfsPtr = rowPtr + 4 + numClasses;
        // 找到得分信息中的最大值，确定目标的最可能类别
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        float score = *maxSPtr;
        // 如果最高得分大于阈值，则处理这个目标
        if (score > PROBABILITY_THRESHOLD) 
        {
            // 从输出行数据中获取边界框坐标
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            // 根据比例因子 m_ratio 调整后的边界框坐标
            float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);

            // 提取得分最高的类别作为目标的类别标签
            int label = maxSPtr - scoresPtr;
            // 构建边界框对象
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            cv::Mat maskConf = cv::Mat(1, SEG_CHANNELS, CV_32F, maskConfsPtr);
            // 存储目标的信息
            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
            maskConfs.push_back(maskConf);
        }
    }

    /* 非极大值抑制
    bboxes: 一个包含目标边界框的向量，每个边界框用一个 cv::Rect 对象表示。这些边界框通常是在检测过程中生成的。
    scores: 一个包含每个目标框对应的置信度分数的向量。这些分数通常是在检测过程中计算得到的。
    labels: 一个包含每个目标框对应的类别标签的向量。这些标签通常是在检测过程中确定的。
    threshold: 非极大值抑制的阈值。如果两个目标框的重叠区域大于等于该阈值，其中一个框将被抑制。
    nmsOverlapThreshold: NMS 的重叠阈值。如果两个目标框的重叠区域大于等于该阈值，其中一个框将被抑制。
    indices: 输出的向量，其中包含经过 NMS 处理后保留的目标框的索引。
    */
    cv::dnn::NMSBoxesBatched(
            bboxes,
            scores,
            labels,
            PROBABILITY_THRESHOLD,
            NMS_THRESHOLD,
            indices
    );

    // 提取分割掩膜
    cv::Mat masks;
    std::vector<Object> objs;
    int cnt = 0;
    for (auto& i : indices) 
    {
        // 判断是否达到最大处理目标数
        if (cnt >= TOP_K) 
        {
            break;
        }
        // 获取当前目标的边界框信息
        cv::Rect tmp = bboxes[i];
        // 创建一个 Object 结构体实例
        Object obj;
        // 将目标的类别标签赋值给 obj.label
        obj.label = labels[i];
        // 将边界框信息赋值给 obj.rect
        obj.rect = tmp;
        // 将目标的置信度（概率）赋值给 obj.probability
        obj.probability = scores[i];
        
        masks.push_back(maskConfs[i]);
        // 将当前目标的信息存储到 objs 向量中
        objs.push_back(obj);
         // 增加已处理目标数
        cnt += 1;
    }
    
    // std::cout << "掩膜Mask： " << masks << std::endl;

    // 将分割掩膜映射回原始图像
    if (!masks.empty()) 
    {
        // 计算分割掩膜在原始图像上的位置和大小
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(), { SEG_W, SEG_H });

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        const auto inputDims = m_trtEngine->getInputDims();

        cv::Rect roi;
        if (m_imgHeight > m_imgWidth) 
        {
            roi = cv::Rect(0, 0, SEG_W * m_imgWidth / m_imgHeight, SEG_H);
        } 
        else 
        {
            roi = cv::Rect(0, 0, SEG_W, SEG_H * m_imgHeight / m_imgWidth);
        }
        // std::cout << "ROI: " << roi << std::endl;

        // 将分割掩膜应用于目标边界框
        for (size_t i = 0; i < indices.size(); i++)
        {
            cv::Mat dest, mask;
            // 计算分割掩膜的中间结果 dest
            cv::exp(-maskChannels[i], dest); // 对 maskChannels[i] 中的每个像素应用指数函数
            dest = 1.0 / (1.0 + dest);   // 将结果进行缩放，使范围在 [0, 1]
            // 在指定的 ROI 区域内进行操作
            dest = dest(roi);   // 仅保留 ROI 区域的数据
            // 调整分割掩膜的大小，使其与原始图像大小匹配
            cv::resize(
                    dest,
                    mask,
                    cv::Size(static_cast<int>(m_imgWidth), static_cast<int>(m_imgHeight)),
                    cv::INTER_LINEAR
            );
            // 生成目标的二进制分割掩膜
            // 将 mask 中大于 SEGMENTATION_THRESHOLD 的像素设置为 1，其余像素设置为 0
            objs[i].boxMask = mask(objs[i].rect) > SEGMENTATION_THRESHOLD;
        }
    }
    return objs;
}


std::vector<Object> YoloV8::postprocessPose(std::vector<float> &featureVector) 
{
    const auto& outputDims = m_trtEngine->getOutputDims();
    auto numChannels = outputDims[0].d[1];
    auto numAnchors = outputDims[0].d[2];

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    std::vector<std::vector<float>> kpss;

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVector.data());
    output = output.t();

    // Get all the YOLO proposals
    for (int i = 0; i < numAnchors; i++) 
    {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto kps_ptr = rowPtr + 5;
        float score = *scoresPtr;
        if (score > PROBABILITY_THRESHOLD) 
        {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            std::vector<float> kps;
            for (int k = 0; k < NUM_KPS; k++) 
            {
                float kpsX = *(kps_ptr + 3 * k) * m_ratio;
                float kpsY = *(kps_ptr + 3 * k + 1) * m_ratio;
                float kpsS = *(kps_ptr + 3 * k + 2);
                kpsX       = std::clamp(kpsX, 0.f, m_imgWidth);
                kpsY       = std::clamp(kpsY, 0.f, m_imgHeight);
                kps.push_back(kpsX);
                kps.push_back(kpsY);
                kps.push_back(kpsS);
            }

            bboxes.push_back(bbox);
            labels.push_back(0); // All detected objects are people
            scores.push_back(score);
            kpss.push_back(kps);
        }
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<Object> objects;

    // Choose the top k detections
    int cnt = 0;
    for (auto& chosenIdx : indices) 
    {
        if (cnt >= TOP_K) 
        {
            break;
        }

        Object obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        obj.kps = kpss[chosenIdx];
        objects.push_back(obj);

        cnt += 1;
    }

    return objects;}


std::vector<Object> YoloV8::postprocessDetect(std::vector<float> &featureVector) 
{
    const auto& outputDims = m_trtEngine->getOutputDims();
    auto numChannels = outputDims[0].d[1];
    auto numAnchors = outputDims[0].d[2];

    auto numClasses = CLASS_NAMES.size();

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVector.data());
    output = output.t();

    // Get all the YOLO proposals
    for (int i = 0; i < numAnchors; i++) 
    {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        float score = *maxSPtr;
        if (score > PROBABILITY_THRESHOLD) 
        {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);

            int label = maxSPtr - scoresPtr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<Object> objects;

    // Choose the top k detections
    int cnt = 0;
    for (auto& chosenIdx : indices) 
    {
        if (cnt >= TOP_K) 
        {
            break;
        }

        Object obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        objects.push_back(obj);

        cnt += 1;
    }

    return objects;
}


void YoloV8::drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, unsigned int scale) 
{
    // If segmentation information is present, start with that
    if (!objects.empty() && !objects[0].boxMask.empty()) 
    {
        cv::Mat mask = image.clone();
        for (const auto& object: objects) 
        {
            // Choose the color
            int colorIndex = object.label % COLOR_LIST.size(); // We have only defined 80 unique colors
            cv::Scalar color = cv::Scalar(COLOR_LIST[colorIndex][0],
                                          COLOR_LIST[colorIndex][1],
                                          COLOR_LIST[colorIndex][2]);

            // Add the mask for said object
            mask(object.rect).setTo(color * 255, object.boxMask);
        }
        // Add all the masks to our image
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }

    // Bounding boxes and annotations
    for (auto & object : objects) 
    {
        // Choose the color
		int colorIndex = object.label % COLOR_LIST.size(); // We have only defined 80 unique colors
        cv::Scalar color = cv::Scalar(COLOR_LIST[colorIndex][0],
                                      COLOR_LIST[colorIndex][1],
                                      COLOR_LIST[colorIndex][2]);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor;
        if (meanColor > 0.5)
        {
            txtColor = cv::Scalar(0, 0, 0);
        }
        else
        {
            txtColor = cv::Scalar(255, 255, 255);
        }

        const auto& rect = object.rect;

        // Draw rectangles and text
        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[object.label].c_str(), object.probability * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = object.rect.x;
        int y = object.rect.y + 1;

        cv::rectangle(image, rect, color * 255, scale + 1);

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);

        // Pose estimation
        if (!object.kps.empty()) 
        {
            auto& kps = object.kps;
            for (int k = 0; k < NUM_KPS + 2; k++) 
            {
                if (k < NUM_KPS) 
                {
                    int   kpsX = std::round(kps[k * 3]);
                    int   kpsY = std::round(kps[k * 3 + 1]);
                    float kpsS = kps[k * 3 + 2];
                    if (kpsS > KPS_THRESHOLD) 
                    {
                        cv::Scalar kpsColor = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                        cv::circle(image, {kpsX, kpsY}, 5, kpsColor, -1);
                    }
                }
                auto& ske    = SKELETON[k];
                int   pos1X = std::round(kps[(ske[0] - 1) * 3]);
                int   pos1Y = std::round(kps[(ske[0] - 1) * 3 + 1]);

                int pos2X = std::round(kps[(ske[1] - 1) * 3]);
                int pos2Y = std::round(kps[(ske[1] - 1) * 3 + 1]);

                float pos1S = kps[(ske[0] - 1) * 3 + 2];
                float pos2S = kps[(ske[1] - 1) * 3 + 2];

                if (pos1S > KPS_THRESHOLD && pos2S > KPS_THRESHOLD) 
                {
                    cv::Scalar limbColor = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                    cv::line(image, {pos1X, pos1Y}, {pos2X, pos2Y}, limbColor, 2);
                }
            }
        }
    }
}