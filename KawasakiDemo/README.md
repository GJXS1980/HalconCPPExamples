# 川崎机器人3D抓取

### TODO
1. 掩膜区域的提取

2. 点云聚类优化

3. 点云拟合和中心坐标点求解优化



#   yolov8 TensorRT
### 模型转换
修改<code>model = YOLO("../models/yolov8m.pt")</code>,替换成转换模型：
```bash
python3 pytorch2onnx.py
```

### 应用
```bash
cd bin

#   图像识别
./detect_object_image --model /path/to/your/onnx/model.onnx --input /path/to/your/image.jpg

#   摄像头实时识别
./detect_object_video --model /path/to/your/onnx/model.onnx --input 0


#   梅卡相机测试(拍单张照片)
./MechEyeDetect --model /path/to/your/onnx/model.onnx


#   梅卡相机测试(实时拍)
./MechEyeDetectBox --model /path/to/your/onnx/model.onnx


# yolo官方命令

```





#   Q & A
> Q1:mecheye_boxes_Point_Cloud_Registration.cpp:287:43: error: ‘make_shared’ is not a member of ‘boost’
  287 |                 extract.setIndices(boost::make_shared<const pcl::PointIndices>(indices));
          

A1: 将boost::make_shared换成std::make_shared

