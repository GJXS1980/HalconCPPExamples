#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from PIL import Image
from ultralytics import YOLO
import numpy as np

# 导入模型
model = YOLO('/home/lsrobot/lsrobot_ws/HalconCPPExamples/KawasakiDemo/models/0907/corner-best-0907.pt')

# 导入图片
results = model('/home/lsrobot/lsrobot_ws/HalconCPPExamples/KawasakiDemo/bin/corner2.jpg')  # results list

# Show the results
for r in results:
    im_array = r.plot()  # 去掉边框，保留掩膜
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('corner-color.jpg')  # save image
