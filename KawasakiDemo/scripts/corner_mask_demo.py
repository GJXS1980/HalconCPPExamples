#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from PIL import Image
from ultralytics import YOLO
import numpy as np

# 导入模型
model = YOLO('/home/lsrobot/lsrobot_ws/HalconCPPExamples/KawasakiDemo/models/0907/corner-best-0907.pt')

# 导入图片
results = model('/home/lsrobot/lsrobot_ws/HalconCPPExamples/KawasakiDemo/bin/corner1.jpg')  # results list

# 导入原图
original_image = Image.open('/home/lsrobot/lsrobot_ws/HalconCPPExamples/KawasakiDemo/bin/corner2.jpg') 
#   转换成图像数组
original_array = np.array(original_image)

#  获取原图大小
width, height = original_image.size

#  将原图变成黑色背景
black_image = Image.new("RGB", (width, height), (0, 0, 0))
# black_image.save("black_image.jpg") 
#   转换成图像数组
black_image_array = np.array(black_image)

# Show the results
for r in results:
    im_array = r.plot(boxes = False, img = black_image_array, pil = True)  # 去掉边框，保留掩膜
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('mask.jpg')  # save image

mask_image_array = np.array(im)
mask_image_array[mask_image_array != 0] = 255
mask_image = Image.fromarray(mask_image_array)
mask_image.save("mask_image_white.jpg") 
#   转换成图像数组
mask_array = np.array(mask_image)

# 创建一个只显示掩膜区域的新图像数组
masked_region = np.where(mask_array == 255, original_array, 0)
# 将新图像数组转换回图像对象
result_image = Image.fromarray(masked_region)
result_image.show()
result_image.save("maskColor.jpg") 
