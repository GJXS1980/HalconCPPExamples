#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from PIL import Image
from ultralytics import YOLO
import numpy as np

# 导入模型
model = YOLO('/home/lsrobot/lsrobot_ws/HalconCPPExamples/KawasakiDemo/models/0919/best.pt')

# 导入图片
beerResults = model('/home/lsrobot/lsrobot_ws/HalconCPPExamples/KawasakiDemo/bin/beer.png', conf=0.8, retina_masks=True, device=0, classes=1)  # results list
strapResults = model('/home/lsrobot/lsrobot_ws/HalconCPPExamples/KawasakiDemo/bin/beer.png', conf=0.6, retina_masks=True, device=0, classes=0)  # results list

# 导入原图
original_image = Image.open('/home/lsrobot/lsrobot_ws/HalconCPPExamples/KawasakiDemo/bin/beer.png') 
#   转换成图像数组
original_array = np.array(original_image)

#  获取原图大小
width, height = original_image.size

#  将原图变成黑色背景
black_image = Image.new("RGB", (width, height), (0, 0, 0))
# black_image.save("black_image.jpg") 
#   转换成图像数组
black_image_array = np.array(black_image)

beerPoint = None
strapPoint = None

# Show the results
for i in beerResults:
    beerBoxes = i.boxes
    # beerNames = r.cls
    # print(beerNames)
    beerPoint = beerBoxes.xyxy
    im_array = i.plot(boxes = False, img = black_image_array, pil = True)  # 去掉边框，保留掩膜
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    im.save('beerMask.jpg')  # save image

# Show the results
for j in strapResults:
    strapBoxes = j.boxes
    # beerNames = r.cls
    # print(beerNames)
    strapPoint = beerBoxes.xyxy
    im_array = j.plot(boxes = False, img = black_image_array, pil = True)  # 去掉边框，保留掩膜
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    im.save('strapMask.jpg')  # save image


mask_image_array = np.array(im)
mask_image_array[mask_image_array != 0] = 255
print(mask_image_array.shape)
mask_image = Image.fromarray(mask_image_array)
mask_image.save("mask_image_white.jpg") 

for k in range(0, (len(beerPoint))):
    #   xy与相机高宽对应
    for y in range(int(beerPoint[k][0].item()), (int(beerPoint[k][2].item()))):
        for x in range(int(beerPoint[k][1].item()), (int(beerPoint[k][3].item()))):
            mask_image_array[x][y] = [0, 0, 0]

mask_image = Image.fromarray(mask_image_array)
mask_image.save("mask_image_white1.jpg") 
#   转换成图像数组
mask_array = np.array(mask_image)

# 创建一个只显示掩膜区域的新图像数组
masked_region = np.where(mask_array == 255, original_array, 0)
# 将新图像数组转换回图像对象
result_image = Image.fromarray(masked_region)
# result_image.show()
# result_image.save("maskColor.jpg") 

# Show the results
for l in beerResults:
    im_array = l.plot()  # 去掉边框，保留掩膜
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    im.save('maskColor.jpg')  # save image

# Show the results
for n in strapResults:
    im_array = n.plot()  # 去掉边框，保留掩膜
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    im.save('maskColor1.jpg')  # save image