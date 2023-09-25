#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

def image_seg(image, beerClass, strapClass, beerConf, strapConf):
    '''
    input:
        image:      待识别图像
        beerClass:  啤酒瓶类
        strapClass: 捆扎带类
        beerConf:   啤酒瓶置信度
        strapConf:  捆扎带置信度
    return:
        mask_image: 处理后的掩膜
        im_result:  实例分割的结果（所有类）
    '''

    # 导入模型
    model = YOLO('/home/lsrobot/lsrobot_ws/HalconCPPExamples/KawasakiDemo/models/0919/best.pt')

    # 导入图片进行识别
    Results = model(image, conf=strapConf)  # 识别全部
    beerResults = model(image, conf=beerConf, retina_masks=True, device=0, classes=beerClass)  # 啤酒瓶
    strapResults = model(image, conf=strapConf, retina_masks=True, device=0, classes=strapClass)  # 捆扎带

    # 导入原图
    original_image = Image.open(image) 

    #  获取原图大小
    width, height = original_image.size

    #  将原图变成黑色背景
    black_image = Image.new("RGB", (width, height), (0, 0, 0))
    #   转换成图像数组
    black_image_array = np.array(black_image)

    beerPoint = None
    strapPoint = None

    # 啤酒瓶掩膜处理
    for i in beerResults:
        beerBoxes = i.boxes
        beerPoint = beerBoxes.xyxy
        im_array = i.plot(boxes = False, img = black_image_array, pil = True)  # 去掉边框，保留掩膜
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

    # 捆扎带掩膜处理
    for j in strapResults:
        strapBoxes = j.boxes
        strapPoint = beerBoxes.xyxy
        im_array = j.plot(boxes = False, img = black_image_array, pil = True)  # 去掉边框，保留掩膜
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

    # 识别到所有的结果
    for m in Results:
        result_array = m.plot()  # 去掉边框，保留掩膜
        im_result = Image.fromarray(result_array[..., ::-1])  # RGB PIL image
        im_result = np.array(im_result)

    mask_image_array = np.array(im)
    mask_image_array[mask_image_array != 0] = 255
    # print(mask_image_array.shape)
    mask_image = Image.fromarray(mask_image_array)

    for k in range(0, (len(beerPoint))):
        #   xy与相机高宽对应
        for y in range(int(beerPoint[k][0].item()), (int(beerPoint[k][2].item()))):
            for x in range(int(beerPoint[k][1].item()), (int(beerPoint[k][3].item()))):
                mask_image_array[x][y] = [0, 0, 0]

    mask_image = Image.fromarray(mask_image_array)
    #   将PIL.Image.Image格式转numpy.ndarray格式
    mask_image = np.array(mask_image)
    return mask_image, im_result

# mask_image, result_image = image_seg('/home/lsrobot/lsrobot_ws/HalconCPPExamples/KawasakiDemo/bin/beer.png', 1, 0, 0.8, 0.6)
# cv2.imwrite("mask_image.png", mask_image)
# cv2.imwrite("result_image.png", result_image)