import cv2
import numpy as np
from math import ceil

# 读取图片
img_path = "../data/img/256.jpg"
img = cv2.imread(img_path)

# 读取图片大小信息
height, width, channel = img.shape

# 获取bbox位置信息
bbox_path = "../data/bbox/256.txt"
with open(bbox_path, "r") as f:
    contents = f.readlines()
    x_min = []
    y_min = []
    x_max = []
    y_max = []
    for _, content in enumerate(contents):
        cls, x1, y1, x2, y2 = content.split()
        x_min.append(float(x1))
        y_min.append(float(y1))
        x_max.append(float(x2))
        y_max.append(float(y2))

    # 目标检测框在新图像中的位置,裁剪新图像
    if max(x_max) - min(x_min) < width/2 and max(y_max) - min(y_min) < height/2:
        x_mean = np.mean(x_min + x_max)
        y_mean = np.mean(y_min + y_max)

        x_min = x_min - x_mean + width/4
        x_max = x_max - x_mean + width/4
        y_min = y_min - y_mean + height/4
        y_max = y_max - y_mean + height/4

        bbox_save_path = "../data/bbox/256_crop.txt"
        with open(bbox_save_path, "a+") as f:
            for i in range(len(x_min)):
                f.write("1 "+str(x_min[i])+" "+str(y_min[i])+" "+str(x_max[i])+" "+str(y_max[i])+"\n")

        # 生成quarter_img并保存
        """
        quarter_img = img[ceil(y_mean-height/4):ceil(y_mean+height/4), int(x_mean-width/4):int(x_mean+width/4)]
        save_path = "../228_crop.jpg"
        cv2.imwrite(save_path, quarter_img)
        """
    # TODO
    else:
        pass





"""   
cv2.imshow("test", img)
cv2.waitKey(0)
"""