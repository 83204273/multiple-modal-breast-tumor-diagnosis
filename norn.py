import numpy as np
import cv2
import os
from PIL import Image



img_h, img_w = 224, 224    #经过处理后你的图片的尺寸大小
means, stdevs = [], []
img_list = []

img_root_path = '/home/modm/zhuzhijie/breast_pro/datasets/MRI/train70/'
imgs_path = img_root_path + 'train/'
imgs_path_list = os.listdir(imgs_path)

len_ = len(imgs_path_list)
i = 0
for it in imgs_path_list:
    sub_imgs_path_list = os.listdir(os.path.join(imgs_path, it))
    for item in sub_imgs_path_list:
        img = cv2.imread(os.path.join(imgs_path, it, item))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        #print(i, '/', len_)

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
