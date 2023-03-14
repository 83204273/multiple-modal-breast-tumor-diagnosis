# coding=utf-8
import shutil
import sys
import scipy.io as scio
import os
import numpy as np


#def cub2dang():
data = scio.loadmat('F:/paper/dataset/car/matlab.mat')

base_path = 'F:/paper/dataset/car/'
newpath_train = "F:/paper/dataset/car/car_train"
newpath_test = "F:/paper/dataset/car/car_test"
images = data['annotations'][0]
classes = data['class_names'][0]
num_images = images.size


for i in range(num_images):
        image_path = os.path.join(base_path, images[i][0][0])
        file_name = images[i][0][0]  # 文件名
        file_name = file_name.split('/')[1]#.encode('utf-8')
        classid = images[i][5][0][0]  # 类别
        # classid=np.array2string(classid)
        classid = classid.astype(np.int32)
        d = classes[classid-1][0]
        file_name_new = os.path.join(str(d), str(file_name))
        #file_name_new = str(d) + str(file_name)
        #print(file_name_new)


        istest = images[i][6][0]  # train/test
        if istest:
            if not os.path.exists(os.path.join(newpath_test, d)):
                os.makedirs(os.path.join(newpath_test, d))
            shutil.copy(image_path, os.path.join(newpath_test, file_name_new))
            with open('F:/paper/dataset/car/car_test.txt', 'a') as f:
                f.write('{} {}\n'.format(file_name_new, classid))

        if not istest:
            if not os.path.exists(os.path.join(newpath_train, d)):
                os.makedirs(os.path.join(newpath_train, d))
            shutil.copy(image_path, os.path.join(newpath_train, file_name_new))
            with open('F:/paper/dataset/car/car_train.txt', 'a') as f:
                f.write('{} {}\n'.format(file_name_new, classid))
