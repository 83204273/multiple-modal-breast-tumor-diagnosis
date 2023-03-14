import os
import random
import time
import sys
import matplotlib.pyplot as plt
import cv2
import torch
import torch.utils.data as data
import numpy as np
import PIL.Image as Image
from PIL import ImageStat



def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def disrupt_image(img_path, save_path, x, y, per):
    para_swap = per
    for t in range(len(img_path)):
        makedir(save_path[t])
        #print(img_path[0])
        image_name = [os.path.join(img_path[t], name) for name in next(os.walk(img_path[t]))[2]]
        #print(image_name)
        save_image_name = [os.path.join(save_path[t], name) for name in next(os.walk(img_path[t]))[2]]
        #print(image_name[0])
        for name_num in range(len(image_name)):
            name = image_name[name_num]
            print(name)
            org_img = cv2.imread(name, -1)
            #org_img = Image.open(name)
            img = cv2.resize(org_img, (224, 224))
            #plt.imshow(org_img)
            #plt.show()
            #time.sleep(100)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            high = img.shape[0]
            width = img.shape[1]
            n = x
            m = y
            crop_side = int(high/n)

            im_list = []

            dis_h = int(np.floor(high / n))
            dis_w = int(np.floor(width / m))
            num = 0
            for i in range(m):
                for j in range(n):
                    num += 1
                    im_list.append(img[dis_h * j:dis_h * (j + 1), dis_w * i:dis_w * (i + 1), :])


            swap_stats = [sum((ImageStat.Stat(Image.fromarray(im).convert("RGB"))).mean) for im in im_list]
            ponit_xy = [0, 0]
            #image = Image.fromarray(arr).convert("RGB")
            disrupt_dis = 1000
            distance=[]



            for j in range(len(swap_stats)):
                for i in range(j+1, len(swap_stats)):
                    distance.append (abs(swap_stats[j] - swap_stats[i]))


                    min_distance = min(distance)

                    if min_distance < disrupt_dis:
                        min_index = distance.index(min(distance))
                        #print(min_index)
                        a = divmod(min_index, n)[1]
                        xx, yy = divmod(a + min_index - j, n)
                        dis_x = x
                        dis_y = abs(yy - a)
                        disrupt_dis = min(dis_x, dis_y)


            to_image = Image.new('RGB', (width, high))  # 创建一个新图
            disrupt_dis = 1

            x_y = np.arange(0, n*m, 1)

            for i in range(len(x_y)):
                if random.randint(0, 100) > para_swap:
                    x_y = swap_x(x_y, disrupt_dis, i)
                else:
                    x_y =  x_y
                if random.randint(0, 100) > para_swap:
                    x_y = swap_y(x_y, disrupt_dis, i)

            num = 0
            for i in range(m):
                for j in range(n):
                    to_image.paste(Image.fromarray(im_list[x_y[num]]).convert("RGB"), (i * crop_side, j * crop_side))
                    num +=1

            #print(str(save_image_name[name_num]))
            #cv2.imwrite(str(save_image_name[name_num]), to_image)
            to_image.save( str(save_image_name[name_num]))



def swap_x(location_list, disrupt_dis, i):

    a = random.randint(- disrupt_dis, disrupt_dis)
    crop_num = (len(location_list)) ** (1/2)
    swap_area = []
    for m in range(-disrupt_dis, disrupt_dis+1, 1):
        for n in range(-disrupt_dis, disrupt_dis+1, 1):
            swap_area.append(int(i + m * crop_num + n))


    if (((i+a)>0) & ((i+a)<len(location_list))):

        if (location_list[i+a] in swap_area):
            location_list[i], location_list[i+a] = location_list[i+a], location_list[i]

    return location_list

def swap_y(location_list, disrupt_dis, i):
    a = random.randint(- disrupt_dis, disrupt_dis)
    crop_num = int((len(location_list)) ** (1/2))
    swap_area = []
    for m in range(-disrupt_dis,disrupt_dis):
        for n in range(-disrupt_dis,disrupt_dis):
            swap_area.append(i + m * crop_num + n)

    if (((i+a*crop_num)>0) & ((i+a*crop_num)<len(location_list))):
        if (location_list[(i+a*crop_num)] in swap_area):
           location_list[i], location_list[i+a*crop_num] = location_list[i+a*crop_num], location_list[i]


    return location_list


def crop_image(self, image, cropnum):
    width, high = image.size
    crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
    crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
    im_list = []
    for j in range(len(crop_y) - 1):
        for i in range(len(crop_x) - 1):
             im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
    return im_list

if __name__ == '__main__':

    x=7
    y=7
    per=50




    #datasets_root_dir = 'F:/paper/Protonet/ProtoPNet-master/ProtoPNet-master/datasets/cub200_cropped/'
    datasets_root_dir = '//home/modm/zhuzhijie/img_aug/ProtoPNet-master/ProtoPNet-master/datasets/cub200_cropped/'

    sysdir = sys.path[0]
    dir = datasets_root_dir + 'train_cropped_augmented/'
    target_dir = datasets_root_dir + 'train_cropped_disrupt' + str(x) + '_'+ str(y) + '_'+ str(per) + '/'

    print(sys.path[0])
    makedir(target_dir)
    folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]

    target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

    disrupt_image(folders, target_folders, x, y, per)