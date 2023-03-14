import torch
import copy
import numpy as np

def match_expand_data(dataset_change, dataset_match):
    imgs_num = len(dataset_match.imgs)
    class_num = len(dataset_match.classes)

    imgs_change_num = len(dataset_change.imgs)
    result = copy.deepcopy(dataset_change)
    result.imgs = []
    result.samples = []
    result.targets = []
    idx = np.zeros([class_num,imgs_change_num], dtype=int)-1
    idx_cp = np.zeros(class_num)
    k = 0
    for j in enumerate(dataset_change.imgs): 
        temp = int(j[1][1])
        idx[temp][int(idx_cp[temp])] = k
        k += 1
        idx_cp[temp] += 1

    idx_cp = np.zeros(class_num)
    temp = int(0)
    for i in enumerate(dataset_match.imgs):
        temp = int(i[1][1])
        #x = dataset_change.imgs[1]
        #print(dataset_change.imgs[int(idx[temp][int(idx_cp[temp])])])
        result.imgs.append(dataset_change.imgs[int(idx[temp][int(idx_cp[temp])])])
        result.targets.append(dataset_change.targets[int(idx[temp][int(idx_cp[temp])])])
        result.samples.append(dataset_change.samples[int(idx[temp][int(idx_cp[temp])])])
        if(idx[temp][int(idx_cp[temp]+1)] == -1 ):
            idx_cp[temp] = 0
        else:
            idx_cp[temp] += 1
             
    del idx
    return result