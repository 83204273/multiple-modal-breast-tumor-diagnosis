import torchvision.transforms as transforms
import torch
import copy
import numpy as np
import torchvision.datasets as datasets
from preprocess import normMean_US, normStd_US, normMean_MG, normStd_MG, normMean_MRI, normStd_MRI
from settings import img_size

def load_data(train_dir, train_dir1, train_dir2):
    #(US, MG, MRI)
    train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.RandomRotation(30, resample=False, expand=False, center=None),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize(normMean_US, normStd_US)
    ]))

    train_dataset1 = datasets.ImageFolder(
    train_dir1,)

    train_dataset2 = datasets.ImageFolder(
    train_dir2,)

    train_dataset1 = match_expand_data(train_dataset1, train_dataset, 'MG')
    train_dataset2 = match_expand_data(train_dataset2, train_dataset, 'MRi')

    return train_dataset, train_dataset1, train_dataset


def match_expand_data(dataset_change, dataset_match, data_name):

    imgs_num = len(dataset_match.imgs)
    class_num = len(dataset_match.classes)

    imgs_change_num = len(dataset_change.imgs)
    result = copy.deepcopy(dataset_change)
    result.imgs = []
    result.samples = []
    result.targets = []
    idx = np.zeros([class_num, imgs_change_num], dtype=int) - 1
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
        # x = dataset_change.imgs[1]
        # print(dataset_change.imgs[int(idx[temp][int(idx_cp[temp])])])
        result.imgs.append(dataset_change.imgs[int(idx[temp][int(idx_cp[temp])])])
        result.targets.append(dataset_change.targets[int(idx[temp][int(idx_cp[temp])])])
        print(dataset_match.transform)
        target = transfrom_data(dataset_change.samples[int(idx[temp][int(idx_cp[temp])])], data_name)
        result.samples.append(target)
        if (idx[temp][int(idx_cp[temp] + 1)] == -1):
            idx_cp[temp] = 0
        else:
            idx_cp[temp] += 1
 
    del idx
    return result

def transfrom_data(img, data_name):
    if (data_name == 'MG'):
        normMean, normStd =  normMean_MG, normStd_MG
    else:
        normMean, normStd = normMean_MRI, normStd_MRI
    tran = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.RandomRotation(30, resample=False, expand=False, center=None),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)
    ])
    img = tran(img)
    return img