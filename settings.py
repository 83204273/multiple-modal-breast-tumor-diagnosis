import numpy as np
#base_architecture = 'vgg16'
base_architecture = 'resnet50'
#base_architecture = 'resnet101'
#base_architecture = 'densenet161'

img_size = 224
prototype_shape = (50, 128, 1, 1)
num_classes = 5
multi_sourece_num = 3


similarity_score = 0.85
seed = 20130702
#'log','linear','cos'
prototype_activation_function = 'cos'
#'bottleneck','regular'
add_on_layers_type = 'bottleneck'

#MG_data = np.array([0.00631, 0.04085, 0.48040, 0.15068, 0.06977])
#MRI_data = np.array([0.03032, 0.11220,	0.35088, 0.31037, 0.81366])

MG_data = np.array([0.006349, 	0.051440, 	0.486192, 	0.154150, 	0.069767 ])
MRI_data = np.array([0.030281, 	0.141289, 	0.354337, 	0.317523, 	0.761628 ])
# data_per_id
experiment_run = 'vis_104'
# US, MG, MRI
data_path = '/home/modm/zhuzhijie/breast_pro/datasets/US/selected_train70/'
data_path1 = '/home/modm/zhuzhijie/breast_pro/datasets/MG/selected_train70/'
data_path2 = '/home/modm/zhuzhijie/breast_pro/datasets/MRI/selected_train70/'

train_dir = data_path + 'train/'
train_dir1 = data_path1 + 'train/'
train_dir2 = data_path2 + 'train/'

test_dir = data_path + 'test/'
test_dir1 = data_path1 + 'test/'
test_dir2 = data_path2 + 'test/'


train_push_dir = data_path + 'train/'
# 80 100 75
batch_size = 64
train_batch_size = batch_size
test_batch_size = batch_size
train_push_batch_size = batch_size 

joint_optimizer_lrs = {'features': 1e-3,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 20

warm_optimizer_lrs = {'features': 3e-4,
                      'add_on_layers': 3e-4,
                      'prototype_vectors': 3e-4}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'crs_ent_com': 1,
    'crs_ent_dif': 1,
    'clst_dif': 0.8,
    'sep_dif': -0.08,
    
    'clst_com': 0.8,
    'sep_com': -0.08,
    'l1': 1e-4,
    'l2': 1e-3,
    
    'weight_decay': 1e-3,
    'momentum': 0.9

}

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 20
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
