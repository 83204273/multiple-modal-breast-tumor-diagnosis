import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re
import numpy as np

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import preprocess_input_function, normMean_US, normStd_US, normMean_MG, normStd_MG, normMean_MRI, normStd_MRI


from torch.utils.data import RandomSampler, BatchSampler
#from torch.utils.data.distributed import DistributedSampler
from data_match import match_expand_data
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0,1,2,3') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

#torch.autograd.set_detect_anomaly(True)

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run, seed, coefs
                     
random.seed(seed) 
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, train_dir1, train_dir2, test_dir,test_dir1,test_dir2, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size

#normalize = transforms.Normalize(mean=mean,  std=std)

# all datasets
# train set
seed = 1024

train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.RandomRotation(40, resample=False, expand=False, center=None),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize(normMean_US, normStd_US),
    ]))
    
#pseudo_dataset = list(range(len(train_dataset.samples)))
#batchSampler1 = BatchSampler(list(range(len(train_dataset.samples))), batch_size=train_batch_size, drop_last=False)
g = torch.Generator()
g.manual_seed(seed)
batchSampler1 = RandomSampler(list(range(len(train_dataset.samples))),generator=g)
train_loader = torch.utils.data.DataLoader(
    train_dataset,batch_size=train_batch_size, sampler=batchSampler1, shuffle=False, generator=g,
    num_workers=0, pin_memory=False)

train_dataset1 = datasets.ImageFolder(
    train_dir1,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.RandomRotation(40, resample=False, expand=False, center=None),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize(normMean_MG, normStd_MG),
    ]))
train_dataset1 =  match_expand_data(train_dataset1, train_dataset)

g = torch.Generator()
g.manual_seed(seed)
batchSampler2 = RandomSampler(list(range(len(train_dataset.samples))),generator=g)

train_loader1 = torch.utils.data.DataLoader(
    train_dataset1,batch_size=train_batch_size, sampler=batchSampler2, shuffle=False, generator=g,
    num_workers=0, pin_memory=False)

g = torch.Generator()
g.manual_seed(seed)
batchSampler3 = RandomSampler(list(range(len(train_dataset.samples))),generator=g)
train_dataset2 = datasets.ImageFolder(
    train_dir2,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.RandomRotation(40, resample=False, expand=False, center=None),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize(normMean_MRI, normStd_MRI),
    ]))
train_dataset2 =  match_expand_data(train_dataset2, train_dataset)
train_loader2 = torch.utils.data.DataLoader(
    train_dataset2, batch_size=train_batch_size, sampler=batchSampler3, shuffle=False,generator=g,
    num_workers=0, pin_memory=False)


# push set
push_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
    
g = torch.Generator()
g.manual_seed(seed)
push_batchSampler1 = RandomSampler(list(range(len(push_dataset.samples))),generator=g)
push_loader = torch.utils.data.DataLoader(
    push_dataset,batch_size=train_batch_size, sampler=push_batchSampler1, shuffle=False, generator=g,
    num_workers=0, pin_memory=False)

push_dataset1 = datasets.ImageFolder(
    train_dir1,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
push_dataset1 =  match_expand_data(push_dataset1, push_dataset)

g = torch.Generator()
g.manual_seed(seed)
push_batchSampler2 = RandomSampler(list(range(len(push_dataset.samples))),generator=g)

push_loader1 = torch.utils.data.DataLoader(
    push_dataset1,batch_size=train_batch_size, sampler=push_batchSampler2, shuffle=False, generator=g,
    num_workers=0, pin_memory=False)

g = torch.Generator()
g.manual_seed(seed)
push_batchSampler3 = RandomSampler(list(range(len(push_dataset.samples))),generator=g)
push_dataset2 = datasets.ImageFolder(
    train_dir2,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
push_dataset2 =  match_expand_data(push_dataset2, push_dataset)
push_loader2 = torch.utils.data.DataLoader(
    push_dataset2, batch_size=train_batch_size, sampler=batchSampler3, shuffle=False,generator=g,
    num_workers=0, pin_memory=False)

    
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(normMean_US, normStd_US),
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

test_dataset1 = datasets.ImageFolder(
    test_dir1,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(normMean_MG, normStd_MG),
    ]))
test_dataset1 =  match_expand_data(test_dataset1, test_dataset)
test_loader1 = torch.utils.data.DataLoader(
    test_dataset1, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

test_dataset2 = datasets.ImageFolder(
    test_dir2,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(normMean_MRI, normStd_MRI),
    ]))
test_dataset2 =  match_expand_data(test_dataset2, test_dataset)
test_loader2 = torch.utils.data.DataLoader(
    test_dataset2, batch_size=test_batch_size, shuffle=False,
    num_workers=0, pin_memory=False)



# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features_com.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': coefs['weight_decay'], 'momentum': coefs['momentum']}, # bias are now also being regularized
 {'params': ppnet.add_on_layers_com.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': coefs['weight_decay'], 'momentum': coefs['momentum']},
 {'params': ppnet.features_dif.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': coefs['weight_decay'], 'momentum': coefs['momentum']}, # bias are now also being regularized
 {'params': ppnet.add_on_layers_dif.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': coefs['weight_decay'], 'momentum': coefs['momentum']},
 
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.prototype_vectors_com, 'lr': joint_optimizer_lrs['prototype_vectors']},
  {'params': ppnet.prototype_vectors_channel, 'lr': joint_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.prototype_vectors_channel_com, 'lr': joint_optimizer_lrs['prototype_vectors']}
]
joint_optimizer = torch.optim.SGD(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers_com.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': coefs['weight_decay'], 'momentum': coefs['momentum']},
{'params': ppnet.add_on_layers_dif.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': coefs['weight_decay'], 'momentum': coefs['momentum']},


 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.prototype_vectors_com, 'lr': warm_optimizer_lrs['prototype_vectors']},
   {'params': ppnet.prototype_vectors_channel, 'lr': joint_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.prototype_vectors_channel_com, 'lr': joint_optimizer_lrs['prototype_vectors']}
]
warm_optimizer = torch.optim.SGD(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr},
{'params': ppnet.last_layer_com.parameters(), 'lr': last_layer_optimizer_lr},
{'params': ppnet.last_layer_dif.parameters(), 'lr': last_layer_optimizer_lr}]

last_layer_optimizer = torch.optim.SGD(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

# train the model
log('start training')
import copy
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader,train_loader1=train_loader1,train_loader2 = train_loader2, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader,train_loader1=train_loader1,train_loader2 = train_loader2, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader, test_loader1=test_loader1, test_loader2 = test_loader2,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.90, log=log)

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            dataloader = push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            dataloader1 = push_loader1,
            dataloader2 =push_loader2,
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader, test_loader1=test_loader1,test_loader2 = test_loader2,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.90, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader,train_loader1=train_loader1,train_loader2 = train_loader2, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader, test_loader1=test_loader1, test_loader2 = test_loader2,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.90, log=log)
   
logclose()


