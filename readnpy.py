#导入所需的包
import numpy as np

#导入npy文件路径位置

root_path = 'F:/paper/Protonet/my/RESULT/source_30_101/'
train_dir = '9nopush0.7715_nearest_train/'
test_dir = '9nopush0.7715_nearest_test/'
proty_num = '1997/'
num = '1'
train_npy_dir = root_path + train_dir + proty_num
test_npy_dir = root_path + test_dir + proty_num

test_id = np.load(test_npy_dir+'class_id.npy')
train_id = np.load(train_npy_dir+'class_id.npy')
print('test ',test_id)
print('train',train_id)

#test_act = np.load(test_npy_dir+'nearest-' +num +'_act.npy')
#train_act = np.load(train_npy_dir+'nearest-' +num +'_act.npy')
#print(test_act, train_act)

#test_patch_indices = np.load(test_npy_dir+'nearest-1_high_act_patch_indices.npy')
#train_patch_indices = np.load(train_npy_dir+'nearest-1_high_act_patch_indices.npy')
#print(test_patch_indices)