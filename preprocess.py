import torch


#US-70%
normMean_US = (0.313, 0.318, 0.328)
normStd_US = (0.212, 0.214, 0.217)

#MG-70%
normMean_MG = (0.162, 0.162, 0.162)
normStd_MG = (0.197, 0.197, 0.197)

#MRI-70%
normMean_MRI = (0.152, 0.152, 0.152)
normStd_MRI = (0.165, 0.165, 0.165)

#imagesnet
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def preprocess_input_function(x, source):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    if source == 'US':
        return preprocess(x, mean=normMean_US, std=normStd_US )
    elif source =='MG':
        return preprocess(x, mean=normMean_MG, std=normStd_MG )
    else:
        return preprocess(x, mean=normMean_MRI, std=normStd_MRI)

def undo_preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y

def undo_preprocess_input_function(x):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    return undo_preprocess(x, mean=mean, std=std)
