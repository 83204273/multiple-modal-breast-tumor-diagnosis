import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

from receptive_field import compute_proto_layer_rf_info_v2
from settings import multi_sourece_num, train_batch_size, similarity_score

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class PPNet(nn.Module):

    def __init__(self, features_com,features_dif, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-8
        
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features_com = features_com
        self.features_dif = features_dif

        features_name = str(self.features_com).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features_com.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
            first_add_on_layer_in_channels_com = \
                [i for i in features_com.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features_com.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
            first_add_on_layer_in_channels_com = \
                [i for i in features_com.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            
            add_on_layers_dif = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers_dif) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers_dif.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers_dif.append(nn.ReLU())
                add_on_layers_dif.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers_dif.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers_dif.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers_dif = nn.Sequential(*add_on_layers_dif)


            add_on_layers_com = []
            current_in_channels_com = first_add_on_layer_in_channels_com
            while (current_in_channels_com > self.prototype_shape[1]) or (len(add_on_layers_com) == 0):
                current_out_channels_com = max(self.prototype_shape[1], (current_in_channels_com // 2))
                add_on_layers_com.append(nn.Conv2d(in_channels=current_in_channels_com,
                                               out_channels=current_out_channels_com,
                                               kernel_size=1))
                add_on_layers_com.append(nn.ReLU())
                add_on_layers_com.append(nn.Conv2d(in_channels=current_out_channels_com,
                                               out_channels=current_out_channels_com,
                                               kernel_size=1))
                if current_out_channels_com > self.prototype_shape[1]:
                    add_on_layers_com.append(nn.ReLU())
                else:
                    assert (current_out_channels_com == self.prototype_shape[1])
                    add_on_layers_com.append(nn.Sigmoid())
                current_in_channels_com = current_in_channels_com // 2
            self.add_on_layers_com = nn.Sequential(*add_on_layers_com)
        else:
            self.add_on_layers_dif = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels_com, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
            self.add_on_layers_com = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels_com, out_channels=self.prototype_shape[1],
                          kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
            )


        self.prototype_vectors_com =  nn.Parameter(torch.rand(self.prototype_shape[0],self.prototype_shape[1],self.prototype_shape[2],self.prototype_shape[3]),
                                              requires_grad=True)

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape[0]*multi_sourece_num,self.prototype_shape[1],self.prototype_shape[2],self.prototype_shape[3]),
                                              requires_grad=True)
        self.prototype_vectors_channel_com =  nn.Parameter(torch.ones(self.prototype_shape[0],self.prototype_shape[1],self.prototype_shape[2],self.prototype_shape[3]),
                                              requires_grad=True)

        self.prototype_vectors_channel = nn.Parameter(torch.ones(self.prototype_shape[0]*multi_sourece_num,self.prototype_shape[1],self.prototype_shape[2],self.prototype_shape[3]),
                                              requires_grad=True)
                                              
        self.prototype_class_identity_dif = torch.zeros(self.num_prototypes*multi_sourece_num,
                                                    self.num_classes*multi_sourece_num)
        for j in range(self.num_prototypes*multi_sourece_num ):
            self.prototype_class_identity_dif[j, j // num_prototypes_per_class] = 1



        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones_com = nn.Parameter(torch.ones(self.prototype_shape[0],self.prototype_shape[1],self.prototype_shape[2],self.prototype_shape[3]),
                                              requires_grad=False)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape[0]*multi_sourece_num,self.prototype_shape[1],self.prototype_shape[2],self.prototype_shape[3]),
                                              requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes * 12, self.num_classes,
                                    bias=False) # do not use bias
        self.last_layer_dif = nn.Linear(self.num_prototypes*9  , self.num_classes ,
                                    bias=False) 
        self.last_layer_com = nn.Linear(self.num_prototypes , self.num_classes,
                                    bias=False) 


        if init_weights:
            self._initialize_weights()

    def conv_features(self, x_US, x_MG, x_MRI):
        '''
        the feature input to prototype layer
        '''
        
        x_com = self.features_com(torch.cat([x_US, x_MG, x_MRI], dim=0))
        x_com = self.add_on_layers_com(x_com)
        
        x_dif = self.features_dif(torch.cat([x_US, x_MG, x_MRI], dim=0))
        x_dif = self.add_on_layers_dif(x_dif)
        #print (x_dif.size(), x_com.size())
        #del temp
        return x_dif, x_com

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)
        

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def _l2_convolution_com(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        #print(x2.size(), self.ones_com.size())
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones_com)

        p2 = self.prototype_vectors_com ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)
        
        xp = F.conv2d(input=x, weight=self.prototype_vectors_com)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances
        
    def _weight_cosine(self,x,w):
        w = F.relu(w)

        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=w)

        p2 = self.prototype_vectors**2
        p2 = torch.sum(p2*w, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)


        xp = F.conv2d(input=x, weight=self.prototype_vectors*w)

        cos_xp = torch.div(xp.float(), ((x2_patch_sum ** 0.5) * (p2_reshape ** 0.5)).float() )

        distances = F.relu(cos_xp )

        return 1-distances
        
    def _weight_cosine_com(self,x,w):
        w = F.relu(w)

        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=w)

        p2 = self.prototype_vectors_com**2
        p2 = torch.sum(p2*w, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)


        xp = F.conv2d(input=x, weight=self.prototype_vectors_com*w)

        cos_xp = torch.div(xp.float(), ((x2_patch_sum ** 0.5) * (p2_reshape ** 0.5)).float() )

        distances = F.relu(cos_xp )

        return 1-distances

    def prototype_distances(self, x_US, x_MG, x_MRI):
        '''
        x is the raw input
        '''
        
        conv_features, conv_features_com = self.conv_features(x_US, x_MG, x_MRI)
        if self.prototype_activation_function == 'cos':
            distances = self._weight_cosine(conv_features, self.prototype_vectors_channel)
            distances_com =self._weight_cosine_com(conv_features_com, self.prototype_vectors_channel_com)
        else :
            distances = self._l2_convolution(conv_features)
            distances_com =self._l2_convolution_com(conv_features_com)
        return distances, distances_com

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'cos':
            return 1-distances
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x_US, x_MG, x_MRI):
        distances, distances_com = self.prototype_distances(x_US, x_MG, x_MRI)
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        size = int(distances_com.size()[0]/3)
        # global min pooling
        min_distances_dif = self.similar_rigion_distances(distances,3)
        prototype_activations_dif = self.distance_2_similarity(min_distances_dif)
        
        min_distances_com = self.similar_rigion_distances(distances_com,1)
        prototype_activations_com = self.distance_2_similarity(min_distances_com)
        #prototype_activations_dif = self.distance_2_similarity(distances)
        #prototype_activations_com = self.distance_2_similarity(distances_com)
        #prototype_activations_dif = prototype_activations_dif.view(-1, self.num_prototypes*3)
        #prototype_activations_com = prototype_activations_com.view(-1, self.num_prototypes)
        #print(min_distances_com.size())

        prototype_activations_com_temp = torch.cat([prototype_activations_com[0:size,:],prototype_activations_com[size:2*size,:],prototype_activations_com[2*size:3*size,:]],dim=1)
        prototype_activations_com_temp.view(-1, self.num_prototypes*3)
        prototype_activations_dif_temp = torch.cat([prototype_activations_dif[0:size,:],prototype_activations_dif[size:2*size,:],prototype_activations_dif[2*size:3*size,:]],dim=1)
        prototype_activations_dif_temp.view(-1, self.num_prototypes*3)

        
        prototype_activations_cat = torch.cat([ prototype_activations_dif_temp,prototype_activations_com_temp], dim=1)
        #print(prototype_activations_cat.size())
        logits = self.last_layer(prototype_activations_cat)
        logits_dif = self.last_layer_dif(prototype_activations_dif_temp)
        logits_com = self.last_layer_com(prototype_activations_com)
        
        #return logits, min_distances_dif, min_distances_com, logits_dif, logits_com
        return logits, min_distances_dif, min_distances_com, logits_dif, logits_com
        
    def similar_rigion_distances(self, distances, prototype_num):
        min_distances = -F.max_pool2d(-distances,
                                     kernel_size=(distances.size()[2],
                                                  distances.size()[3]))
        if similarity_score != 1:
            temp_sim = torch.ceil(F.relu((1 - distances) - (similarity_score- min_distances) ))
            local_image = temp_sim * distances

            temp_similar_features_sum = torch.sum(local_image, dim=(2, 3))
            temp_num = torch.sum(temp_sim, dim=(2, 3))

            region_features = temp_similar_features_sum / (temp_num + self.epsilon)
            min_distances = region_features.view(-1, self.num_prototypes*prototype_num)
        else:
            min_distances = min_distances.view(-1, self.num_prototypes*prototype_num)
        
        return min_distances

    def push_forward(self, x_US, x_MG, x_MRI):
        '''this method is needed for the pushing operation'''
        conv_features, conv_features_com = self.conv_features(x_US, x_MG, x_MRI)
        if self.prototype_activation_function == 'cos':
            distances = self._weight_cosine(conv_features, self.prototype_vectors_channel)
        else :
            distances = self._l2_convolution(conv_features)
        
        return conv_features,distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        #temp = self.prototype_class_identity.repeat(1, 3)
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        #print(positive_one_weights_locations.size())
        negative_one_weights_locations = 1 - positive_one_weights_locations
        
    

        #positive_one_weights_locations_dif_temp  = torch.t(self.prototype_class_identity_dif[:,0:5] + self.prototype_class_identity_dif[:,5:10] + self.prototype_class_identity_dif[:,10:15])
        positive_one_weights_locations_dif_temp  = torch.t(torch.cat([self.prototype_class_identity_dif[:,0:5] , self.prototype_class_identity_dif[:,5:10] , self.prototype_class_identity_dif[:,10:15]],dim=0))
        negative_one_weights_locations_dif_temp = 1 - positive_one_weights_locations_dif_temp
        
        positive_one_weights_locations_dif  = torch.t(torch.cat([self.prototype_class_identity_dif[:,0:5] , self.prototype_class_identity_dif[:,5:10] , self.prototype_class_identity_dif[:,10:15]],dim=0))
        #positive_one_weights_locations_dif  = positive_one_weights_locations_dif_temp.repeat(1,3)
        negative_one_weights_locations_dif = 1 - positive_one_weights_locations_dif

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        positive_one_weights_locations_cat = torch.cat([positive_one_weights_locations_dif_temp, positive_one_weights_locations.repeat(1, 3)], dim=1)
        negative_one_weights_locations_cat = torch.cat([negative_one_weights_locations_dif_temp, negative_one_weights_locations.repeat(1, 3)], dim=1)
        
        initial_dif = correct_class_connection * positive_one_weights_locations_dif + incorrect_class_connection * negative_one_weights_locations_dif
        initial_com = correct_class_connection * positive_one_weights_locations + incorrect_class_connection * negative_one_weights_locations
        
        initial = correct_class_connection * positive_one_weights_locations_cat + incorrect_class_connection * negative_one_weights_locations_cat

        #avg_initial = (initial[0:6,:] + initial[6:12,:] + initial[12:18,:])/3
        self.last_layer.weight.data.copy_(initial)
        self.last_layer_dif.weight.data.copy_(initial_dif)
        self.last_layer_com.weight.data.copy_(initial_com)
        #self.last_layer_com.weight.data.copy_(initial_com)
        

    def _initialize_weights(self):

        for m in self.add_on_layers_com.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.add_on_layers_dif.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

                

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)



def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 512, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features_com = base_architecture_to_features[base_architecture](pretrained=pretrained)
    features_dif = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features_com.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return PPNet(features_com=features_com,
                 features_dif=features_dif,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)

