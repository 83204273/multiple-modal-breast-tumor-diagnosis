import time
import torch

from helpers import list_of_distances, make_one_hot
from settings import  multi_sourece_num, train_batch_size, prototype_activation_function, MG_data, MRI_data

def _train_or_test(model, dataloader, train_loader1,train_loader2,optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    
    total_cluster_cost_com = 0
    total_separation_cost_com = 0
    total_avg_separation_cost_com = 0
    total_cross_entropy_dif = 0
    total_cross_entropy_com = 0
    
    train_data1_num = len(dataloader.dataset)
    train_data2_num = len(train_loader1.dataset)
    train_data3_num = len(train_loader2.dataset)
    train_data1 = 0
    train_data2 = 0
    train_data3 = 0
    n_correct_1 = 0
    n_correct_2 = 0
    n_correct_3 = 0
    n_correct_1_com = 0
    n_correct_2_com = 0
    n_correct_3_com = 0
    n_correct_1_dif = 0
    n_correct_2_dif = 0
    n_correct_3_dif = 0
    correct_all = torch.zeros(5)
    correct_MG = torch.zeros(5)
    correct_MRI = torch.zeros(5)
    
    dataloader_iterator1 = iter(train_loader1)
    dataloader_iterator2 = iter(train_loader2)

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        #target = label.cuda()
        images1, label1 = next(dataloader_iterator1)
        images2, label2 = next(dataloader_iterator2)

        #try:
           # images1, label1 = next(dataloader_iterator1)
       # except StopIteration:
            #dataloader_iterator1 = iter(train_loader1)
            #images1, label1 = next(dataloader_iterator1)

        #try:
            #images2, label2 = next(dataloader_iterator2)
        #except StopIteration:
            #dataloader_iterator2 = iter(train_loader2)
            #images2, label2 = next(dataloader_iterator2)

        input1 = images1.cuda()
        input2 = images2.cuda()
        target1 = label1.cuda()
        target2 = label2.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()

        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            #input_combine = torch.cat([input, input1, input2], dim =0)
            output1, min_distances, min_distances_com, output2, output3 = model(input, input1, input2)

            # compute loss
            label_com = label #torch.cat([label, label1, label2], dim=0)
            label_com_all = torch.cat([label, label1, label2], dim=0).cuda()
            label_dif = torch.cat([label, label1 + 5, label2 + 10], dim=0)
            target_dif = label_dif.cuda()
            target_com = label_com.cuda()
            output_dif = output1.cuda()
            
            size_num = int(output1.size()[0])
            cross_entropy = torch.nn.functional.cross_entropy(output1, target_com)   
            cross_entropy_dif = torch.nn.functional.cross_entropy(output2, target_com)   
            cross_entropy_com = torch.nn.functional.cross_entropy(output3, label_com_all)   

            if class_specific:
                if prototype_activation_function == 'log':
                    max_dist = (model.module.prototype_shape[1] * multi_sourece_num
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])
                else:
                    max_dist = 1

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                #print(min_distances.size())
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity_dif[:,label_dif]).cuda()
                prototypes_of_correct_class_combine = prototypes_of_correct_class[0:size_num,:] + prototypes_of_correct_class[size_num:2*size_num,:] + prototypes_of_correct_class[2*size_num:3*size_num,:] 

                
                #prototypes_of_correct_class_cat =  torch.t(torch.cat([prototypes_of_correct_class[0:size_num,:],prototypes_of_correct_class[size_num:2*size_num,:],prototypes_of_correct_class[2*size_num:3*size_num,:]],dim=0))
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                #prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                #inverted_distances_to_nontarget_prototypes, _ = \
                    #torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                #separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
                
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class_combine.repeat(3,1)
                #prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)


                #com_prototype
                max_dist_com = max_dist

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class_com = torch.t(model.module.prototype_class_identity[:, label_com_all]).cuda()
                inverted_distances_com, _ = torch.max((max_dist_com - min_distances_com) * prototypes_of_correct_class_com, dim=1)
                cluster_cost_com = torch.mean(max_dist_com - inverted_distances_com)

                # calculate separation cost
                prototypes_of_wrong_class_com = 1 - prototypes_of_correct_class_com
                inverted_distances_to_nontarget_prototypes_com, _ = \
                    torch.max((max_dist_com - min_distances_com) * prototypes_of_wrong_class_com, dim=1)
                separation_cost_com = torch.mean(max_dist_com - inverted_distances_to_nontarget_prototypes_com)

                # calculate avg cluster cost
                avg_separation_cost_com = \
                    torch.sum(min_distances_com * prototypes_of_wrong_class_com, dim=1) / torch.sum(prototypes_of_wrong_class_com,
                                                                                            dim=1)
                avg_separation_cost_com = torch.mean(avg_separation_cost_com)


                l2 = (model.module.prototype_vectors_channel_com).norm(p=1) + (model.module.prototype_vectors_channel).norm(p=1)
                if use_l1_mask:
                    identity_temp = model.module.prototype_class_identity_dif.cuda()
                    #identity_dif = torch.cat([identity_temp[:,0:5], identity_temp[:,5:10], identity_temp[:,10:15]], dim=0)
                    identity_dif = identity_temp[:,0:5] + identity_temp[:,5:10] + identity_temp[:,10:15]
                    identity_com = model.module.prototype_class_identity.cuda()
                    #expand_identity_com =identity_com.repeat(1, 3)
                    #de_com_identity = identity_com[:,0:6] + identity_com[:,6:12] + identity_com[:,12:18]
                    #de_dif_identity = identity_dif #identity_dif[:,0:6] + identity_dif[:,6:12] + identity_dif[:,12:18]
                    identity = torch.cat([identity_dif.repeat(3,1),  identity_com.repeat(3,1)],dim=0)
                    
                    l1_mask = 1 - torch.t(identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                    #l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    #l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output_dif.data, 1)
            _, predicted_dif = torch.max(output2.data, 1)
            _, predicted_com = torch.max(output3.data, 1)
            
            target = target_com
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()
            
            total_cross_entropy_dif += cross_entropy_dif.item()
            total_cross_entropy_com += cross_entropy_com.item()
            total_cluster_cost_com += cluster_cost_com.item()
            total_separation_cost_com += separation_cost_com.item()
            total_avg_separation_cost_com += avg_separation_cost_com.item()
            
            temp1 = len(label)
            temp2 = temp1 + len(label1)
            temp3 = temp2 + len(label2)
            #if train_data1  < train_data1_num :
            n_correct_1 += (predicted[0:temp1] == target[0:temp1]).sum().item()
            n_correct_1_dif += (predicted_dif[0:temp1] == target_com[0:temp1]).sum().item()
            n_correct_1_com += (predicted_com[0:temp1] == target_com[0:temp1]).sum().item()
                #train_data1 += len(label)

            #if  train_data2  < train_data2_num :
            #n_correct_2 += (predicted[0:temp1] == target_com[0:temp1]).sum().item()
            #n_correct_2_dif += (predicted_dif[0:temp1] == target_com[0:temp1]).sum().item()
            #n_correct_2_com += (predicted_com[temp1:2*temp1] == target_com[0:temp1]).sum().item()
                #train_data2 += len(label1)

            #if train_data3 < train_data3_num :
            #n_correct_3 += (predicted[0:temp1] == target[0:temp1]).sum().item()
            #n_correct_3_dif += (predicted_dif[0:temp1] == target_com[0:temp1]).sum().item()
            #n_correct_3_com += (predicted_com[2*temp1:3*temp1] == target_com[0:temp1]).sum().item()
                #train_data3 += len(label2)
            for i in range(5):
                correct_all[i] += ((predicted[0:temp1] == target[0:temp1]) * (predicted[0:temp1] == i)).sum().item()
                correct_MG[i] += ((predicted_com[temp1:2*temp1] == target_com[0:temp1]) * (predicted_com[temp1:2*temp1] == i)).sum().item()
                correct_MRI[i] += ((predicted_com[2*temp1:3*temp1] == target_com[0:temp1]) * (predicted_com[2*temp1:3*temp1] == i)).sum().item()
            
            
            
            
            #print( n_correct_1,n_correct_2,n_correct_3, )

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          #+coefs['crs_ent_com'] * cross_entropy_com
                          #+coefs['crs_ent_dif'] * cross_entropy_dif
                          + coefs['clst_dif'] * cluster_cost
                          + coefs['sep_dif'] * separation_cost
                          + coefs['l1'] * l1
                          +  coefs['clst_com'] * cluster_cost_com
                          +  coefs['sep_com'] * separation_cost_com
                          +  coefs['l2'] * l2
                            )
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        #del output
        del predicted
        del min_distances



    end = time.time()
    
    MG_correct = 0
    MRI_correct = 0
    MG_correct_com = 0
    MRI_correct_com = 0
    
    MG_correct = (correct_all*MG_data).sum().item()
    MRI_correct = (correct_all*MRI_data).sum().item()
    MG_correct_com = (correct_MG*MG_data).sum().item()
    MRI_correct_com = (correct_MRI*MRI_data).sum().item()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcross ent_dif: \t{0}'.format(total_cross_entropy_dif / n_batches))
    log('\tcross ent_com: \t{0}'.format(total_cross_entropy_com / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\tcluster_com: \t{0}'.format(total_cluster_cost_com / n_batches))
    log('\tseparation_com:\t{0}'.format(total_separation_cost_com / n_batches))
    log('\tavg separation_com:\t{0}'.format(total_avg_separation_cost_com / n_batches))
    log('\tUS: \t\t{0}%'.format(n_correct_1 / train_data1_num * 100))
    log('\tMG: \t\t{0}%'.format(MG_correct / (train_data1_num *1597 / 9814) * 100))
    log('\tMRI: \t\t{0}%'.format(MRI_correct / (train_data1_num *1854 / 9814) * 100))
    
    log('\tUS_dif: \t\t{0}%'.format(n_correct_1_dif / train_data1_num * 100))
    log('\tUS_com: \t\t{0}%'.format(n_correct_1_com / train_data1_num * 100))
    log('\tMG_com: \t\t{0}%'.format(MG_correct_com / (train_data1_num *1597 / 9814) * 100))
    log('\tMRI_com: \t\t{0}%'.format(MRI_correct_com / (train_data1_num *1854 / 9814) * 100))

    
    #log('\tUS_com: \t\t{0}%'.format(n_correct_1_com / train_data1_num * 100))
    #log('\tMG_com: \t\t{0}%'.format(n_correct_2_com / train_data1_num * 100))
    #log('\tMRI_com: \t\t{0}%'.format(n_correct_3_com / train_data1_num * 100))
    #log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    #p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    #with torch.no_grad():
        #p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    #log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train(model, dataloader,train_loader1, train_loader2, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader,train_loader1=train_loader1,train_loader2 = train_loader2, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, test_loader1, test_loader2, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, train_loader1=test_loader1,train_loader2 = test_loader2, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features_com.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_com.parameters():
        p.requires_grad = False
    for p in model.module.features_dif.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_dif.parameters():
        p.requires_grad = False
        
        
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    for p in model.module.last_layer_com.parameters():
        p.requires_grad = True
    for p in model.module.last_layer_dif.parameters():
        p.requires_grad = True
        

    model.module.prototype_vectors_com.requires_grad = False

    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features_com.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_com.parameters():
        p.requires_grad = True
    for p in model.module.features_dif.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers_dif.parameters():
        p.requires_grad = True
        
        
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    for p in model.module.last_layer_com.parameters():
        p.requires_grad = True
    for p in model.module.last_layer_dif.parameters():
        p.requires_grad = True
        

    model.module.prototype_vectors_com.requires_grad = True

    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features_com.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers_com.parameters():
        p.requires_grad = True
    for p in model.module.features_dif.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers_dif.parameters():
        p.requires_grad = True

        
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    for p in model.module.last_layer_com.parameters():
        p.requires_grad = True
    for p in model.module.last_layer_dif.parameters():
        p.requires_grad = True
    

    model.module.prototype_vectors_com.requires_grad = True
    log('\tjoint')
