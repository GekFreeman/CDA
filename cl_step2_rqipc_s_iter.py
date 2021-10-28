from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader_sampler as data_loader
import save_pred
import resnet_step2_cs as models
from net_utils import save_net, load_net,save_checkpoint, save_checkpoints,euclid_dist,class_loss
from net_utils import semantic_loss_calc as semantic_loss_c
#from centroids import Centroids
import pdb
from tqdm import tqdm,trange
import heapq
from collections import defaultdict
from heapq import * 
import time
#from tsne import get_data_before,get_data_after, plot_embedding_2D,tsne_2D,tsne_3D
num_gpu=[0,1,2,3]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in num_gpu)
device_ids=num_gpu

# Training settings
batch_size =64
batch_size_tgt= 64#128 #64#
iteration = 20000#00#000#00#000
iteration_point = 10000#00
lr = 0.01
momentum = 0.9
cuda = True
seed = 8
log_interval = 50
l2_decay = 5e-4
root_path = "/userhome/chengyl/UDA/multi-source/dataset/DomainNet_test/"
output_dir="./iteroutput"
"""
source1_name ="clipart_test"
source2_name ="clipart_test"
source3_name ="clipart_test"
source4_name = "clipart_test"
source5_name ="clipart_test"

target_name = "sketch_test"
"""
##############
source1_name ="real" 
source2_name ="quickdraw"
source3_name ="infograph"
source4_name = "painting"
source5_name ="clipart"

target_name = "sketch"
###############

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

source1_loader1, source1_loader2= data_loader.load_training_source1(root_path, source1_name, batch_size, kwargs)
source2_loader1, source2_loader2= data_loader.load_training_source2(root_path, source2_name, batch_size, kwargs)
source3_loader1, source3_loader2= data_loader.load_training_source3(root_path, source3_name, batch_size, kwargs)
source4_loader1, source4_loader2= data_loader.load_training_source4(root_path, source4_name, batch_size, kwargs)
source5_loader1, source5_loader2= data_loader.load_training_source5(root_path, source5_name, batch_size, kwargs)

target_train_loader1,target_train_loader2 = data_loader.load_training_target(root_path, target_name, batch_size_tgt,kwargs)
source_test_loader = data_loader.load_testing(root_path, source2_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size_tgt, kwargs)


def train(model):
    source1_iter1 = iter(source1_loader1)   
    source1_iter2 = iter(source1_loader2)
    target_iter1 = iter(target_train_loader1)
    target_iter2 = iter(target_train_loader2)
    target_iter1_index = iter(target_train_loader1.batch_sampler)
    target_iter2_index = iter(target_train_loader2.batch_sampler)
    alpha = len(source1_loader2) / len(source1_loader1)
    correct = 0
    s_correct=0
    savepath='./heapq_target_rqipc_s_iter.txt'
    dic1=defaultdict(list)    
    model=nn.DataParallel(model)
    for i in range(1, iteration + 1):
        semantic_loss=0
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate: ", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.module.sharedNet.parameters()},
            {'params': model.module.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.sonnet2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        try:
            source_data1, source_label1 = source1_iter1.next()
            source_data2, source_label2 = source1_iter2.next()
        except Exception as err:
            source1_iter1 = iter(source1_loader1)
            source_data1, source_label1 = source1_iter1.next()
            
            source1_iter2 = iter(source1_loader2)
            source_data2, source_label2 = source1_iter2.next()
        try:
            target_data1,_  = target_iter1.next()
            target_index1=target_iter1_index.__next__()
            target_data2,_= target_iter2.next()
            target_index2=target_iter2_index.__next__()
        except Exception as err:
            target_iter1 = iter(target_train_loader1)
            target_data1,_ = target_iter1.next()
            target_iter2 = iter(target_train_loader2)
            target_data2,_ = target_iter2.next()
                
            target_iter1_index = iter(target_train_loader1.batch_sampler)
            target_index1=target_iter1_index.__next__()
            target_iter2_index = iter(target_train_loader2.batch_sampler)
            target_index2=target_iter2_index.__next__()
        if cuda:
            source_data1, source_label1 = source_data1.cuda(), source_label1.cuda()
            source_data2, source_label2 = source_data2.cuda(), source_label2.cuda()
            target_data1 = target_data1.cuda()
            target_data2= target_data2.cuda()
        source_data1, source_label1 = Variable(source_data1), Variable(source_label1)
        source_data2, source_label2 = Variable(source_data2), Variable(source_label2)
        target_data1 = Variable(target_data1)
        target_data2 = Variable(target_data2)
        if i<=iteration_point:
            optimizer.zero_grad()
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
            cls_loss,sim_loss,mmd_loss,pred_tgt1,pred_tgt2 = model(source_data1,source_label1, source_data2,source_label2,target_data1 ,target_data2,step = 1,alpha=alpha)
            dic1=save_pred.save_target(dic1,pred_tgt1,target_index1)
            dic1=save_pred.save_target(dic1,pred_tgt2,target_index1)
            loss = cls_loss + gamma*sim_loss#+gamma*mmd_loss
            loss.mean().backward()
            optimizer.step()
            if i % log_interval == 0:
                print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tsim_Loss: {:.6f}\t'.format(i, 100. * i / iteration, loss.mean().item(),    cls_loss.mean().item(), sim_loss.mean().item()))
                print('mmd_loss:{}'.format(mmd_loss.mean().item()))
            if i % (log_interval * 10) == 0:      
                t_correct = test(model,1)
               # s_correct=test_source(model,1)
               # print('again test source1:{}'.format(s_correct))
                if t_correct > correct:
                    correct = t_correct
                    save_pred.save_results(dic1,savepath)
                    save_name = os.path.join(output_dir, 'cl_step2_iter_res50_{}_{}_{}_{}_{}_to_{}.pth'.format(source1_name, source2_name, source3_name,source4_name,source5_name,target_name))
                    save_checkpoint({'model': model.state_dict()}, save_name)
                    print('save model: {}'.format(save_name))
                    
                print(source1_name, source2_name,source3_name,source4_name,source5_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n") 
        else:
            la_thresh=0.5
            optimizer.zero_grad()
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
            tgt_data1,tgt_index1,tgt_pse1,tgt_data2, tgt_index2,tgt_pse2=data_loader.load_sameclass_target(root_path, target_name,dic1,source_data1, source_label1, source_data2,source_label2,batch_size)
            if cuda:
               # print(source_label1)
                target_data1 = tgt_data1.cuda()
                target_data2= tgt_data2.cuda()  
            tgt_data1 = Variable(target_data1)
            tgt_data2 = Variable(target_data2)
                
            cls_loss,mmd_loss,pred_tgt1,pred_tgt2 = model(source_data1,source_label1, source_data2,source_label2,tgt_data1,tgt_data2,step = 2,alpha=alpha, label_tgt1=tgt_pse1, label_tgt2=tgt_pse2)
            
            dic1=save_pred.save_target(dic1,pred_tgt1,tgt_index1)
            dic1=save_pred.save_target(dic1,pred_tgt2,tgt_index2)
            loss=cls_loss+gamma*mmd_loss
            loss.mean().backward()
            optimizer.step()
            if i % log_interval == 0:
                print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tmmd_loss: {:.6f}\tcls_Loss: {:.6f}\t'.format(i, 100. * i / iteration, loss.mean().item(), mmd_loss.mean().item(),cls_loss.mean().item()))
               
            if i % (log_interval * 10) == 0:      
                t_correct = test(model,1)
              #  s_correct=test_source(model,1)
             #   print('again test source1:{}'.format(s_correct))
                if t_correct > correct:
                    save_pred.save_results(dic1,savepath)
                    correct = t_correct
                    save_name = os.path.join(output_dir, 'cl_step2_iter_res50_{}_{}_{}_{}_{}_to_{}.pth'.format(source1_name, source2_name, source3_name,source4_name,source5_name,target_name))
                    save_checkpoint({'model': model.state_dict()}, save_name)
                    print('save model: {}'.format(save_name))
                    
                print(source1_name, source2_name, source3_name,source4_name,source5_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")
           
               
    ########连续学习-source2#################
    source2_iter1 = iter(source2_loader1)
    source2_iter2 = iter(source2_loader2)
    target_iter1 = iter(target_train_loader1)
    target_iter2 = iter(target_train_loader2)
    alpha = len(source2_loader2) / len(source2_loader1)
    target_iter1_index = iter(target_train_loader1.batch_sampler)
    target_iter2_index = iter(target_train_loader2.batch_sampler)
    for i in range(1, iteration + 1):
        
        semantic_loss=0
        src_inputs = []
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate: ", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.module.sharedNet.parameters()},
            {'params': model.module.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.sonnet2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        try:
            source_data1, source_label1 = source2_iter1.next()
            source_data2, source_label2= source2_iter2.next()
        except Exception as err:
            source2_iter1 = iter(source2_loader1)
            source_data1, source_label1= source2_iter1.next()
            
            source2_iter2 = iter(source2_loader2)
            source_data2, source_label2 = source2_iter2.next()
        try:
            target_data1,_  = target_iter1.next()
            target_index1=target_iter1_index.__next__()
            target_data2,_= target_iter2.next()
            target_index2=target_iter2_index.__next__()
        except Exception as err:
            target_iter1 = iter(target_train_loader1)
            target_data1,_ = target_iter1.next()
            target_iter2 = iter(target_train_loader2)
            target_data2,_ = target_iter2.next()
                
            target_iter1_index = iter(target_train_loader1.batch_sampler)
            target_index1=target_iter1_index.__next__()
            target_iter2_index = iter(target_train_loader2.batch_sampler)
            target_index2=target_iter2_index.__next__()
        if cuda:
               # print(source_label1)
            source_data1, source_label1 = source_data1.cuda(), source_label1.cuda()
            source_data2, source_label2 = source_data2.cuda(), source_label2.cuda()
            target_data1 = target_data1.cuda()
            target_data2= target_data2.cuda()
        source_data1, source_label1 = Variable(source_data1), Variable(source_label1)
        source_data2, source_label2 = Variable(source_data2), Variable(source_label2)
        target_data1 = Variable(target_data1)
        target_data2 = Variable(target_data2)
        
        if i<=iteration_point:
            optimizer.zero_grad()
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
            cls_loss,sim_loss,mmd_loss,pred_tgt1,pred_tgt2 = model(source_data1,source_label1, source_data2,source_label2,target_data1 ,target_data2,step = 1,alpha=alpha)
            dic1=save_pred.save_target(dic1,pred_tgt1,target_index1)
            dic1=save_pred.save_target(dic1,pred_tgt2,target_index1)
            loss = cls_loss + gamma*sim_loss#+gamma*mmd_loss
            loss.mean().backward()
            optimizer.step()
            if i % log_interval == 0:
                print('Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tsim_Loss: {:.6f}\t'.format(i, 100. * i / iteration, loss.mean().item(),    cls_loss.mean().item(), sim_loss.mean().item()))
                print('mmd_loss:{}'.format(mmd_loss.mean().item()))
            if i % (log_interval * 10) == 0:      
                t_correct = test(model,1)
               # s_correct=test_source(model,1)
               # print('again test source1:{}'.format(s_correct))
                if t_correct > correct:
                    save_pred.save_results(dic1,savepath)
                    correct = t_correct
                    save_name = os.path.join(output_dir, 'cl_step2_iter_res50_{}_{}_{}_{}_{}_to_{}.pth'.format(source1_name, source2_name, source3_name,source4_name,source5_name,target_name))
                    save_checkpoint({'model': model.state_dict()}, save_name)
                    print('save model: {}'.format(save_name))
                   
                print(source1_name, source2_name,source3_name, source4_name,source5_name,"to", target_name, "%s max correct:" % target_name, correct.item(), "\n") 
        else:
            la_thresh=0.5
            optimizer.zero_grad()
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
            tgt_data1,tgt_index1,tgt_pse1,tgt_data2, tgt_index2,tgt_pse2=data_loader.load_sameclass_target(root_path, target_name,dic1,source_data1, source_label1, source_data2,source_label2,batch_size)
            if cuda:
               # print(source_label1)
                target_data1 = tgt_data1.cuda()
                target_data2= tgt_data2.cuda()         
            tgt_data1 = Variable(target_data1)
            tgt_data2 = Variable(target_data2)
           
                
            cls_loss,mmd_loss,pred_tgt1,pred_tgt2 = model(source_data1,source_label1, source_data2,source_label2,tgt_data1,tgt_data2,step = 2,alpha=alpha, label_tgt1=tgt_pse1, label_tgt2=tgt_pse2)
            dic1=save_pred.save_target(dic1,pred_tgt1,tgt_index1)
            dic1=save_pred.save_target(dic1,pred_tgt2,tgt_index2)
            loss=cls_loss+gamma*mmd_loss
            loss.mean().backward()
            optimizer.step()
            if i % log_interval == 0:
                print('Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tmmd_loss: {:.6f}\tcls_Loss: {:.6f}\t'.format(i, 100. * i / iteration, loss.mean().item(), mmd_loss.mean().item(),cls_loss.mean().item()))
               
            if i % (log_interval * 10) == 0:      
                t_correct = test(model,1)
              #  s_correct=test_source(model,1)
              #  print('again test source1:{}'.format(s_correct))
                if t_correct > correct:
                    save_pred.save_results(dic1,savepath)
                    correct = t_correct
                    save_name = os.path.join(output_dir, 'cl_step2_iter_res50_{}_{}_{}_{}_{}_to_{}.pth'.format(source1_name, source2_name, source3_name,source4_name,source5_name,target_name))
                    save_checkpoint({'model': model.state_dict()}, save_name)
                    print('save model: {}'.format(save_name))
                    
                
                print(source1_name, source2_name,source3_name, source4_name,source5_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")
    ########连续学习-source3#################
    source3_iter1 = iter(source3_loader1)   
    source3_iter2 = iter(source3_loader2)
    target_iter1 = iter(target_train_loader1)
    target_iter2 = iter(target_train_loader2)
    alpha = len(source3_loader2) / len(source3_loader1)
    target_iter1_index = iter(target_train_loader1.batch_sampler)
    target_iter2_index = iter(target_train_loader2.batch_sampler)
    for i in range(1, iteration + 1):
        semantic_loss=0
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate: ", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.module.sharedNet.parameters()},
            {'params': model.module.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.sonnet2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        try:
            source_data1, source_label1 = source3_iter1.next()
            source_data2, source_label2= source3_iter2.next()
        except Exception as err:
            source3_iter1 = iter(source3_loader1)
            source_data1, source_label1= source3_iter1.next()
            
            source3_iter2 = iter(source3_loader2)
            source_data2, source_label2 = source3_iter2.next()
        try:
            target_data1,_  = target_iter1.next()
            target_index1=target_iter1_index.__next__()
            target_data2,_= target_iter2.next()
            target_index2=target_iter2_index.__next__()
        except Exception as err:
            target_iter1 = iter(target_train_loader1)
            target_data1,_ = target_iter1.next()
            target_iter2 = iter(target_train_loader2)
            target_data2,_ = target_iter2.next()
                
            target_iter1_index = iter(target_train_loader1.batch_sampler)
            target_index1=target_iter1_index.__next__()
            target_iter2_index = iter(target_train_loader2.batch_sampler)
            target_index2=target_iter2_index.__next__()
        if cuda:
               # print(source_label1)
            source_data1, source_label1 = source_data1.cuda(), source_label1.cuda()
            source_data2, source_label2 = source_data2.cuda(), source_label2.cuda()
            target_data1 = target_data1.cuda()
            target_data2= target_data2.cuda()
        source_data1, source_label1 = Variable(source_data1), Variable(source_label1)
        source_data2, source_label2 = Variable(source_data2), Variable(source_label2)
        target_data1 = Variable(target_data1)
        target_data2 = Variable(target_data2)
        
        if i<=iteration_point:
            optimizer.zero_grad()
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
            cls_loss,sim_loss,mmd_loss,pred_tgt1,pred_tgt2 = model(source_data1,source_label1, source_data2,source_label2,target_data1 ,target_data2,step = 1,alpha=alpha, label_tgt1=tgt_pse1, label_tgt2=tgt_pse2)
            dic1=save_pred.save_target(dic1,pred_tgt1,target_index1)
            dic1=save_pred.save_target(dic1,pred_tgt2,target_index1)
            loss = cls_loss + gamma*sim_loss#+gamma*mmd_loss
            loss.mean().backward()
            optimizer.step()
            if i % log_interval == 0:
                print('Train source3 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tsim_Loss: {:.6f}\t'.format(i, 100. * i / iteration, loss.mean().item(),    cls_loss.mean().item(), sim_loss.mean().item()))
                print('mmd_loss:{}'.format(mmd_loss.mean().item()))
            if i % (log_interval * 10) == 0:      
                t_correct = test(model,1)
               # s_correct=test_source(model,1)
              #  print('again test source1:{}'.format(s_correct))
                if t_correct > correct:
                    correct = t_correct
                    save_name = os.path.join(output_dir, 'cl_step2_iter_res50_{}_{}_{}_{}_{}_to_{}.pth'.format(source1_name, source2_name, source3_name,source4_name,source5_name,target_name))
                    save_checkpoint({'model': model.state_dict()}, save_name)
                    print('save model: {}'.format(save_name))
                   
                print(source1_name, source2_name, source3_name, source4_name,source5_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n") 
        else:
            la_thresh=0.5
            optimizer.zero_grad()
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
            tgt_data1,tgt_index1,tgt_pse1,tgt_data2, tgt_index2,tgt_pse2=data_loader.load_sameclass_target(root_path, target_name,dic1,source_data1, source_label1, source_data2,source_label2,batch_size)
            if cuda:
               # print(source_label1)
                target_data1 = tgt_data1.cuda()
                target_data2= tgt_data2.cuda()         
            tgt_data1 = Variable(target_data1)
            tgt_data2 = Variable(target_data2)
             
            cls_loss,mmd_loss,pred_tgt1,pred_tgt2 = model(source_data1,source_label1, source_data2,source_label2,tgt_data1,tgt_data2,step = 2,alpha=alpha)
            dic1=save_pred.save_target(dic1,pred_tgt1,tgt_index1)
            dic1=save_pred.save_target(dic1,pred_tgt2,tgt_index2)
            loss=cls_loss+gamma*mmd_loss
            loss.mean().backward()
            optimizer.step()
            if i % log_interval == 0:
                print('Train source3 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tmmd_loss: {:.6f}\tcls_Loss: {:.6f}\t'.format(i, 100. * i / iteration, loss.mean().item(), mmd_loss.mean().item(),cls_loss.mean().item()))
                
            if i % (log_interval * 10) == 0:      
                t_correct = test(model,1)
              #  s_correct=test_source(model,1)
              #  print('again test source1:{}'.format(s_correct))
                if t_correct > correct:
                    save_pred.save_results(dic1,savepath)
                    correct = t_correct
                    save_name = os.path.join(output_dir, 'cl_step2_iter_res50_{}_{}_{}_{}_{}_to_{}.pth'.format(source1_name, source2_name, source3_name,source4_name,source5_name,target_name))
                    save_checkpoint({'model': model.state_dict()}, save_name)
                    print('save model: {}'.format(save_name))
                print(source1_name, source2_name,source3_name,source4_name,source5_name,  "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")
    ########连续学习-source4444#################
    source4_iter1 = iter(source4_loader1)   
    source4_iter2 = iter(source4_loader2)
    target_iter1 = iter(target_train_loader1)
    target_iter2 = iter(target_train_loader2)
    alpha = len(source4_loader2) / len(source4_loader1)
    target_iter1_index = iter(target_train_loader1.batch_sampler)
    target_iter2_index = iter(target_train_loader2.batch_sampler)
    for i in range(1, iteration + 1):
        semantic_loss=0
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate: ", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.module.sharedNet.parameters()},
            {'params': model.module.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.sonnet2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        try:
            source_data1, source_label1 = source4_iter1.next()
            source_data2, source_label2= source4_iter2.next()
        except Exception as err:
            source4_iter1 = iter(source4_loader1)
            source_data1, source_label1= source4_iter1.next()
            
            source4_iter2 = iter(source4_loader2)
            source_data2, source_label2 = source4_iter2.next()
        try:
            target_data1,_  = target_iter1.next()
            target_index1=target_iter1_index.__next__()
            target_data2,_= target_iter2.next()
            target_index2=target_iter2_index.__next__()
        except Exception as err:
            target_iter1 = iter(target_train_loader1)
            target_data1,_ = target_iter1.next()
            target_iter2 = iter(target_train_loader2)
            target_data2,_ = target_iter2.next()
                
            target_iter1_index = iter(target_train_loader1.batch_sampler)
            target_index1=target_iter1_index.__next__()
            target_iter2_index = iter(target_train_loader2.batch_sampler)
            target_index2=target_iter2_index.__next__()
        if cuda:
               # print(source_label1)
            source_data1, source_label1 = source_data1.cuda(), source_label1.cuda()
            source_data2, source_label2 = source_data2.cuda(), source_label2.cuda()
            target_data1 = target_data1.cuda()
            target_data2= target_data2.cuda()
        source_data1, source_label1 = Variable(source_data1), Variable(source_label1)
        source_data2, source_label2 = Variable(source_data2), Variable(source_label2)
        target_data1 = Variable(target_data1)
        target_data2 = Variable(target_data2)
        
        if i<=iteration_point:
            optimizer.zero_grad()
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
            cls_loss,sim_loss,mmd_loss,pred_tgt1,pred_tgt2 = model(source_data1,source_label1, source_data2,source_label2,target_data1 ,target_data2,step = 1,alpha=alpha)
            dic1=save_pred.save_target(dic1,pred_tgt1,target_index1)
            dic1=save_pred.save_target(dic1,pred_tgt2,target_index1)
            loss = cls_loss + gamma*sim_loss#+gamma*mmd_loss
            loss.mean().backward()
            optimizer.step()
            if i % log_interval == 0:
                print('Train source4 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tsim_Loss: {:.6f}\t'.format(i, 100. * i / iteration, loss.mean().item(),    cls_loss.mean().item(), sim_loss.mean().item()))
                print('mmd_loss:{}'.format(mmd_loss.mean().item()))
            if i % (log_interval * 10) == 0:      
                t_correct = test(model,1)
               # s_correct=test_source(model,1)
              #  print('again test source1:{}'.format(s_correct))
                if t_correct > correct:
                    correct = t_correct
                    save_name = os.path.join(output_dir, 'cl_step2_iter_res50_{}_{}_{}_{}_{}_to_{}.pth'.format(source1_name, source2_name, source3_name,source4_name,source5_name,target_name))
                    save_checkpoint({'model': model.state_dict()}, save_name)
                    print('save model: {}'.format(save_name))
                   
                print(source1_name, source2_name, source3_name,source4_name,source5_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n") 
        else:
            la_thresh=0.5
            optimizer.zero_grad()
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
            tgt_data1,tgt_index1,tgt_pse1,tgt_data2, tgt_index2,tgt_pse2=data_loader.load_sameclass_target(root_path, target_name,dic1,source_data1, source_label1, source_data2,source_label2,batch_size)
            if cuda:
               # print(source_label1)
                target_data1 = tgt_data1.cuda()
                target_data2= tgt_data2.cuda()             
            tgt_data1 = Variable(target_data1)
            tgt_data2 = Variable(target_data2)

            cls_loss,mmd_loss,pred_tgt1,pred_tgt2 = model(source_data1,source_label1, source_data2,source_label2,tgt_data1,tgt_data2,step = 2,alpha=alpha, label_tgt1=tgt_pse1, label_tgt2=tgt_pse2)
          #  cls_loss =(cls1_loss+cls2_loss)*0.5# 0.5*(cls_loss1+sim_loss1)+cls_loss #+ cls_loss1#sim_loss1
            dic1=save_pred.save_target(dic1,pred_tgt1,tgt_index1)
            dic1=save_pred.save_target(dic1,pred_tgt2,tgt_index2)
            loss=cls_loss+gamma*mmd_loss
            loss.mean().backward()
            optimizer.step()
            if i % log_interval == 0:
                print('Train source4 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tmmd_loss: {:.6f}\tcls_Loss: {:.6f}\t'.format(i, 100. * i / iteration, loss.mean().item(), mmd_loss.mean().item(),cls_loss.mean().item()))
                
            if i % (log_interval * 10) == 0:      
                t_correct = test(model,1)
              #  s_correct=test_source(model,1)
              #  print('again test source1:{}'.format(s_correct))
                if t_correct > correct:
                    correct = t_correct
                    save_name = os.path.join(output_dir, 'cl_step2_iter_res50_{}_{}_{}_{}_{}_to_{}.pth'.format(source1_name, source2_name, source3_name,source4_name,source5_name,target_name))
                    save_checkpoint({'model': model.state_dict()}, save_name)
                    print('save model: {}'.format(save_name))
                    
                
                print(source1_name, source2_name,source3_name, source4_name,source5_name,"to", target_name, "%s max correct:" % target_name, correct.item(), "\n")
    ########连续学习-source5555#################
    source5_iter1 = iter(source5_loader1)   
    source5_iter2 = iter(source5_loader2)
    target_iter1 = iter(target_train_loader1)
    target_iter2 = iter(target_train_loader2)
    alpha = len(source5_loader2) / len(source5_loader1)
    target_iter1_index = iter(target_train_loader1.batch_sampler)
    target_iter2_index = iter(target_train_loader2.batch_sampler)
    for i in range(1, iteration + 1):
        
        semantic_loss=0
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate: ", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.module.sharedNet.parameters()},
            {'params': model.module.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.module.sonnet2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

        try:
            source_data1, source_label1 = source5_iter1.next()
            source_data2, source_label2= source5_iter2.next()
        except Exception as err:
            source5_iter1 = iter(source5_loader1)
            source_data1, source_label1= source5_iter1.next()
            
            source5_iter2 = iter(source5_loader2)
            source_data2, source_label2 = source5_iter2.next()
        try:
            target_data1,_  = target_iter1.next()
            target_index1=target_iter1_index.__next__()
            target_data2,_= target_iter2.next()
            target_index2=target_iter2_index.__next__()
        except Exception as err:
            target_iter1 = iter(target_train_loader1)
            target_data1,_ = target_iter1.next()
            target_iter2 = iter(target_train_loader2)
            target_data2,_ = target_iter2.next()
                
            target_iter1_index = iter(target_train_loader1.batch_sampler)
            target_index1=target_iter1_index.__next__()
            target_iter2_index = iter(target_train_loader2.batch_sampler)
            target_index2=target_iter2_index.__next__()
        if cuda:
               # print(source_label1)
            source_data1, source_label1 = source_data1.cuda(), source_label1.cuda()
            source_data2, source_label2 = source_data2.cuda(), source_label2.cuda()
            target_data1 = target_data1.cuda()
            target_data2= target_data2.cuda()
        source_data1, source_label1 = Variable(source_data1), Variable(source_label1)
        source_data2, source_label2 = Variable(source_data2), Variable(source_label2)
        target_data1 = Variable(target_data1)
        target_data2 = Variable(target_data2)
        
        if i<=iteration_point:
            optimizer.zero_grad()
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
            cls_loss,sim_loss,mmd_loss,pred_tgt1,pred_tgt2 = model(source_data1,source_label1, source_data2,source_label2,target_data1 ,target_data2,step = 1,alpha=alpha)
            dic1=save_pred.save_target(dic1,pred_tgt1,target_index1)
            dic1=save_pred.save_target(dic1,pred_tgt2,target_index2)
            loss = cls_loss + gamma*sim_loss#+gamma*mmd_loss
            loss.mean().backward()
            optimizer.step()
            if i % log_interval == 0:
                print('Train source5 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tsim_Loss: {:.6f}\t'.format(i, 100. * i / iteration, loss.mean().item(),    cls_loss.mean().item(), sim_loss.mean().item()))
                print('mmd_loss:{}'.format(mmd_loss.mean().item()))
            if i % (log_interval * 10) == 0:      
                t_correct = test(model,1)
               # s_correct=test_source(model,1)
              #  print('again test source1:{}'.format(s_correct))
                if t_correct > correct:
                    save_pred.save_results(dic1,savepath)
                    correct = t_correct
                    save_name = os.path.join(output_dir, 'cl_step2_iter_res50_{}_{}_{}_{}_{}_to_{}.pth'.format(source1_name, source2_name, source3_name,source4_name,source5_name,target_name))
                    save_checkpoint({'model': model.state_dict()}, save_name)
                    print('save model: {}'.format(save_name))
                   
                print(source1_name, source2_name, source3_name, source4_name,source5_name,"to", target_name, "%s max correct:" % target_name, correct.item(), "\n") 
        else:
            la_thresh=0.5
            optimizer.zero_grad()
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
            tgt_data1,tgt_index1,tgt_pse1,tgt_data2, tgt_index2,tgt_pse2=data_loader.load_sameclass_target(root_path, target_name,dic1,source_data1, source_label1, source_data2,source_label2,batch_size)
            if cuda:
               # print(source_label1)
                target_data1 = tgt_data1.cuda()
                target_data2= tgt_data2.cuda()         
            tgt_data1 = Variable(target_data1)
            tgt_data2 = Variable(target_data2)
            
                
            cls_loss,mmd_loss,pred_tgt1,pred_tgt2 = model(source_data1,source_label1, source_data2,source_label2,tgt_data1,tgt_data2,step = 2,alpha=alpha, label_tgt1=tgt_pse1, label_tgt2=tgt_pse2)
            dic1=save_pred.save_target(dic1,pred_tgt1,tgt_index1)
            dic1=save_pred.save_target(dic1,pred_tgt2,tgt_index2)
            loss=cls_loss+gamma*mmd_loss
            loss.mean().backward()
            optimizer.step()
            if i % log_interval == 0:
                print('Train source5 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tmmd_loss: {:.6f}\tcls_Loss: {:.6f}\t'.format(i, 100. * i / iteration, loss.mean().item(), mmd_loss.mean().item(),cls_loss.mean().item()))
                
            if i % (log_interval * 10) == 0:      
                t_correct = test(model,1)
              #  s_correct=test_source(model,1)
              #  print('again test source1:{}'.format(s_correct))
                if t_correct > correct:
                    save_pred.save_results(dic1,savepath)
                    correct = t_correct
                    save_name = os.path.join(output_dir, 'cl_step2_iter_res50_{}_{}_{}_{}_{}_to_{}.pth'.format(source1_name, source2_name, source3_name,source4_name,source5_name,target_name))
                    save_checkpoint({'model': model.state_dict()}, save_name)
                    print('save model: {}'.format(save_name))
                    
                
                print(source1_name, source2_name,source3_name,source4_name,source5_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")    

def test(model,source):
    model.eval()
    model.cuda()
    test_loss = 0
    s=source
    num=0
    correct = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            num+=1
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1 = model(data)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)

            pred = pred1 

             #2021/03/04
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
        num=num*batch_size_tgt#len(target_test_loader.dataset)
        test_loss /= num#len(target_test_loader.dataset)
       # from __future__ import division
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, 1.0*correct, num,
            100.0 * correct / (1.0*num)))
       # print('\nsource1 accnum {}'.format(correct1))
    return correct
"""def test_source(model,source):
    model.eval()
    model.cuda()
    test_loss = 0
    s=source
    num=0
    correct = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for data, target in source_test_loader:
            num+=1
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1 = model(data)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)

            pred = pred1 

             #2021/03/04
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
        num=batch_size*num
        test_loss /= num#len(target_test_loader.dataset)
       # from __future__ import division
        print(target_name, '\nTest source1 set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, 1.0*correct, num,
            100. * correct / num))
        print('\nsource1 accnum {}'.format(correct1))
    return correct"""
if __name__ == '__main__':
    model = models.MFSAN(num_classes=100)
    #print('with simi')
    #print('models1-resnetdsbn')
    #print('cl_mfsandsbn')
    if cuda:
        model.cuda()
    train(model)
   # test(model,3)
    #print('begin 2d-tsne')
   # data_all,label_all=tsne(model)
   # fig=tsne_2D(data_all,label_all)
   # print('begin 3d-tsne')
    #fig=tsne_3D(data_all,label_all)