import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
#from model.utils.config import cfg
#from model.roi_crop.functions.roi_crop import RoICropFunction
#import cv2
import pdb
import random
from collections import defaultdict
import re
import os

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def load_best_checkpoints(load_dir,model):
   # pattern = r'^best_((?!optimizer_)(?!centroids_))(?P<model_name>.+)\+(?P<jitter>.+)\+i(?P<in_features>.+)_' + \
           #   r'(?P<source_dataset>.+)2(?P<target_dataset>.+)\.pth$'
    best_model_ckpt_filename = None
    
    loaded_model = torch.load(load_dir)
    state_dict = torch.load(load_dir)
        # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
            name = k#.replace(".module","")  # remove `module.`
            new_state_dict[name] = v
        # load params
    model.load_state_dict(new_state_dict,strict=False)

    best_optimizer = 'optimizer'# + best_model_ckpt_filename[5:]
    try:
        loaded_optimizer = torch.load(load_dir)
    except FileNotFoundError:
        logger.warning('best optimizer is not found. Set to default Optimizer states!')
        loaded_optimizer = {}

       # if args.sm_loss:
    best_centroids = 'centroids' #+ best_model_ckpt_filename[5:]
    try:
        loaded_centorids = torch.load(load_dir)
    except FileNotFoundError:
        logger.warning('best centroids is not found. Set to default centroids!')
        loaded_centorids = {}

    return model, loaded_optimizer, loaded_centorids

def save_checkpoint(state, filename):
    torch.save(state, filename)
#def save_checkpoints(save_dir, i_iter, model_dict, optimizer_dict, centroids_dict):
def save_checkpoints(save_dir, model_dict, optimizer_dict, centroids_dict):
    save_path = save_dir  
  #  model_dict.update({'iteration': i_iter})
    torch.save(model_dict, save_path)
    if optimizer_dict:
        #optimizer_dict.update({'iteration': i_iter})
        torch.save(optimizer_dict, save_path)
    if centroids_dict:
        #centroids_dict.update({'iteration': i_iter})
        torch.save(centroids_dict, save_path)
def save_centroids(centroids_dict1,centroids_dict2): 
    centroids1=centroids_dict1.data.detach().cpu().numpy()
    centroids1=centroids1.reshape(-1)
    centroids1_path='output/scentroids1.txt'  
    contents1=(["{}".format(centroids1)+'\n'])
    with open(centroids1_path, 'a+') as f:
        f.writelines(contents1)
    centroids2=centroids_dict2.data.detach().cpu().numpy()
    centroids2=centroids2.reshape(-1)
    centroids2_path='output/scentroids2.txt'  
    contents2=(["{}:".format(centroids2)+'\n'])
    with open(centroids2_path, 'a+') as f:
        f.writelines(contents2)

def get_lr_at_iter(alpha):
    return 1. / 3.* (1 - alpha) + alpha

def semantic_loss_calc(x, y, mean=True):
    loss = (x - y) ** 2
    if mean:
        return torch.mean(loss)
    else:
        return loss

def euclid_dist(x, y):
    x_sq = (x ** 2).mean(-1)
    x_sq_ = torch.stack([x_sq] * y.size(0), dim = 1)
    y_sq = (y ** 2).mean(-1)
    y_sq_ = torch.stack([y_sq] * x.size(0), dim = 0)
    xy = torch.mm(x, y.t()) / x.size(-1)
    dist = x_sq_ + y_sq_ - 2 * xy
   # dist=torch.mean(dist)
    return dist

"""def class_loss(num_classes,centroids1,centroids2):
    #类内损失
    loss_intra=0
    for c in range(num_classes):
        loss_intra += torch.mean(centroids1[c,:] - centroids2[c,:]) ** 2
    loss_intra=loss_intra
    loss_inters=0
    for c in range(num_classes-1):
        loss_inters+=torch.mean(centroids1[c,:] - centroids2[c+1,:])** 2
    loss_inter=1-loss_inters
    return loss_intra,loss_inter"""

def class_loss(num_classes,centroids1,centroids2):
    #类内损失
   
    x=centroids1#[c,:]
    y=centroids2#[c,:]
    x_sq = (x ** 2).mean(-1)     
    x_sq_ = torch.stack([x_sq] * y.size(0), dim = 1)
    y_sq = (y ** 2).mean(-1)
    y_sq_ = torch.stack([y_sq] * x.size(0), dim = 0)
    xy =torch.abs(torch.mm(x, y.t()) / x.size(-1))
    dist = x_sq_ + y_sq_ - 2 * xy
    dia = torch.diagonal(dist)
    loss_intra=torch.mean(dia)

    sum_all=torch.sum(torch.triu(dist))
    left_all=sum_all-torch.sum(dia)
    loss_inters=left_all/(num_classes*(num_classes-1)/2)#** 2465
    loss_inter=1-0.5*loss_inters
    return loss_intra,loss_inter