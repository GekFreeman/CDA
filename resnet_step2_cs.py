import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch.nn.functional as F
from torch.autograd import Variable
import torch
#from dsbn import DomainSpecificBatchNorm2d
import data_loader_sampler as data_loader
import numpy as np
import pdb
import os

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class MFSAN(nn.Module):

    def __init__(self, num_classes=31):
        super(MFSAN, self).__init__()
        self.sharedNet = resnet101(True)
        self.sonnet1 = ADDneck(2048, 256)
        self.sonnet2 = ADDneck(2048, 256)
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)  
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.sigmoid=nn.Sigmoid().cuda()

        self.gate=torch.ones(256).cuda()#.cuda(1)
       # self.gate=torch.nn.DataParallel(self.gate)
        
        
        #self.softmax=nn.functional.softmax()
        
        #self.mm = MM(256, 128)

    def forward(self, data_src1, label_src1 = 0,data_src2=0, label_src2 = 0, data_tgt1 = 0, data_tgt2 = 0, step = 1,alpha=1,label_tgt1 = 0,label_tgt2= 0,):
        mmd_loss = 0
        m=0.9*torch.ones(1).cuda()
        if self.training == True:
            if step==1:
            ######batchsize-1#####
                #print(self.gate.device)
               # gate=self.gate
                data_src1 = self.sharedNet(data_src1)
                data_src1 = self.sonnet1(data_src1)
                data_src1_sim=data_src1
                data_src1_pool = self.avgpool(data_src1)
                pred_src1=data_src1_pool
                pred_src1 = pred_src1.view(pred_src1.size(0), -1)
                mmd_src1=pred_src1
                pred_src1 = self.cls_fc_son1(pred_src1)  
            #########tgt###############
               # print(data_tgt1)
                data_tgt1 = self.sharedNet(data_tgt1)        
                data_tgt_son1 = self.sonnet1(data_tgt1)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                pred_tgt1 = self.cls_fc_son1(data_tgt_son1)
                mmd_tgt1=data_tgt_son1
                mmd_loss1= mmd.mmd(mmd_src1, mmd_tgt1)
            #######batchsize-2#########
                data_src2 = self.sharedNet(data_src2)
                data_src2 = self.sonnet1(data_src2)
                data_src2_sim=data_src2
                data_src2_pool = self.avgpool(data_src2)
                pred_src2=data_src2_pool
                pred_src2 = pred_src2.view(pred_src2.size(0), -1)
                mmd_src2=pred_src2
                pred_src2 = self.cls_fc_son1(pred_src2)
                #########tgt###############
                data_tgt2 = self.sharedNet(data_tgt2)        
                data_tgt_son2 = self.sonnet1(data_tgt2)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
                pred_tgt2 = self.cls_fc_son1(data_tgt_son2)
                mmd_tgt2=data_tgt_son2
                mmd_loss2= mmd.mmd(mmd_src2, mmd_tgt2)
                mmd_loss=mmd_loss1+ mmd_loss2         
            ######distance######
                cos_sim = Dotdist(data_src1_sim,data_src2_sim)##########256-d output similarity#######
            ############norma###33
                sim_loss = torch.mean(cos_sim, dim = 0)
#             data_src1_pool = data_src1_pool** 2
#             data_src2_pool = data_src2_pool** 2 
#             data_src1_pool = torch.sum(data_src1_pool*dist)
#             data_src2_pool = torch.sum(data_src2_pool*dist)
#             data_src1_pool = torch.sqrt(data_src1_pool)
#             data_src2_pool = torch.sqrt(data_src2_pool)0

               # cls_loss = F.nll_loss(F.log_softmax(pred_src1, dim=1), label_src1)+F.nll_loss(F.log_softmax(pred_src2, dim=1), label_src2)
               # cls_loss /= 2
                cls_loss = F.nll_loss(F.log_softmax(pred_src1, dim=1), label_src1) + alpha * F.nll_loss(F.log_softmax(pred_src2, dim=1), label_src2)
                cls_loss /= (1 + alpha)
                
                return cls_loss, sim_loss, mmd_loss,pred_tgt1, pred_tgt2
            if step==2:
                ###############batchsize1#########
                data_src1 = self.sharedNet(data_src1)
                data_src1 = self.sonnet1(data_src1)
                data_src1_sim=data_src1#源域前一半类别，256维特征图
               # data_src1_pool = self.avgpool(data_src1)
                #pred_src1=data_src1_pool
                #pred_src1 = pred_src1.view(pred_src1.size(0), -1)                 
              #  mmd_loss += mmd.mmd(data_src1, data_tgt_son1)
                #######batchsize-2#########
                data_src2 = self.sharedNet(data_src2)
                data_src2 = self.sonnet1(data_src2)
                data_src2_sim=data_src2#源域后一半类别，256维特征图
                #data_src2_pool = self.avgpool(data_src2)
                #pred_src2=data_src2_pool
                #pred_src2 = pred_src2.view(pred_src2.size(0), -1)
                
                ########target#################
                data_tgt1  = self.sharedNet(data_tgt1)
                data_tgt1 = self.sonnet1(data_tgt1)
                data_tgt_sim1=data_tgt1#目标域，256维特征图
                
                data_tgt2  = self.sharedNet(data_tgt2)
                data_tgt2 = self.sonnet1(data_tgt2)
                data_tgt_sim2=data_tgt2#目标域，256维特征图
                
                data_tgt_pool1 = self.avgpool(data_tgt1)
                pred_tgt1=data_tgt_pool1
                pred_tgt_view1 = pred_tgt1.view(pred_tgt1.size(0), -1)
                pred_target1 = self.cls_fc_son1(pred_tgt_view1)
                pred_tgt1=nn.functional.softmax(pred_target1, dim=1)
                target_label1 = pred_tgt1.data.max(1)[1]
                
                data_tgt_pool2 = self.avgpool(data_tgt2)
                pred_tgt2=data_tgt_pool2
                pred_tgt_view2 = pred_tgt2.view(pred_tgt2.size(0), -1)
                pred_target2 = self.cls_fc_son1(pred_tgt_view2)
                pred_tgt2=nn.functional.softmax(pred_target2, dim=1)
                target_label2 = pred_tgt2.data.max(1)[1]
                ##########源域B1-B2不同类别之间距离########越大越好
                #cos_sim_b12 = Dotdist(data_src1_sim,data_src2_sim)
                #sim_loss_b12 = torch.mean(cos_sim_b12, dim = 0)
                ##########取源域一个batch内同类别的b1-T&B2-T的特征图######
                fea_src1,lab_src1,fea_t1,lab_t1 =data_loader.sample_sameclass(data_src1_sim,label_src1,data_tgt_sim1,label_tgt1)
                fea_src2,lab_src2,fea_t2,lab_t2 = data_loader.sample_sameclass(data_src2_sim,label_src2,data_tgt_sim2,label_tgt2)
                
               # fea_src1,lab_src1,fea_t1,lab_t1 = data_src1_sim,label_src1,data_tgt_sim1,label_src1
               # fea_src2,lab_src2,fea_t2,lab_t2 = data_src2_sim,label_src2,data_tgt_sim2,label_src2 
                ###########做mmd###########
               # tgt_pool=self.avgpool(data_tgt)
                #import pdb
               # pdb.set_trace()
            ######源域和目标域同类别的distance######越小越好
                if fea_src1.shape[0]>0 and fea_src2.shape[0]>0:
                    #print(fea_src1.shape[0],fea_src1.shape[0])
               
                    cos_sim1 = Dotdist(fea_src1,fea_t1)
                    cos_sim2 = Dotdist(fea_src2,fea_t2)          
           #######sigmoid-gate#########
                    gate1=self.sigmoid(100*cos_sim1).cuda()
                    gate2=self.sigmoid(100*cos_sim2).cuda()
               # print(gate1.shape,gate2.shape)
                    devices=gate1.device
                    self.gate=self.gate.to(devices)
                    self.gate=m*self.gate+(1-m)*(gate1+gate2)/2
               # print(self.gate.shape)
                ###########同类别的B1-T,B2-T构建新的特征图进行分类#########
                    data_src1_pool = self.avgpool(data_src1)
                    pred_src1=data_src1_pool
                    pred_src1 = pred_src1.view(pred_src1.size(0), -1)
                    fea_src1_pool=pred_src1
                    data_src2_pool = self.avgpool(data_src2)
                    pred_src2=data_src2_pool
                    pred_src2 = pred_src2.view(pred_src2.size(0), -1)
                    fea_src2_pool=pred_src2
                ###############mmdloss##########3
               
                    data_fea1=self.avgpool(fea_src1)
                    data_fea1=data_fea1.view(data_fea1.size(0),-1)
                    data_fea2=self.avgpool(fea_src2)
                    data_fea2=data_fea2.view(data_fea2.size(0),-1)
                    data_tgt1=self.avgpool(fea_t1)
                    data_tgt1=data_tgt1.view(data_tgt1.size(0),-1)
                    data_tgt2=self.avgpool(fea_t2)
                    data_tgt2=data_tgt2.view(data_tgt2.size(0),-1)
                    mmd_loss += mmd.mmd(data_fea1, data_tgt1)
                    mmd_loss += mmd.mmd(data_fea2, data_tgt2)
                ############gate作用于特征图得不同类相似度#################
                    pred_src1_gate=pred_src1*gate1
                    pred_src2_gate=pred_src2*gate2
                ##########属于不同类别的B1和B2#######
                    simb12_loss_gate=difdist(pred_src1_gate,pred_src2_gate)
                
                ##########################
                    pred_src1 = self.cls_fc_son1(pred_src1_gate)      
                    pred_src2 = self.cls_fc_son1(pred_src2_gate) 

                    cls_loss = F.nll_loss(F.log_softmax(pred_src1, dim=1), label_src1) + alpha * F.nll_loss(F.log_softmax(pred_src2, dim=1), label_src2)
                    cls_loss /= (1 + alpha)
                    return cls_loss.mean(), mmd_loss.mean(),pred_target1,pred_target2#, mmd_loss, f_src,f_tgt   
                else:
                    data_src1_pool = self.avgpool(data_src1)
                    pred_src1=data_src1_pool
                    pred_src1 = pred_src1.view(pred_src1.size(0), -1)
                    
                    data_src2_pool = self.avgpool(data_src2)
                    pred_src2=data_src2_pool
                    pred_src2 = pred_src2.view(pred_src2.size(0), -1)
                    data_tgt_son1 = self.avgpool(data_tgt1[:32,:,:,:])
                    data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                    mmd_loss += mmd.mmd(pred_src1, data_tgt_son1)
                    
                    pred_src1 = self.cls_fc_son1(pred_src1)      
                    pred_src2 = self.cls_fc_son1(pred_src2) 
                    cls_loss = F.nll_loss(F.log_softmax(pred_src1, dim=1), label_src1) + alpha * F.nll_loss(F.log_softmax(pred_src2, dim=1), label_src2)
                    cls_loss /= (1 + alpha)
                    return cls_loss.mean(),mmd_loss,pred_target1,pred_target2
                    
                       
        
        else:            
            data = self.sharedNet(data_src1)
            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            devices=fea_son1.device
            self.gate=self.gate.to(devices)
           # print(data_src1.device)
           # print(self.gate.device)
            fea_son1=fea_son1*self.gate
            pred1 = self.cls_fc_son1(fea_son1)
          #  print("test")
            return pred1

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

##############similarity############
###dot distance###
def Dotdist(input1,input2):
    if input1.shape[0]==0:
        mean=torch.zeros(input2.shape[1])
        return mean
    if input2.shape[0]==0:
        mean=torch.zeros(input1.shape[1])
        return mean
    else:
        input1 = input1.view(input1.shape[0], input1.shape[1], -1)
        input2 = input2.view(input2.shape[0], input2.shape[1], -1)
        cos_sim = torch.cosine_similarity(input1, input2, dim=2) # BxC
        mean_sim = torch.mean(cos_sim, dim=0) # C
#     soft_sim = torch.softmax(mean_sim, dim=0) # C
        return mean_sim
##
def difdist(input1,input2):
    b1,b2=input1.shape[0],input2.shape[0]
   # print(b1,b2)
    cos_sim=0
    for i in range(b1):
        d1=input1[i,:]
        for j in range(b2): 
            d2=input2[j,:]
            d1 = d1.view(d1.shape[0],  -1)
            d2 = d2.view(d2.shape[0], -1)
            cos_sim+=torch.cosine_similarity(d1, d2, dim=1) # BxC
    cos_simb12=-F.log_softmax(cos_sim)
    mean_sim = 0.1*torch.mean(cos_simb12, dim=0) # C
#     soft_sim = torch.softmax(mean_sim, dim=0) # C
  #  print(mean_sim)
    nnsoftmax=torch.mean(torch.nn.functional.softmax(cos_simb12, dim=0),dim=0)
  #  print(nnsoftmax)
    return mean_sim
##cosine distance
def Cosine(input1,input2):
    input1=torch.nn.functional.normalize(input1, dim=1)
    input2=torch.nn.functional.normalize(input2, dim=1)
    d=torch.matmul(input1, torch.transpose(input2))
    
    return d