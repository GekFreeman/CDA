from torchvision import datasets, transforms
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import pdb
import random
from collections import defaultdict
from PIL import Image
from collections import defaultdict
import heapq
from heapq import * 


def save_target(dic,target_pred,target_path):
    ##h:key:{confidence,data}
  #  h=defaultdict(list)
    target_pred= torch.nn.functional.softmax(target_pred, dim=1)
    target_label=target_pred.data.max(1)[1]
    pred_confidence=target_pred.data.max(1)[0]
    top_n=50
    for i in range(target_pred.shape[0]):
        #if pred_confidence[i]>1:
         #   pdb.set_trace()
       # print(pred_confidence[i])
        
        if pred_confidence[i]>0.7:    
           # print(target_label[i],target_path[i])
            key=target_label[i].detach().cpu().numpy()
           
            key=str(key)  
            hea=dic[key]
            
            heappush(hea,( pred_confidence[i],target_path[i]))
            hea=heapq.nlargest(top_n,hea,key=lambda x:x[0])
            dic[key]=hea       
    #
    
    return dic
            
    
    
############

def load_target(dic,source_data1, source_label1, source_data2,source_label2):
    num_s1,num_s2=source_label1.shape[0],source_label2.shape[0]
    c,h,w=source_data1.shape[1],source_data1.shape[2],source_data1.shape[3]
    t_data=torch.zeros((num_s1+num_s2,c,h,w),dtype=torch.float).cuda()
    t_label=torch.zeros(num_s1+num_s2,dtype=torch.long).cuda() 
    t1_list= [random.randint(0,10) for i in range(num_s1)]
    t2_list= [random.randint(0,10) for i in range(num_s2)]
    
    
    for i in range(num_s1):
        key=str(source_label1[i].detach().cpu().numpy())
        index1=t1_list[i]
        t_data[i]=dic[key][index1]
        t_label[i]=source_label1[i]
    for j in range(num_s2):
        key=str(source_label2[j].detach().cpu().numpy())
        index2=t2_list[j]
        t_data[num_s1+j]=dic[key][index2]
        t_label[num_s1+j]=source_label2[j]
    return t_data#,t_label
def sample_sameclass(source_data,source_label,tgt_data,tgt_label):
    print(torch.unique(source_label),torch.unique(tgt_label))
    index_s=[]
    index_t=[]
    b=source_data.shape[0]
    c,h,w=source_data.shape[1],source_data.shape[2],source_data.shape[3]
    tgt_index=[i for i in range(tgt_data.shape[0])]
    dic=defaultdict(list)
    for dat,lab in zip(tgt_index,tgt_label):
        #if lab==dic.keys():
        key=lab.detach().cpu().numpy()
        key=str(key)
        dic[key].append(dat)
    #t=0
    for s in range(len(source_label)):
        s_idx=str(source_label[s].detach().cpu().numpy())        
        if s_idx in dic.keys():#在源域中找到与目标域一样的类，记录编号
                t=random.choice(dic[s_idx])
                index_s.append(s)
                index_t.append(t)
               # visited.add(t)
               # break
#     # dt = {0: [0, 1, 2], 1: [3, 4, 5]}
#     index_t = []
#     for i in s:
#         index_t.append(np.random.choice(dt[source_label[i]], size=1))
    num=len(index_s)
    s_data=torch.zeros((num,c,h,w),dtype=torch.float).cuda()
    s_label=torch.zeros(num,dtype=torch.long).cuda()
    t_data=torch.zeros((num,c,h,w),dtype=torch.float).cuda()
    t_label=torch.zeros(num,dtype=torch.long).cuda()             
   # print(s_data.dtype)
    for i in range(num):#按照编号去处数据替换原来值
        index=index_s[i]
        s_data[i,:]=source_data[index,:]
        s_label[i]=source_label[index]
    for j in range(num):
        index=index_t[j]
        t_data[j,:]=tgt_data[index,:]
        t_label[j]=tgt_label[index]
   
    return s_data, s_label ,t_data,t_label
def save_results(dic, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(dic)
    
    with open(filename, mode) as file:
        # header
       # file.write('Classes:\t')
        idx=0
        save_dic=defaultdict(list)
        for key,value in dic.items():
           # file.write('Classes:\t')
           # file.write('{:s}\t'.format(key))
           # print(key)
          #  print(value)
            idx+=len(value)
            for j in range(len(value)):
               # print(value[j])
                
                if save_dic[key] is not None:
                    save_dic[key].append(value[j][1])
                else:
                    save_dic[key]=value[j][1]
                  
                    
        for keys,values in save_dic.items():
           
           file.write('class:{:s}:{:s}\t'.format(keys,str(values)))
        print('have save classes:{:s},have save samples:{:s}'.format(str(len(dic)),str(idx)))
        

        