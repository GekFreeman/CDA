from torchvision import datasets, transforms
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import pdb
import random
from collections import defaultdict
import os
from PIL import Image
"""def load_training_source1(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform);
    cls1_num = len(data.classes)
    
    data_cls1=list(np.random.choice(cls1_num,size=cls1_num // 2,replace=False))
    data_cls1set=list(data.classes[i] for i in data_cls1)
    data_cls2=[]
    for i in range(cls1_num):
        if i not in data_cls1:
            data_cls2.append(i)
   # data_cls1=set(list(range(cls1_num // 2))) # 以类别中位数分割,1为前N/2个类别,2为后N/2个类别
   # data_cls2=set(list(range(cls1_num // 2, cls1_num))) 
    # 按类别重新提取索引
    data_cls2set=list(data.classes[i] for i in data_cls2)
    index1 = []
    index2 = []
    classes=[]
    for i in range(len(data)):
        if data.imgs[i][1] in data_cls1:
            index1.append(i) 
        else:
            index2.append(i)
            
    
    # shuffle, 但这里的shuffle仅执行一次, 而不是每个epoch执行一次, 可优化
   
    random.shuffle(index1)
    random.shuffle(index2)
    #sampler1 = torch.utils.data.sampler.SubsetRandomSampler(index1)                                                                                                                                                                    
    #sampler2 = torch.utils.data.sampler.SubsetRandomSampler(index2)
    
    train_loader1 = torch.utils.data.DataLoader(myImageFloder(root=root_path + dir, label=data_cls1set,index=data_cls1, transform=transform),batch_size=batch_size, shuffle=True, **kwargs)
    train_loader2 = torch.utils.data.DataLoader(myImageFloder(root=root_path + dir, label=data_cls2set,index=data_cls2, transform=transform),batch_size=batch_size, shuffle=True, **kwargs)

    #torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler1, **kwargs)
   # train_loader2 = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler2, **kwargs)
    
    
    
    
  #  print(index)
    
    return train_loader1,train_loader2#,index"""
def load_training_source1(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform);
    
    cls1_num = len(data.classes) # 类别总数
    
    data_cls1=list(np.random.choice(cls1_num,size=cls1_num // 2,replace=False))
    
    data_cls2=[]
    for i in range(cls1_num):
        if i not in data_cls1:
            data_cls2.append(i)
   # data_cls1=set(list(range(cls1_num // 2))) # 以类别中位数分割,1为前N/2个类别,2为后N/2个类别
   # data_cls2=set(list(range(cls1_num // 2, cls1_num))) 
    # 按类别重新提取索引
    index1 = []
    index2 = []
    for i in range(len(data)):
        if data.imgs[i][1] in data_cls1:
            index1.append(i)
        else:
            index2.append(i)
#     pdb.set_trace()
    # shuffle, 但这里的shuffle仅执行一次, 而不是每个epoch执行一次, 可优化
    random.shuffle(index1)
    random.shuffle(index2)
    sampler1 = torch.utils.data.sampler.SubsetRandomSampler(index1)                                                                                                                                                                    
    sampler2 = torch.utils.data.sampler.SubsetRandomSampler(index2)
    train_loader1 = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler1, **kwargs)
    train_loader2 = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler2, **kwargs)
    
    return train_loader1,train_loader2
def load_training_source2(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform);
    
    cls1_num = len(data.classes) # 类别总数
    
    data_cls1=list(np.random.choice(cls1_num,size=cls1_num // 2,replace=False))
    
    data_cls2=[]
    for i in range(cls1_num):
        if i not in data_cls1:
            data_cls2.append(i)
   # data_cls1=set(list(range(cls1_num // 2))) # 以类别中位数分割,1为前N/2个类别,2为后N/2个类别
   # data_cls2=set(list(range(cls1_num // 2, cls1_num))) 
    # 按类别重新提取索引
    index1 = []
    index2 = []
    for i in range(len(data)):
        if data.imgs[i][1] in data_cls1:
            index1.append(i)
        else:
            index2.append(i)
#     pdb.set_trace()
    # shuffle, 但这里的shuffle仅执行一次, 而不是每个epoch执行一次, 可优化
    random.shuffle(index1)
    random.shuffle(index2)
    sampler1 = torch.utils.data.sampler.SubsetRandomSampler(index1)                                                                                                                                                                    
    sampler2 = torch.utils.data.sampler.SubsetRandomSampler(index2)
    train_loader1 = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler1, **kwargs)
    train_loader2 = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler2, **kwargs)
    
    return train_loader1,train_loader2
def load_training_source3(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform);
    
    cls1_num = len(data.classes) # 类别总数
    
    data_cls1=list(np.random.choice(cls1_num,size=cls1_num // 2,replace=False))
    
    data_cls2=[]
    for i in range(cls1_num):
        if i not in data_cls1:
            data_cls2.append(i)
   # data_cls1=set(list(range(cls1_num // 2))) # 以类别中位数分割,1为前N/2个类别,2为后N/2个类别
   # data_cls2=set(list(range(cls1_num // 2, cls1_num))) 
    # 按类别重新提取索引
    index1 = []
    index2 = []
    for i in range(len(data)):
        if data.imgs[i][1] in data_cls1:
            index1.append(i)
        else:
            index2.append(i)
#     pdb.set_trace()
    # shuffle, 但这里的shuffle仅执行一次, 而不是每个epoch执行一次, 可优化
    random.shuffle(index1)
    random.shuffle(index2)
    sampler1 = torch.utils.data.sampler.SubsetRandomSampler(index1)                                                                                                                                                                    
    sampler2 = torch.utils.data.sampler.SubsetRandomSampler(index2)
    train_loader1 = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler1, **kwargs)
    train_loader2 = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler2, **kwargs)
    
    return train_loader1,train_loader2
###########44444444444444444444444444444444444
def load_training_source4(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform);
    
    cls1_num = len(data.classes) # 类别总数
    
    data_cls1=list(np.random.choice(cls1_num,size=cls1_num // 2,replace=False))
    
    data_cls2=[]
    for i in range(cls1_num):
        if i not in data_cls1:
            data_cls2.append(i)
   # data_cls1=set(list(range(cls1_num // 2))) # 以类别中位数分割,1为前N/2个类别,2为后N/2个类别
   # data_cls2=set(list(range(cls1_num // 2, cls1_num))) 
    # 按类别重新提取索引
    index1 = []
    index2 = []
    for i in range(len(data)):
        if data.imgs[i][1] in data_cls1:
            index1.append(i)
        else:
            index2.append(i)
#     pdb.set_trace()
    # shuffle, 但这里的shuffle仅执行一次, 而不是每个epoch执行一次, 可优化
    random.shuffle(index1)
    random.shuffle(index2)
    sampler1 = torch.utils.data.sampler.SubsetRandomSampler(index1)                                                                                                                                                                    
    sampler2 = torch.utils.data.sampler.SubsetRandomSampler(index2)
    train_loader1 = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler1, **kwargs)
    train_loader2 = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler2, **kwargs)
    
    return train_loader1,train_loader2
##############################################
###########5555555555555555555555555555555555
def load_training_source5(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform);
    
    cls1_num = len(data.classes) # 类别总数
    
    data_cls1=list(np.random.choice(cls1_num,size=cls1_num // 2,replace=False))
    
    data_cls2=[]
    for i in range(cls1_num):
        if i not in data_cls1:
            data_cls2.append(i)
   # data_cls1=set(list(range(cls1_num // 2))) # 以类别中位数分割,1为前N/2个类别,2为后N/2个类别
   # data_cls2=set(list(range(cls1_num // 2, cls1_num))) 
    # 按类别重新提取索引
    index1 = []
    index2 = []
    for i in range(len(data)):
        if data.imgs[i][1] in data_cls1:
            index1.append(i)
        else:
            index2.append(i)
#     pdb.set_trace()
    # shuffle, 但这里的shuffle仅执行一次, 而不是每个epoch执行一次, 可优化
    random.shuffle(index1)
    random.shuffle(index2)
    sampler1 = torch.utils.data.sampler.SubsetRandomSampler(index1)                                                                                                                                                                    
    sampler2 = torch.utils.data.sampler.SubsetRandomSampler(index2)
    train_loader1 = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler1, **kwargs)
    train_loader2 = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler2, **kwargs)
    
    return train_loader1,train_loader2
#############################################
def load_training_target(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    ###########################
    """
    index1 = []
    index2 = []
    for i in range(len(data)):
        if data.imgs[i][1] in data_cls1:
            index1.append(i)
        else:
            index2.append(i)
#     pdb.set_trace()
    # shuffle, 但这里的shuffle仅执行一次, 而不是每个epoch执行一次, 可优化
    random.shuffle(index1)
    random.shuffle(index2)
    sampler1 = torch.utils.data.sampler.SubsetRandomSampler(index1)                                                                                                                                                                    
    sampler2 = torch.utils.data.sampler.SubsetRandomSampler(index2)"""
    sample_num = len(data)
    index=list(i for i in range(sample_num))
    random.shuffle(index)
   
    batch_sampler1 = torch.utils.data.sampler.BatchSampler(index,batch_size=batch_size,drop_last=True)  
    batch_sampler2 = torch.utils.data.sampler.BatchSampler(index,batch_size=batch_size,drop_last=True)  
 
    
    train_loader1 = torch.utils.data.DataLoader(data, batch_sampler=batch_sampler1, **kwargs)
    train_loader2 = torch.utils.data.DataLoader(data, batch_sampler=batch_sampler2, **kwargs)
   
   # import pdb
  # pdb.set_trace()
    return train_loader1,train_loader2
def load_sameclass_target(root_path, dir,dic,source_data1, source_label1, source_data2,source_label2,batch_size):
    num_s1,num_s2=source_label1.shape[0],source_label2.shape[0]
    c,h,w=source_data1.shape[1],source_data1.shape[2],source_data1.shape[3]
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    
    cls_num1 = num_s1
    cls_num2 = num_s2
    data_cls1=list(i for i in source_label1)
    data_cls2=list(i for i in source_label2)
    sampler_index1=myImageTargetFloder(dic=dic, index=data_cls1, transform=transform)
    sampler_index2=myImageTargetFloder(dic=dic, index=data_cls2, transform=transform)
    sampler1 = torch.utils.data.sampler.SequentialSampler(sampler_index1)                                                                                                                       
    sampler2 = torch.utils.data.sampler.SequentialSampler(sampler_index2)


    train_loader1 = torch.utils.data.DataLoader(data, batch_size=len(sampler_index1), shuffle=False, drop_last=True, sampler=sampler1)
    train_loader2 = torch.utils.data.DataLoader(data, batch_size=len(sampler_index2), shuffle=False, drop_last=True, sampler=sampler2)
    train_loader1=iter(train_loader1).next()[0]
    train_loader2=iter(train_loader2).next()[0]
    return train_loader1,sampler_index1,train_loader2, sampler_index2 
def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,shuffle=True, drop_last=True, **kwargs)#drop_last=True,
    
    ## pdb.set_trace()
    return test_loader
def sample_sameclass(source_data,source_label,tgt_data,tgt_label):
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

###################
def random_choose_data(label_path):
    random.seed(1)
    file = open(label_path)
    lines = file.readlines()
    slice_initial = random.sample(lines, 200000)  # if don't change this ,it will be all the same
    slice = list(set(lines)-set(slice_initial))
    random.shuffle(slice)

    train_label = slice[:150000]
    test_label = slice[150000:200000]
    return train_label, test_label  # output the list and delvery it into ImageFolder


 # def my data loader, return the data and corresponding label
def default_loader(path):
    return Image.open(path).convert('RGB') 

class myImageFloder(torch.utils.data.Dataset):  # Class inheritance，继承Ｄａｔａｓｅｔ类
    def __init__(self, root, label,index, transform=None, target_transform=None, loader=default_loader):
            
         # fh = open(label)
        c = 0
        imgs = []
        class_names = []
        for i in range(len(index)):  # label is a list
           # print(line)
            # cls is a list
            fn = label[i]
            path1=os.path.join(root, fn)
            for ims in os.listdir(path1):
               # print(ims)
                path=os.path.join(path1, ims)
                imgs.append((index[i],path))
                # access the last label
                # images is the list,and the content is the tuple, every image corresponds to a label
                # despite the label's dimension
                 # we can use the append way to append the element for list
            c = c + 1
       # print('the total image is',c)
        #print(class_names)
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    
    def __getitem__(self, index):
        label,path = self.imgs[index]  # even though the imgs is just a list, it can return the elements of it
        # in a proper way
       # path2=os.path.join(self.root, label)
       
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label,path
    #　在这里返回图像数据以及对应的ｌａｂｅｌ以及对应的名称
    def __len__(self):##
        return len(self.imgs)

    def getName(self):
        return self.classes
def myImageTargetFloder(dic, index, transform=None, loader=default_loader):
    
    idx_tgt = []
  
    for i in range(len(index)):  # label is a list

        key=str(index[i].item())
            #print(index[i])
           # print(key)
          #  print(dic[key])
        if len(dic[key])!=0:
                
            idx=np.random.choice(len(dic[key]))
               # idx=np.random.choice(len(dic[key]),size=1,replace=False)
               # print(idx)
            idx_now=dic[key][idx][1]
            idx_tgt.append(idx_now)
            
    return idx_tgt
"""def myImageTargetFloder(dic, index, transform=None, loader=default_loader):
    
    path = []
    label=[]
    imgs = []
    for i in range(len(index)):  # label is a list

        key=str(index[i].item())
            #print(index[i])
           # print(key)
          #  print(dic[key])
        if len(dic[key])!=0:
                
            idx=np.random.choice(len(dic[key]))
               # idx=np.random.choice(len(dic[key]),size=1,replace=False)
               # print(idx)
            path_now=dic[key][idx][1]
            path.append(path_now)
            label.append(index[i].item())
            img = loader(path_now)
            if transform is not None:
                img = transform(img)
                img=np.array(img)
                imgs.append(img)
    
    image=torch.Tensor(imgs)
    return image,label,path"""
