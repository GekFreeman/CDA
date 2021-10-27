from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import resnet_step2 as models
from net_utils import save_net, load_net,save_checkpoint, save_checkpoints,euclid_dist,class_loss
from net_utils import semantic_loss_calc as semantic_loss_c
#from centroids import Centroids
import pdb
#from tsne import get_data_before,get_data_after, plot_embedding_2D,tsne_2D,tsne_3D
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 16
iteration = 10000#000#00#000
lr = 0.01
momentum = 0.9
cuda = True
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "/userhome/chengyl/UDA/multi-source/MFSAN/dataset/OfficeHomeDataset_10072016/"
output_dir="./output"

"""source1_name = "amazon"
source2_name = "webcam"
target_name = "dslr"
"""
source1_name = "Art"
source2_name = "Clipart"
source3_name = "Product"
target_name = "Real World"
"""
source1_name = "amazon"
source2_name = "dslr"
target_name = "webcam"
"""
domain_label={'source1':1,'source2':2,'target':0}
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
num_domains = len(domain_label)

#source1_loader1, source1_loader2= data_loader.load_training_source1(root_path, source1_name, batch_size, kwargs)
#source2_loader1, source2_loader2 = data_loader.load_training_source2(root_path, source2_name, batch_size, kwargs)
#target_train_loader = data_loader.load_training_target(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path,target_name, batch_size, kwargs)
num_domains = len(domain_label)
num_source1_domains = 1
num_source2_domains = 1
num_target_domains = 1
num_classes = 31
in_features =31



def test(model,source):
    model.eval()
    test_loss = 0
    s=source
    num=0
    correct = 0
    correct1 = 0
    correct2 = 0
    pred_num=[0 for i in range(65)]
    class_num=[0 for i in range(65)]
    with torch.no_grad():
        for data, target in target_test_loader:
            num+=1
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            for k in range(len(target)):
                class_num[target[k]]+=1
            pred1 = model(data,step=0)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)

            pred = pred1 

             #2021/03/04
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            for k in range(len(pred)):
                pred_num[pred[k]]+=1
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
        num=batch_size*num
        test_loss /= num#len(target_test_loader.dataset)
       # from __future__ import division
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, 1.0*correct, num,
            100. * correct / num))
        print('\nsource1 accnum {}'.format(correct1))
        print('\ntruthclass_num {}'.format(class_num))
        print('\npredclass_num {}'.format(pred_num))
        gap=[class_num[i]-pred_num[i] for i in range(len(class_num))]
        print('\ngap_num {}'.format(gap))
    return correct

if __name__ == '__main__':
    model = models.MFSAN(num_classes=65)
   # print('with simi')
    #print('models1-resnetdsbn')
    #print('cl_mfsandsbn')
    checkpoint =torch.load("/userhome/chengyl/UDA/multi-source/MFSAN/cl_office31/model/cl_step2_dslr_webcam_to_amazon.pth")
  #  print(checkpoint['model'])
    model.load_state_dict(checkpoint['model'])
    model.cuda()
   # print(model)

    test(model,3)
   # test(model,3)
    #print('begin 2d-tsne')
   # data_all,label_all=tsne(model)
   # fig=tsne_2D(data_all,label_all)
   # print('begin 3d-tsne')
    #fig=tsne_3D(data_all,label_all)