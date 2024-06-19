import os
import sys
import copy
import math
from torch.autograd import Variable
# import chainer
# import chainer.functions as CF
# import chainer.links as CL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from list_dataloader_fixed_patch import PTHDataloader
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from list_feature_patch import model_own_tune
# from utils_rank import plot_result
# from utils_rank import NNfuncs
import time
from pointnet_utils import PointNetEncoder,STN3d,STNkd
from model import DGCNN
#from fe import FeatureExtraction_tune

def generation_fea_lab(txt_dir_path, tune_feature_model, load_model, list_length, feature_length, feature_saveroad,
                       label_saveload, way):
    # 输入参数:1.txt_dir_path：生成特征列表和标签列表的点云数据集文本文件
    #         2.tune_feature_model：需要进行迁移学习的类模板
    #         3.load_model：训练好待导入的模型
    #         4.list_length：生成列表的长度
    #         5.feature_length：点云或者patch特征的维度
    #         6.feature_saveroad：生成特征列表保存路径
    #         7.label_saveload：生成标签列表保存路径
    #         8.way:表示点云以整个模型作为输入还是分patch
    print('\n', torch.cuda.is_available())
    Use_gpu = torch.cuda.is_available() 

    PTHDataset = PTHDataloader(txt_dir_path, True)
    trainloader = DataLoader(PTHDataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)
    feature_model = tune_feature_model
    print(feature_model)
    if Use_gpu:
        feature_model = feature_model.cuda().eval()
    # 读取修改之前自己模型保存的参数设置
    pre_feature_net_params = torch.load(load_model)
    print(pre_feature_net_params)
    for k, v in pre_feature_net_params.state_dict().items():
        print(k, '\t', v)
    # 读取网络参数
    model_dict = feature_model.state_dict()  # 读取修改之后的网络参数
    # 将之前保存的模型参数<pre_feature_net_params>里不属于net_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pre_feature_net_params.state_dict().items() if k in model_dict}
    # 寻找网络中公共层，并保留预训练参数
    print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)  # 将预训练参数更新到新的网络层
    feature_model.load_state_dict(model_dict)
    print(feature_model)
    for parma in feature_model.parameters():
        parma.requires_grad = False

    # out_feature_list = torch.zeros(int(list_length), int(feature_length)) #以model用
    out_feature_list = torch.zeros(int(list_length), 64, int(feature_length))  # 64表示patch个数，以model时候需去掉
    out_label_list = torch.zeros(int(list_length), 1)
    list_num = 0
    pbar = tqdm(total= trainloader.__len__())
    for i, (sample_tensor, label_tensor) in enumerate(trainloader):
        pbar.update()
        if list_num % int(list_length) == 0:
            order = np.arange(0, int(list_length))
            np.random.shuffle(order)
            order_num = 0
            #print(order)
        #print(i)
        list_num += 1
        if way == 'model_way':
            sample_tensor = sample_tensor.permute(0, 2, 1).cuda()  # 整个模型
        if way == 'patch_way':
            sample_tensor = sample_tensor.permute(0, 3, 1, 2).cuda()  # 分patch
        sample_tensor.requires_grad = False
        out_feature= feature_model(sample_tensor)
        out_label = label_tensor
        # print('out_feature=', out_feature.shape)
        # print('out_label', out_label)

        # 以model用
        # out_feature_list[int(list_num%11 - 1),:] = out_feature
        # out_label_list[int(list_num%11 - 1),:] = out_label

        # 以patch用
        out_feature_list[int(order[order_num]), :, :] = out_feature
        out_label_list[int(order[order_num]), :] = out_label

        order_num += 1
        if list_num % int(list_length) == 0:
            feature_path = feature_saveroad + str(int(list_num / list_length)) + '.pth'
            torch.save(out_feature_list.detach().cpu(), feature_path)
            label_path = label_saveload + str(int(list_num / list_length)) + '.pth'
            torch.save(out_label_list.detach().cpu(), label_path)
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()


if __name__ == '__main__':
    print('\n', torch.cuda.is_available())
    Use_gpu = torch.cuda.is_available()

    txt_dir_path = r'.\index\listwise_test_10level.txt'
    tune_feature_model = model_own_tune()
    load_model = r'.\model_own_test_patch_470.pth'
    list_length = 11
    feature_length = 64
    feature_saveroad = r'.\out_feature_test\'
    label_saveload = r'.\out_label_test\'
    if not os.path.exists(feature_saveroad):
        os.mkdir(feature_saveroad)
    if not os.path.exists(label_saveload):
        os.mkdir(label_saveload)
    way = 'patch_way'

    generation_fea_lab(txt_dir_path, tune_feature_model, load_model, list_length, feature_length, feature_saveroad,
                       label_saveload, way)
