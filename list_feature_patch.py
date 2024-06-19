import os
import sys
import copy
import math

# import chainer
#import chainer.functions as CF
# import chainer.links as CL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from utils_rank import plot_result
# from utils_rank import NNfuncs
# from pct import Point_Transformer


# @data:2022/06/05


def knn(x, k):
    '''

    :param x: [B C N]
    :param k: int
    :return: [B N K]
    '''
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True) 
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    # print("inner=",inner.shape)
    # print("xx=",xx.shape)
    # print("pairwise_distance=",pairwise_distance.shape)
    # print("idx",idx.shape)
    return idx


def get_graph_feature(x, k=20, idx=None):
    '''

    :param x: [B C N]
    :param k: int
    :param idx:
    :return: [B C*2 N K]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size,num_points,k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    # idx_base = torch.arange(0, batch_size, device='cpu').view(-1, 1, 1) * num_points

    # print("num_points", num_points)
    # print("x=", x.shape)
    # print("idx_base=", idx_base.shape)
    # print("idx_base=", idx_base)

    idx = idx + idx_base
    # print("idx",idx.shape)
    idx = idx.view(-1)
    # print("idx",idx.shape)
    _, num_dims, _ = x.size()
    # print("num_dims=",num_dims)

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    #print("feature=", feature.shape)
    #print("feature=",feature)

    return feature

class model_own(nn.Module):

    # input:
    #       point:[B,C,S,patch_size]
    # return:
    #       feature:[patch_num,k]
    def __init__(self, k=20, dropout=0.5, output_channels=256):
        super(model_own, self).__init__()

        # self.args = args
        self.k = k

        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(32)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=1, bias=False),
            self.bn1,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d((3 + 16) * 2, 32, kernel_size=1, bias=False),
            self.bn2,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

        self.bn_down_1 = torch.nn.BatchNorm2d(32)
        self.bn_down_2 = torch.nn.BatchNorm2d(32)
        self.conv_down_1 = nn.Sequential(
            torch.nn.Conv2d(16 * 2, 32, kernel_size=1, bias=False),
            self.bn_down_1,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_down_2 = nn.Sequential(
            torch.nn.Conv2d(32 * 2, 32, kernel_size=1, bias=False),
            self.bn_down_2,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear_down_1 = nn.Linear(64, 64, bias=True)
        self.linear_down_2 = nn.Linear(64, 32, bias=True)
        # self.linear_down_3 = nn.Linear(32, 6, bias=True) #原始实验设置，模型一共分为6个等级
        self.linear_down_3 = nn.Linear(32, 11, bias=True)

    def forward(self,x):
        # x = [B C S patch_size]
        device = torch.device('cuda:0')
        x = x.to(device)
        # print('x.shape=', x.shape)
        x = x.squeeze(0).permute(1,0,2)
        # print('x.shape=',x.shape)
        batch_size = x.size(0)
        c = x.size(1)
        n = x.size(2)

        x1_gf = get_graph_feature(x, k=self.k)
        # print("x1_gf.shape=",x1_gf.shape)
        x1_conv = self.conv1(x1_gf)
        # print("x1_conv.shape=", x1_conv.shape)
        x1_max = x1_conv.max(dim=-1, keepdim=False)[0]
        # print("x1_max.shape=", x1_max.shape)

        x2_gf = get_graph_feature(x1_max, k=self.k)
        # print("x2_gf.shape=", x2_gf.shape)
        x2_gf = torch.cat((x1_gf, x2_gf), dim=1)
        # print("x2_gf.shape=", x2_gf.shape)
        x2_conv = self.conv2(x2_gf)
        # print("x2_conv.shape=", x2_conv.shape)
        x2_max = x2_conv.max(dim=-1, keepdim=False)[0]
        # print("x2_max.shape=", x2_max.shape)

        #降维操作
        x_down_1 = get_graph_feature(x1_max, k=self.k)
        # print('x_down_1.shape',x_down_1.shape)
        x_down_1 = self.conv_down_1(x_down_1)
        x_down_1 = x_down_1.max(dim=-1, keepdim=False)[0]
        # print('x_down_1.shape',x_down_1.shape)

        x_down_2 = get_graph_feature(x2_max, k=self.k)
        x_down_2 = self.conv_down_2(x_down_2)
        x_down_2 = x_down_2.max(dim=-1, keepdim=False)[0]
        # print('x_down_2.shape', x_down_2.shape)

        x = torch.cat((x_down_1, x_down_2), dim=1)
        # print("x.shape", x.shape)
        x = x.view(batch_size, -1, 64)
        # print("x.shape", x.shape)
        x = x.mean(axis = 1, keepdim=False)
        print("x.shape", x.shape)
        x = F.leaky_relu(self.linear_down_1(x), negative_slope=0.2)
        x = F.leaky_relu(self.linear_down_2(x), negative_slope=0.2)
        x_classify = F.leaky_relu(self.linear_down_3(x), negative_slope=0.2)

        return  x_classify

class model_own_tune(nn.Module):

    # input:
    #       point:[B,C,S,patch_size]
    # return:
    #       feature:[patch_num, feature]
    def __init__(self, k=20, dropout=0.5, output_channels=256):
        super(model_own_tune, self).__init__()

        # self.args = args
        self.k = k

        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(32)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=1, bias=False),
            self.bn1,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d((3 + 16) * 2, 32, kernel_size=1, bias=False),
            self.bn2,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

        self.bn_down_1 = torch.nn.BatchNorm2d(32)
        self.bn_down_2 = torch.nn.BatchNorm2d(32)
        self.conv_down_1 = nn.Sequential(
            torch.nn.Conv2d(16 * 2, 32, kernel_size=1, bias=False),
            self.bn_down_1,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_down_2 = nn.Sequential(
            torch.nn.Conv2d(32 * 2, 32, kernel_size=1, bias=False),
            self.bn_down_2,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear_down_1 = nn.Linear(64, 64, bias=True)
        self.linear_down_2 = nn.Linear(64, 32, bias=True)
        # self.linear_down_3 = nn.Linear(32, 6, bias=True) #原始实验设置，模型一共分为6个等级
        self.linear_down_3 = nn.Linear(32, 11, bias=True)

    def forward(self,x):
        # x = [B C S patch_size]
        device = torch.device('cuda:0')
        x = x.to(device)
        x = x.squeeze(0).permute(1,0,2)
        #print('x.shape=',x.shape)
        batch_size = x.size(0)
        c = x.size(1)
        n = x.size(2)

        x1_gf = get_graph_feature(x, k=self.k)
        #print("x1_gf.shape=",x1_gf.shape)
        x1_conv = self.conv1(x1_gf)
        # print("x1_conv.shape=", x1_conv.shape)
        x1_max = x1_conv.max(dim=-1, keepdim=False)[0]
        #print("x1_max.shape=", x1_max.shape)

        x2_gf = get_graph_feature(x1_max, k=self.k)
        #print("x2_gf.shape=", x2_gf.shape)
        x2_gf = torch.cat((x1_gf, x2_gf), dim=1)
        # print("x2_gf.shape=", x2_gf.shape)
        x2_conv = self.conv2(x2_gf)
        # print("x2_conv.shape=", x2_conv.shape)
        x2_max = x2_conv.max(dim=-1, keepdim=False)[0]
        #print("x2_max.shape=", x2_max.shape)

        #降维操作
        x_down_1 = get_graph_feature(x1_max, k=self.k)
        # print('x_down_1.shape',x_down_1.shape)
        x_down_1 = self.conv_down_1(x_down_1)
        x_down_1 = x_down_1.max(dim=-1, keepdim=False)[0]
        #print('x_down_1.shape',x_down_1.shape)

        x_down_2 = get_graph_feature(x2_max, k=self.k)
        x_down_2 = self.conv_down_2(x_down_2)
        x_down_2 = x_down_2.max(dim=-1, keepdim=False)[0]
        # print('x_down_2.shape', x_down_2.shape)

        x = torch.cat((x_down_1, x_down_2), dim=1)
        #print("x.shape", x.shape)
        x = x.view(batch_size, -1, 64)

        x = x.mean(axis = 1, keepdim=False)

        return  x
class PCT_model(nn.Module):

    # input:
    #       point:[B,C,S,patch_size]
    # return:
    #       feature:[patch_num,k]
    def __init__(self, k=20, dropout=0.5, output_channels=256):
        super(PCT_model, self).__init__()

        # self.args = args
        self.k = k

        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(32)

        self.linear_down_1 = nn.Linear(64, 64, bias=True)
        self.linear_down_2 = nn.Linear(64, 32, bias=True)
        # self.linear_down_3 = nn.Linear(32, 6, bias=True) #原始实验设置，模型一共分为6个等级
        self.linear_down_3 = nn.Linear(32, 11, bias=True)
        self.pct = Point_Transformer()

    def forward(self,x):
        # x = [B C S patch_size]
        device = torch.device('cuda:0')
        x = x.to(device)
        # print('x.shape=', x.shape)
        x = x.squeeze(0).permute(1,0,2)
       # print('x.shape=',x.shape)
        x = self.pct(x)
        #print("x.shape", x.shape)
        x = F.leaky_relu(self.linear_down_1(x), negative_slope=0.2)
        x = F.leaky_relu(self.linear_down_2(x), negative_slope=0.2)
        x_classify = F.leaky_relu(self.linear_down_3(x), negative_slope=0.2)

        return  x_classify

    class PCT_model_own(nn.Module):

        # input:
        #       point:[B,C,S,patch_size]
        # return:
        #       feature:[patch_num,k]
        def __init__(self, k=20, dropout=0.5, output_channels=256):
            super(PCT_model, self).__init__()

            # self.args = args
            self.k = k

            self.bn1 = torch.nn.BatchNorm2d(16)
            self.bn2 = torch.nn.BatchNorm2d(32)

            self.linear_down_1 = nn.Linear(64, 64, bias=True)
            self.linear_down_2 = nn.Linear(64, 32, bias=True)
            # self.linear_down_3 = nn.Linear(32, 6, bias=True) #原始实验设置，模型一共分为6个等级
            self.linear_down_3 = nn.Linear(32, 11, bias=True)
            self.pct = Point_Transformer()

        def forward(self, x):
            # x = [B C S patch_size]
            device = torch.device('cuda:0')
            x = x.to(device)
            # print('x.shape=', x.shape)
            x = x.squeeze(0).permute(1, 0, 2)
            ##print('x.shape=', x.shape)
            x = self.pct(x)
            # print("x.shape", x.shape)
            x = F.leaky_relu(self.linear_down_1(x), negative_slope=0.2)
            x = F.leaky_relu(self.linear_down_2(x), negative_slope=0.2)
            x_classify = F.leaky_relu(self.linear_down_3(x), negative_slope=0.2)

            return x_classify


if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    a = torch.randn(1, 3, 64, 512).to(device)
    model = PCT_model().cuda()
    x=model(a)
    print(x)
