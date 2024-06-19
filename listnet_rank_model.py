import os
import sys
import copy
import math
from torch.autograd import Variable
# import chainer
#import chainer.functions as CF
# import chainer.links as CL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# @data: 2022/03/18


#列表网误差函数
def ndcg(self, y_true, y_score, k=6):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    y_true_sorted = sorted(y_true, reverse=True) #sorted() 函数对所有可迭代的对象进行排序操作
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
    dcg = 0
    argsort_indices = np.argsort(y_score)[::-1]
    for i in range(k):
        dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
    ndcg = dcg / ideal_dcg
    return ndcg

class listNet(nn.Module): 
    '''
        training data:[M L C],M=size_of_PCmodel;L=num of every list;C=feature
        input: X:[B L C],B=batch_size;L=size of every list;C=feature
               Y:[B L 1],B=batch_size;L=size of every list;1表示标签维度
    '''
    #先输出patch得分，再输出模型得分
    def __init__(self,input_len=64,out_len=1):
        super(listNet,self).__init__()
        self.K=11
        # self.feature,_,_=model_patch_own()
        self.patch_score = nn.Sequential(
            nn.Linear(in_features=64, out_features=64,bias=True),
            nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32, bias=True),
            nn.BatchNorm1d(32),
            # torch.nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(32, out_len, bias=True),
            nn.ReLU(),
        )
        self.weights = nn.Sequential(
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32, bias=True),
            nn.BatchNorm1d(32),
            # torch.nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(32, out_len, bias=True),
        )
        self.model_score = nn.Sequential(
            nn.Linear(64, 64, bias=True),
            nn.BatchNorm1d(64),
            # torch.nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_len, bias=True),
            # nn.ReLU(),
        )
        self.sm = nn.Sigmoid()

        self.patch_score_noweighted = nn.Sequential(
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32, bias=True),
            nn.BatchNorm1d(32),
            # torch.nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(32, out_len, bias=True),
            nn.ReLU(),
        )



    # 输出patch得分，再输出模型得分的训练函数
    def train_patchScore_to_modelScore(self,data_path,label_path,n_epochs,learning_rate):
        save_output_filename = open(
              r'./listnet_loss_test.txt', mode='w', encoding='utf-8')
        time_open = time.time()
        print(data_path)
        print(os.getcwd())
        print(os.path.exists(data_path))
        data = os.listdir(data_path)
        label = os.listdir(label_path)
        size = len(data)
        #设置两个参数组
        params_dict = [{'params':self.patch_score.parameters(),'lr':learning_rate}]
        optimizer = torch.optim.Adam(params_dict)
        loss_l = torch.nn.CrossEntropyLoss()
        epoch_loss_min = 30

        for epoch in range(n_epochs):
            running_loss = 0
            i = 0
            for feature, feature_label in zip(data, label):
                i += 1
                feature = data_path + '\\' + feature
                feature_label = label_path + '\\' + feature_label
                feature_load = torch.load(feature)
                label_load = torch.load(feature_label)
                feature_f = feature_load.float().cuda()
                label_f = label_load.float().cuda()
                feature_f = feature_f.view(-1,64)
                y_head = self.patch_score(feature_f).cuda()
                weights = self.weights(feature_f)
                # y_head = y_head.view(11, -1)
                weights_score = torch.max(y_head, weights).view(11, -1)
                y_head = (weights_score.sum(dim=1) / weights.view(11, -1).sum(dim=1)).reshape(11, -1)

                # breakpoint()
                # y_head = self.model_score(y_head).cuda()
                # y_head = self.sm(y_head)
                # y_head = (y_head - y_head.mean())/y_head.std()
                # y_head = self.sm(y_head)
                y_head = self.order_y_head(y_head, label_f).cuda()

                # print('y_head.shape=',y_head.shape)
                # logp_y_head = F.log_softmax(y_head, dim=-1)
                # p_label_f = F.softmax(label_f, dim=-1)
                # loss_kl_sum = F.kl_div(logp_y_head, p_label_f, reduction='sum')
                # print('y_head=',y_head)
                optimizer.zero_grad()
                loss = self.forward(y_head)
                # print('label=', label_f)
                # print('y_head=',y_head)
                loss = loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                running_loss += loss.data

                if i % 10 == 0:
                    print('-' * 20, '\n')
                    print('batch {},train loss:{:.4f}'.format(i, running_loss.item() / i))
                    print('-' * 20, '\n')

                    #self.predict_original(feature_f)
                    # print('y_head=', y_head)
                    # print('label_f',label_f)
                    # id = np.array([id for id in range(0, y_head.shape[0])], dtype=int).reshape(-1, 1)
                    # # id = np.array([id for id in range(feature_f.shape[0]-1, -1, -1)], dtype=int).reshape(-1,1)
                    # perms = self.permutation_original(id)
                    # perms = self.permutation(label_f)
                    # print(perms)
                    # print(label_f)

            epoch_loss = running_loss / size
            save_output_filename.write('%.4f\n' % (epoch_loss))
            save_output_filename.flush()
            print("Loss:{:.4f}".format(epoch_loss.item()))

            if epoch_loss < epoch_loss_min:
                epoch_loss_min = epoch_loss
                torch.save(self.state_dict(),
                           r'./listnet1212_min_loss.pth')
            if epoch % 10 == 0:
                torch.save(self.state_dict(),
                           r'./listnet1212_' + str(epoch) + '.pth')
        time_end = time.time() - time_open
        print(time_end)

    def predict_patchtoscore(self, x):
        self.probability = self.patch_score(x.float())
        self.probability = self.probability.view(6, -1)
        self.probability = self.model_score(self.probability)
        self.probability = (self.probability - self.probability.mean()) / self.probability.std()
        self.probability = self.sm(self.probability)
        return self.probability

    def predict_original(self, x):
        self.probability = self.patch_score(x.float())
        # self.probability = self.probability.view(6, -1)
        self.probability = self.probability.view(11, -1) #0926使用
        # self.probability = self.probability.view(21, -1) #1201使用
        self.probability = self.model_score(self.probability)
        self.probability = (self.probability - self.probability.mean()) / self.probability.std()
        self.probability = self.sm(self.probability)
        return self.probability

    '''
        return a permutation with highest probability
    '''
    def permutation(self, label):
        probability_cpu = self.probability.detach().cpu()
        label_cpu = label.cpu().numpy()
        perm = np.hstack((label_cpu, probability_cpu.numpy()))
        # perm = np.hstack((id, self.probability.detach().numpy()))
        # np.hstack:数组在水平方向上拼接,
        # tensor.detach():使得该tensor不计算梯度
        return sorted(perm, key=lambda x: x[1], reverse=True)
        # return perm
    # sorted(d.items(), key=lambda x: x[1]) 中 d.items() 为待排序的对象；
    # key=lambda x: x[1] 为对前面的对象中的第二维数据（即value）的值进行排序。 key=lambda 变量：变量[维数] 。维数可以按照自己的需要进行设置。

    def permutation_original(self, id):
        perm = np.hstack((id, self.probability.detach().cpu().numpy()))
        return sorted(perm, key=lambda x: x[1], reverse=True)

    def forward_original(self, y_head, y):
        sigma = 0
        for i in range(np.math.factorial(self.K)): #np.math.factorial求阶乘
            sigma += self.P_original(y_head) * torch.log(self.P_original(y))
        return -sigma
        return -torch.log(self.P(y_head))


    """
        calculate probabily of each/all permutation(s)
    """
    def P_original(self, fw):
        p = 1
        for t in range(self.K):
            upper = torch.exp(fw[t])
            downer = 0
            for l in range(fw.shape[0]):
                downer += torch.exp(fw[l])
            p *= upper / downer
        return p.float()


    """
        Cross Entropy Error Function
    """
    def forward(self,y_head):
        return -torch.log(self.P(y_head))
    '''
        calculate probabily of each/all permutation(s),计算每个/所有置换的概率
    '''
    def P(self, fw):
        p = 1
        for t in range(self.K):
            upper = torch.exp(fw[t])
            downer = 0
            for l in range(t, fw.shape[0]):
                downer += torch.exp(fw[l])
            p *= upper / downer
        return p.float()

    '''将y_head根据label进行排序'''
    def order_y_head(self, y_head, label):


        y_head = torch.cat((y_head, label), dim=1).cuda()
        y_head = torch.stack(sorted(y_head, key=lambda a: a[1]))
        y_head = torch.flip(y_head, dims=[0])
        y_head = torch.index_select(y_head, dim=1, index=torch.tensor([0]).cuda())
        return y_head


if __name__=='__main__':

    print('\n', torch.cuda.is_available())
    Use_gpu = torch.cuda.is_available()

    listnet = listNet().cuda()

    feature_path = r'./out_feature' #两行0926使用
    label_path = r'./out_label'
    listnet.train_patchScore_to_modelScore(feature_path,label_path,800,0.0001)


