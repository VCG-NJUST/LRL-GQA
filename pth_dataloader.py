import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def pth_txt_generate(_input_feature_path,_input_label_path,_out_train_txt,_out_test_txt,ratio_for_train):
    pth_feature = os.listdir(_input_feature_path)
    pth_label = os.listdir(_input_label_path)
    size = len(pth_feature) #列表集数量
    pth_txt_train = open(_out_train_txt,mode='w',encoding='utf-8')
    pth_txt_test = open(_out_test_txt,mode='w',encoding='utf-8')
    for i in range(0, int(ratio_for_train * size)):
        if pth_feature[i].endswith('.pth') and pth_label[i].endswith('.pth'):
            path_feature = _input_feature_path + '/' +pth_feature[i]
            path_label = _input_label_path + '/' +pth_label[i]
            print(path_feature,' ',path_label)
            pth_txt_train.write('%s %s\n' % (path_feature, path_label))

    for i in range(int(ratio_for_train * size),size):
        if pth_feature[i].endswith('.pth') and pth_label[i].endswith('.pth'):
            path_feature = _input_feature_path + '/' +pth_feature[i]
            path_label = _input_label_path + '/' +pth_label[i]
            print(path_feature,' ',path_label)
            pth_txt_test.write('%s %s\n' % (path_feature, path_label))


def read_root_txt(root_txt_path):
    assert root_txt_path.endswith('.txt')
    features_list, labels_list = [],[]
    try:
        with open(root_txt_path,'r',encoding='utf-8') as f_txt:
            lines = f_txt.readlines() #读取文本中全部内容，以列表形式返回
            for line in lines:
                feature,label = line.strip().split(' ')
                features_list.append(feature)
                labels_list.append(label)
    except UnicodeDecodeError:
        with open(root_txt_path,'r',encoding='utf-8') as f_txt:
            lines = f_txt.readlines() #读取文本中全部内容，以列表形式返回
            for line in lines:
                feature,label = line.strip().split(' ')
                features_list.append(feature)
                labels_list.append(label)
    return features_list,labels_list


class ListDataloader(Dataset):
    def __init__(self,root_txt_path):
        super(ListDataloader,self).__init__()
        assert root_txt_path.endswith('.txt') and isinstance(root_txt_path,str)
        # isinstance()函数用来判断一个对象是否是一个已知的类型，类似type()
        self.features_list,self.labels_list = read_root_txt(root_txt_path)

    def __getitem__(self, index):
        # 如果在类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[key]取值。当实例对象做P[key]运算时，就会调用类中的__getitem__()方法
        '''

        :param index: 列表序号
        :return: [B x patch_num, patch_feature]
                 [-1, 1]:列表label信息
        '''
        feature_load = self.features_list[index]
        label_load = self.labels_list[index]
        features = torch.load(feature_load)
        # features = features.view(6*64,64)
        features = features.view(11 * 64, 64) #0925使用
        labels = torch.load(label_load)
        return features,labels

    def __len__(self):
        return(len(self.features_list))

if __name__ == '__main__':

    #以下五行为1028使用
    input_feature_path = '.\pth_feature'
    input_label_path = '.\pth_label'
    out_train_txt = '.\pth_train_txt.txt'
    out_test_txt = '.\pth_test_txt.txt'
    pth_txt_generate(input_feature_path, input_label_path, out_train_txt, out_test_txt, 0.9)