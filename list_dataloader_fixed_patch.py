import torch
from torch.utils.data import Dataset, DataLoader
from FPS import load_ply, draw_points, farthest_point_sample, index_points, KNN_sample, query_ball_point, \
    relative_cordinate
import numpy as np
from tqdm import tqdm
import re

# @data 2022/12/12

def read_root_txt(root_txt_path):
    assert root_txt_path.endswith('.txt')  # 判断文件某个条件是否成立，false引发异常
    path_list, labels_list, distortion_type = [], [], []
    try:
        with open(root_txt_path, 'r', encoding='utf-8') as f_txt:
            lines = f_txt.readlines()  # 读取文本中全部内容，以列表形式返回
            for line in lines:
                path, label, distortion = line.strip().split(' ')  # 此语句和下一条根据数据集文本形式选择
                print(line)
                #path, label= line.strip().split(' ')
                label = float(label)
                path_list.append(path)
                labels_list.append(label)
                # distortion_type.append(distortion)
    except UnicodeDecodeError:
        with open(root_txt_path, 'r', encoding='utf-8') as f_txt:
            lines = f_txt.readline()  # 读取文本中全部内容，以列表形式返回
            for line in lines:
                path, label, distortion = line.strip().split(' ')
                path, label, distortion = line.strip().split(' ')
                label = int(label)
                path_list.append(path)
                labels_list.append(label)
                distortion_type.append(distortion)
    # return path_list,labels_list,distortion_type
    return path_list, labels_list


# txt_full_path: 样本数据txt完整路径，读取某个路径文件
def read_sample_txt_files(txt_full_path):
    assert txt_full_path.endswith('.txt')
    with open(txt_full_path, "r", encoding='utf-8') as f_txt:
        lines = f_txt.readlines()  # 读取全部内容 ，并以列表方式返回
        info = lines[0].strip()
        info = info[1: -1]
        split_infos = info.split(',')
        # print(split_infos)
        split_infos = [float(elem) for elem in split_infos]
        # split_infos =split_infos[0:45]
        return split_infos


def pc_normalize(pc):
    '''
    input param pc: [B,N,C]
    output :零均值化 [B,N,C]
    '''

    centroid = np.mean(pc, axis=1)  # B X C
    B = pc.shape[0]
    data = np.zeros(pc.shape, dtype=pc.type.type)
    # for batch in range(B):
    #     data[batch]=pc[batch]-centroid[batch] #[B,N,C]
    # m=np.max(np.sqrt(np.sum(pc**2,axis=-1)),axis=-1)  #(B,)
    # for batch in range(B):
    #     pc[batch] = data[batch] / m[batch]

    m = np.max(np.sqrt(np.sum(pc ** 2, axis=-1)), axis=-1)  # (B,)
    for batch in range(B):
        data[batch] = pc[batch] - centroid[batch]  # [B,N,C]
        pc[batch] = data[batch] / m[batch]
    return pc


def index_to_points(points, idx):
    '''
    在N个点中按照序号S挑选S个值 ,与FPS中index_points功能基本一致
      Input:
          points: input points data, [B, N, C]
          idx: sample index data, [B, S]
          如果是B N C与 S  可以直接写成 points[:,index,:]  idex=[0 1 2...]
      Return:
          new_points:, indexed points data, [B, S, C]
    '''
    B = points.shape[0]
    S = idx.shape[1]
    C = points.shape[2]
    new_points = np.zeros((B, S, C), dtype=points.dtype.type)
    for batch in range(B):
        points_batch = points[batch]
        new_points[batch] = points_batch[idx[batch]]
    return new_points


def random_select(model, patch_npy, random_size):
    '''
     模型随机挑选patch并转换为坐标值
    :param model: 点云坐标 BxNxC
    :param patch_npy: 各个patch中相对model的坐标序号 BxS x patch_size
    :param random_size: 需要挑选的patch数目
    :return: 返回挑选后的patch点  B x random_size x patch_size x C
    '''
    S = model.shape[1]
    B = model.shape[0]
    patch_size = patch_npy.shape[2]
    C = model.shape[2]
    index = np.arange(0, S)
    select_index = np.random.choice(index, random_size, replace=False)  # 从总的patch数S中随机挑选random_size个patch进行训练输入
    select_index = select_index.reshape(-1, random_size)
    select_index = np.repeat(select_index, B, axis=0)  # [B random_size] 此操作每个Batch挑选的序号都相同,增加了矩阵的行数
    select_patch_index = index_to_points(patch_npy, select_index)  # [B S patch_size] -> [B random_size patch_size]
    result = np.zeros((B, random_size, patch_size, C), dtype=np.float)

    for patch in range(random_size):
        patch_index = select_patch_index[:, patch, :]  # [B patch_size]
        value = index_to_points(model, patch_index)  # [B N C]->[B patch_size C]
        for batch in range(B):
            result[batch][patch] = value[batch]
    return result  # [B random_size patch_size C]


# root_txt_path:放置路径和标签信息的txt文件完整路劲
class PCDataloader(Dataset):
    def __init__(self, root_txt_path, catergory=False, select_patch_numbers=64):
        super(PCDataloader, self).__init__()
        assert root_txt_path.endswith('.txt') and isinstance(root_txt_path, str)
        # isinstance()函数用来判断一个对象是否是一个已知的类型，类似type()

        # self.sample_path_list,self.labels_list,self.distortion_type=read_root_txt(root_txt_path)
        self.sample_path_list, self.labels_list = read_root_txt(root_txt_path)
        # print(self.sample_path_list,'\n',self.labels_list,'\n',self.distortion_type)
        self.catergory = catergory  # 用于判断是否返回噪声或者采样类型
        self.select_path_numbers = select_patch_numbers
        self.or_anchor_dict = {}

    def __getitem__(self, index):
        # 如果在类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[key]取值。当实例对象做P[key]运算时，就会调用类中的__getitem__()方法
        '''

        :param index: 样本序号
        :return:[]
        '''
        path = self.sample_path_list[index]  # 样本路径
        label = self.labels_list[index]
        or_model = re.findall('raw_model_\d+', path)[0]
        # distortion_type=self.distortion_type
        model = load_ply(path).reshape(1, -1, 3)  # 点云ply模型，[B N C] B=1,C=3
        model = torch.from_numpy(model)
        # label_to_tensor = torch.tensor(label, dtype=torch.float32)
        # distortion_type_to_tensor=torch.tensor(distortion_type,dtype=torch.float32)

        # 对模型进行采样，取patch
        if or_model not in self.or_anchor_dict:
            centroids_index = farthest_point_sample(model, self.select_path_numbers)  # 每次采样点数
            centroids = index_points(model, centroids_index)  # centroids:[B S C]
            self.or_anchor_dict[or_model] = centroids
        else:
            centroids = self.or_anchor_dict[or_model]
        # radius采样
        result = query_ball_point(0.2, 512, model, centroids)  # result:[B S nsample]
        result_np = result.numpy()
        B, S, patch_size = result_np.shape
        result_value = np.zeros((B, S, patch_size, 3), dtype=np.float32)
        model_numpy = model.numpy()  # 此部分代码基于numpy运算，故转换

        for patch in range(S):
            patch_index = result_np[:, patch, :]  # [B nsample]，nsample=patch_size
            value = index_to_points(model_numpy, patch_index)  # value:[B patch_size C]
            for batch in range(B):
                result_value[batch][patch] = value[batch]  # result_value:[B S patch_size C],S*patch_size=N
        data_tensor = torch.tensor(result_value, dtype=torch.float)

        # 相对坐标转换，将绝对坐标转化为相对于中心点的坐标
        data_tensor = relative_cordinate(data_tensor, centroids)  # [B S patch_size C]
        data_tensor = data_tensor[0]  # [S patch_size C]
        label_tensor = torch.tensor(label, dtype=torch.float)
        # print(label_tensor.shape)

        # if self.catergory:
        #     return data_tensor,label_tensor,self.distortion_type[index]
        # else:
        #     return data_tensor,label_tensor
        path_save = path[:-4] + '.pth'
        torch.save(data_tensor, path_save)
        return data_tensor, label_tensor

    def __len__(self):
        return len(self.sample_path_list)


class PTHDataloader(Dataset):
    def __init__(self, root_txt_path, catergory=False, select_patch_numbers=64):
        super(PTHDataloader, self).__init__()
        assert root_txt_path.endswith('.txt') and isinstance(root_txt_path, str)
        # isinstance()函数用来判断一个对象是否是一个已知的类型，类似type()

        # self.sample_path_list,self.labels_list,self.distortion_type=read_root_txt(root_txt_path)
        self.sample_path_list, self.labels_list = read_root_txt(root_txt_path)
        # print(self.sample_path_list,'\n',self.labels_list,'\n',self.distortion_type)
        self.catergory = catergory  # 用于判断是否返回噪声或者采样类型
        self.select_path_numbers = select_patch_numbers

    def __getitem__(self, index):
        # 如果在类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[key]取值。当实例对象做P[key]运算时，就会调用类中的__getitem__()方法
        '''

        :param index: 样本序号
        :return:[]
        '''
        path = self.sample_path_list[index]  # 样本路径
        label = self.labels_list[index]
        data_tensor = torch.load(path)
        label_tensor = torch.tensor(label, dtype=torch.float)
        # print(label_tensor.shape)

        # if self.catergory:
        #     return data_tensor,label_tensor,self.distortion_type[index]
        # else:
        #     return data_tensor,label_tensor
        return data_tensor, label_tensor

    def __len__(self):
        return len(self.sample_path_list)


if __name__ == '__main__':
    PCDataset = PCDataloader('./index/listwise_train_10level.txt', True)
    trainloader = DataLoader(PCDataset, batch_size=1, num_workers=0, shuffle=True, drop_last=False)
    # sample_tensor_list=[B patch_num patch_size C]

    sample_tensor_list = []
    label_tensor_list = []
    type_tensor_list = []
    for i, (sample_tensor, label_tensor) in enumerate(tqdm(trainloader)):
        sample_tensor_list.append(sample_tensor)
        label_tensor_list.append(label_tensor)
    # print("sample_tensor_list.reshape=",sample_tensor_list.shape)
    # print("label_tensor_list.reshape=", label_tensor_list.shape)

    # for i,(sample_tensor,label_tensor,type_tensor) in enumerate(trainloader):
    #      print(label_tensor)
    #      print(type_tensor[0]=='com&down')
