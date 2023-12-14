import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import sys
import copy
import math
from list_feature_patch import model_own,PCT_model
from list_dataloader_fixed_patch import PTHDataloader
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
from pct import Point_Transformer


# @data:2022/06/05


print('\n',torch.cuda.is_available())
Use_gpu = torch.cuda.is_available()
model = PCT_model()
if Use_gpu:
   model=model.cuda()

txt_dir_path = '.\index\listwise_train_10level.txt'
PTHataset = PTHDataloader(txt_dir_path, True)
trainloader = DataLoader(PTHataset, batch_size=1, num_workers=0, shuffle=True, drop_last=False)

save_output_filename = open('.\LossAndAcc.txt', mode='w',encoding='utf-8')
#model = model_own()
loss_f = torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())

n_epochs = 201
learning_rate = 0.001
time_open = time.time()
pbar = tqdm(total=n_epochs *trainloader.__len__() )
for epoch in tqdm(range(n_epochs)):
    pbar.set_description('Epoch %3d' % epoch)
    running_loss = 0.0
    running_correct = 0
    print('epoch {}/{}'.format(epoch, n_epochs))
    print('--'*10)
    for i,(sample_tensor,label_tensor) in enumerate(trainloader):
        pbar.update()
        sample_tensor = sample_tensor.permute(0, 3, 1, 2)
        x_train = Variable(sample_tensor)
        s = x_train.shape[2]
        #print('sample_tensor.shape', x_train.shape)
        y_train = label_tensor.repeat(s)
        y_train = Variable(y_train).cuda()
        #print('label_tensor',label_tensor)

        y_pred = model(x_train)
        #print('y_pred',y_pred.shape)
        _,pred = torch.max(y_pred.data,1)

        optimizer.zero_grad()
        loss = loss_f(y_pred,y_train.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_correct += (torch.sum(pred == y_train.data)/s)

        if i%10 == 0:
            print('-' * 20, '\n')
            print('batch {},train loss:{:.4f},Train ACC:{:.4f}'.format(i,running_loss/i,100*running_correct/(i)))
            print('-' * 20, '\n')

    epoch_loss = running_loss/len(trainloader)
    epoch_acc = 100*running_correct/len(trainloader)
    save_output_filename.write('%.4f %.4f\n' % (epoch_loss, epoch_acc))
    save_output_filename.flush()
    print("Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss,epoch_acc))
    if epoch % 10 == 0:
        torch.save(model,'.\PCT_' + str(280+epoch) +'.pth')
time_end = time.time() - time_open
print(time_end)

# if __name__ == '__main__':
#     # pre_feature_net_params = torch.load('model_own_test_patch2.pth')
#     # for k, v in pre_feature_net_params.state_dict().items():
#     #     print(k, '\t', v)
