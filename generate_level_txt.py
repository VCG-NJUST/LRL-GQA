import os
import torch

'''
@data:2022-09-13
'''

def data_txt_generate(_input_dir_path,_out_train_txt,_out_test_txt,ratio_for_train):
    data=os.listdir(_input_dir_path)
    size=len(data)  #模型数量   

    # for k in range(150):
    #     _out_train_txt=_out_train_txt+str(k)
    #     _out_test_txt=_out_test_txt+str(k)
    listwise_txt_train=open(_out_train_txt,mode='w',encoding='utf-8')
    listwise_txt_test=open(_out_test_txt,mode='w',encoding='utf-8') #以写入形式打开文档
    distortion_level = 9
    for i in range(0, int(ratio_for_train * size)):
        model_dir = os.listdir(_input_dir_path + '/' + data[i])  # 某一模型文件夹
        original_model=''
        for j in model_dir:
            path_j=_input_dir_path+'/'+data[i]+'/'+j
            print(path_j)
            if os.path.isfile(path_j) and j.endswith('.ply'):
                original_model=path_j

                print(path_j) #获取原始模型
                #listwise_txt_train.write('%s %d %s\n' % (original_model,distortion_level,'groudtruth'))
        for j in model_dir:
            if os.path.isdir(_input_dir_path+'/'+data[i]+'/'+j): #模型的某一种失真类型文件夹
                distortion_type=str(j)
                file_train=[]
                for k in os.listdir(_input_dir_path+'/'+data[i]+'/'+j):
                    path=_input_dir_path+'/'+data[i]+'/'+j+'/'+k
                    if k.endswith('.ply'):
                        # file_train.append(k)
                        if k[:9] == 'raw_model' :
                            listwise_txt_train.write('%s %d %s\n' % (path, distortion_level + 1, 'groudtruth'))
                        # elif distortion_type=='random' or distortion_type=='gridAverage' or distortion_type=='OctreeCom':
                        #     distortion_label=distortion_level-int(k[-5])
                        #     listwise_txt_train.write('%s %d %s\n' % (path,distortion_label,'com&down'))
                        # else:
                        #     distortion_label = distortion_level - int(k[-5])
                        #     listwise_txt_train.write('%s %d %s\n' % (path,distortion_label,'noise'))
                for k in os.listdir(_input_dir_path+'/'+data[i]+'/'+j):
                    path=_input_dir_path+'/'+data[i]+'/'+j+'/'+k
                    if k.endswith('.ply'):
                        file_train.append(k)
                        if k[:9] == 'raw_model' :
                            # listwise_txt_train.write('%s %d %s\n' % (path, distortion_level, 'groudtruth')
                            continue
                        if distortion_type=='random_downsample' or distortion_type=='gridAverage_downsample' or distortion_type=='OctreeCom':
                            distortion_label=distortion_level-int(k[-5])
                            listwise_txt_train.write('%s %d %s\n' % (path,distortion_label,'com&down'))
                        else:
                            distortion_label = distortion_level - int(k[-5])
                            listwise_txt_train.write('%s %d %s\n' % (path,distortion_label,'noise'))

    for i in range(int(ratio_for_train*size),size):
        model_dir=os.listdir(_input_dir_path+'/'+data[i]) #某一类型文件夹
        original_model=''
        for j in model_dir:
            path_j=_input_dir_path+'/'+data[i]+'/'+j
            print(path_j)
            if os.path.isfile(path_j) and j.endswith('.ply'):
                original_model=path_j
                # listwise_txt_test.write('%s %d %s\n' % (original_model,distortion_level,'groudtruth'))
        for j in model_dir:
            if os.path.isdir(_input_dir_path+'/'+data[i]+'/'+j):
                distortion_type=str(j)
                file_test=[]
                for k in os.listdir(_input_dir_path+'/'+data[i]+'/'+j):
                    path=str(_input_dir_path+'/'+data[i]+'/'+j+'/'+k)
                    if k.endswith('.ply'):
                        # file_test.append(k)
                        if k[:9] == 'raw_model' :
                            listwise_txt_test.write('%s %d %s\n' % (path, distortion_level + 1, 'groudtruth'));
                        # elif distortion_type=='random' or distortion_type=='gridAverage' or distortion_type=='OctreeCom':
                        #     distortion_label = distortion_level - int(k[-5])
                        #     listwise_txt_test.write('%s %d %s\n' % (path,distortion_label,'com&down'))
                        # else:
                        #     distortion_label = distortion_level - int(k[-5])
                        #     listwise_txt_test.write('%s %d %s\n' % (path,distortion_label,'noise'))
                for k in os.listdir(_input_dir_path+'/'+data[i]+'/'+j):
                    path=str(_input_dir_path+'/'+data[i]+'/'+j+'/'+k)
                    if k.endswith('.ply'):
                        file_test.append(k)
                        if k[:9] == 'raw_model' :
                            # listwise_txt_test.write('%s %d %s\n' % (path, distortion_level, 'groudtruth'));
                            continue
                        elif distortion_type=='random_downsample' or distortion_type=='gridAverage_downsample' or distortion_type=='OctreeCom':
                            distortion_label = distortion_level - int(k[-5])
                            listwise_txt_test.write('%s %d %s\n' % (path,distortion_label,'com&down'))
                        else:
                            distortion_label = distortion_level - int(k[-5])
                            listwise_txt_test.write('%s %d %s\n' % (path,distortion_label,'noise'))

if __name__=='__main__':
    input_dir_path = r'..\Data10'
    out_train_txt = '.\index\listwise_train_10level.txt'
    out_test_txt = '.\index\listwise_test_10level.txt'
    data_txt_generate(input_dir_path, out_train_txt, out_test_txt, 0.9)