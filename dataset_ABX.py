import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation

def normalize(data):
    #normScale = float(1/(data.max()-data.min()))
    #x = data * normScale
    return data/255.

def prepare_train_data(data_path_A, data_path_B, patch_size=60, stride=10, aug_times=1, if_reseize=True):
    # train
    print('process training data')
    scales = [1]#, 0.9, 0.8, 0.7]
    files_A = glob.glob(os.path.join('datasets', data_path_A, '*.*'))
    files_A.sort()

    files_B = glob.glob(os.path.join('datasets', data_path_B, '*.*'))
    files_B.sort()

    # assume all images in a single set have the same size
    lenA = len(files_A)
    lenScale = len(scales)
    h_a, w_a, _ = cv2.imread(files_A[0]).shape
    h_b, w_b, _ = cv2.imread(files_B[0]).shape
    dataLength = aug_times * lenA
    print(dataLength)
    data = np.empty(shape=(2,dataLength, h_a, w_a))


    train_num = 0
    for i in range(lenA):
        img_A = cv2.imread(files_A[i])
        img_B = cv2.imread(files_B[i])
        
        if if_reseize == True:
            img_B = cv2.resize(img_B, (h_a, w_a), interpolation=cv2.INTER_CUBIC)

        for k in range(lenScale):
            Img_A = cv2.resize(img_A, (int(h_a*scales[k]), int(w_a*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img_A = np.expand_dims(Img_A[:,:,0].copy(), 0)
            Img_A = np.float32(normalize(Img_A))

            Img_B = cv2.resize(img_B, (int(h_b*scales[k]), int(w_b*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img_B = np.expand_dims(Img_B[:,:,0].copy(), 0)
            Img_B = np.float32(normalize(Img_B))

            data_AB= [Img_A, Img_B]
            data[:, train_num,:,:] = data_AB

            train_num += 1
            for m in range(aug_times-1):
                rand = np.random.randint(1,8)
                data_aug_A = data_augmentation(Img_A, rand)
                data_aug_B = data_augmentation(Img_B, rand)
                data_aug_AB= [data_aug_A, data_aug_B]

                data[:, train_num,:,:] = data_aug_AB
                train_num += 1
    print('training set, # samples %d\n' % train_num)
    return data

def prepare_val_data(data_path_A, data_path_B, if_reseize=True):
    # val
    print('\nprocess validation data')
    files_A = glob.glob(os.path.join('datasets', data_path_A, '*.*'))
    files_A.sort()

    files_B = glob.glob(os.path.join('datasets', data_path_B, '*.*'))
    files_B.sort()

    h_a, w_a, _ = cv2.imread(files_A[0]).shape
    h_b, w_b, _ = cv2.imread(files_B[0]).shape
    data = np.empty(shape=[2, 0, h_a, w_a])
    val_num = 0
    for i in range(len(files_A)):
        print("file: %s" % files_A[i])
        img_A = cv2.imread(files_A[i])
        img_B = cv2.imread(files_B[i])
        if if_reseize == True:
            img_B = cv2.resize(img_B, (h_a, w_a), interpolation=cv2.INTER_CUBIC)
            
        img_A = np.expand_dims(img_A[:,:,0], 0)
        img_A = np.float32(normalize(img_A))
        img_B = np.expand_dims(img_B[:,:,0], 0)
        img_B = np.float32(normalize(img_B))

        data = np.append(data, [img_A, img_B], axis=1)
        val_num += 1

    print('val set, # samples %d\n' % val_num)
    return data


class Dataset(udata.Dataset):
    def __init__(self, train, data_path_A, data_path_B, data_path_val_A, data_path_val_B,  patch_size=50, stride=10, aug_times=2, if_reseize=False):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            self.dataList = prepare_train_data(data_path_A, data_path_B, patch_size, stride, aug_times, if_reseize)
            #print(self.dataList.shape)
        else:
            self.dataList = prepare_val_data(data_path_val_A, data_path_val_B, if_reseize)
            #print(self.dataList.shape)
        self.num = self.dataList.shape[1]
        self.A = data_path_A
        self.B = data_path_B
        self.valA = data_path_val_A
        self.valB = data_path_val_B
        self.patch_size = patch_size        
        self.stride = stride
        self.aug_times = aug_times
        self.if_reseize = if_reseize
        
        
    def __len__(self):
        return self.num
    def __getitem__(self, index):
        #print(index)
        one_data = self.dataList[:,index,:,:]
        #print(self.dataList.shape)
        return torch.Tensor(one_data)