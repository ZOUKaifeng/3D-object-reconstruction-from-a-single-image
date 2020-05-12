import torch as T
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import json
import random
from read_obj import *

data_path = './train/image'
obj_path = './train/model/'

def init_pointcloud_loader(num_points):
    Z = np.random.rand(num_points) + 1.
    h = np.random.uniform(10., 214., size=(num_points,))
    w = np.random.uniform(10., 214., size=(num_points,))
    X = (w - 111.5) / 248. * -Z
    Y = (h - 111.5) / 248. * Z
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    Z = np.reshape(Z, (-1, 1))
    XYZ = np.concatenate((X, Y, Z), 1)
    return XYZ.astype('float32')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def collate(batch):
    data = [b for b in zip(*batch)]
    if len(data) == 3:
        init_pc, imgs, gt_pc = data
    elif len(data) == 4:
        init_pc, imgs, gt_pc, metadata = data
    else:
        raise ValueError('Unknown data values')

    init_pc = T.from_numpy(np.array(init_pc)).requires_grad_(False)
    imgs = T.from_numpy(np.array(imgs)).requires_grad_(False)
    gt_pc = [T.from_numpy(pc).requires_grad_(False) for pc in gt_pc]
    return (init_pc, imgs, gt_pc) if len(data) == 3 else (init_pc, imgs, gt_pc, metadata)

def OnUnitCube(pointcloud):
    c = np.max(pointcloud) - np.min(pointcloud)
    s = np.max(c)
    v = pointcloud / s
    return v - v.mean(axis = 0,keepdims = True)

def resampler(pointcloud, num):
    all_num = pointcloud.shape[0]
    index=random.sample(range(1,all_num),num)
    new_pc = pointcloud[index]
    return new_pc



class ShapeNet(Dataset):
    def __init__(self, train_set, grayscale=None, type='train', n_points=2048, **kwargs):
        self.n_points = n_points
        self.grayscale = grayscale
        self.train_set = train_set
        self.filenames = []
        self.obj_name = []
        self.labels = []
        for key in train_set:
            self.filenames.append(key['image'])
            self.obj_name.append(key['model']+'.obj')

        self.file_nums = len(self.filenames)

    def __len__(self):
        return self.file_nums

    def __getitem__(self, idx):
        img_ = os.path.join(data_path, self.filenames[idx])
        obj_ = os.path.join(obj_path, self.obj_name[idx])

        #img = np.asarray(Image.open(img_).resize((224, 224))).astype('float32')
        img = np.asarray(Image.open(img_).resize((227, 227))).astype('float32')
        img = rgb2gray(img)[..., None] if self.grayscale else img
        img = (np.transpose(img / 255.0, (2, 0, 1)) - .5) * 2

        obj = ObjLoader(obj_).vertices.astype('float32')

        
        pc = resampler(obj, self.n_points)
        pc = OnUnitCube(pc)
        #pc -= np.mean(pc, 0, keepdims=True)
        #return  init_pointcloud_loader(self.n_points), np.array(img, 'float32'), pc
        return np.array(img, 'float32'), pc

if __name__ == '__main__':
    with open("./train_data.json",'r') as load_f:
        load_dict = json.load(load_f)
    dataloader = ShapeNet(load_dict, grayscale = True)
    for i in range(len(dataloader)):
        init, img, pc = dataloader[i]
        print("init shape:", init.shape)
        print("img shape:", img.shape)
        print("pc shape:", pc.shape)
    
