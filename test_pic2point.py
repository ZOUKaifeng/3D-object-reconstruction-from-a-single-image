from __future__ import print_function
import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.cuda as cuda
from pic2points import pic2points
from torch.nn.parallel import DataParallel
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch
import model.chamfer.dist_chamfer as ext
from tqdm import tqdm
from PIL import Image

class ShapeNet(Dataset):
    def __init__(self, test_set, grayscale=None, n_points=2048, **kwargs):
        self.n_points = n_points
        self.grayscale = grayscale
        self.test_set = test_set
        self.filenames = os.listdir(self.test_set)

        self.file_nums = len(self.filenames)

    def __len__(self):
        return self.file_nums

    def __getitem__(self, idx):
        img_ = os.path.join(self.test_set, self.filenames[idx])


        img = np.asarray(Image.open(img_).convert('RGB').resize((227, 227))).astype('float32')
        img = rgb2gray(img)[..., None] if self.grayscale else img
        img = (np.transpose(img / 255.0, (2, 0, 1)) - .5) * 2


        #pc -= np.mean(pc, 0, keepdims=True)
        return  np.array(img, 'float32'), self.filenames[idx]

def main():
    test_path = '/home/zou/img2obj/data/validation/image/'
    dataset = ShapeNet(test_path, n_points=2048)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
    print("number of testing data:"+ str(len(dataset)))
    #model = pic2points(num_points=2048)
    model = torch.load('model_q.pkl')
    print('testing......')
    model.eval()

    for i, data in tqdm(enumerate(dataloader, 0)):
        im, filenames = data
        
        im = Variable(im)
        im= im.cuda()
        with torch.no_grad():
            pred = model(im).cpu().numpy()
        for j, filename in enumerate(filenames,0):
            
            np.save('./test_1/'+filename.split('.')[0], pred[j])

if __name__ == '__main__':
    main()