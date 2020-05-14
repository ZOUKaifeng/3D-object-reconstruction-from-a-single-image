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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.cuda as cuda
from pic2points import pic2points
from torch.nn.parallel import DataParallel
from sklearn.model_selection import train_test_split
from data_loader import ShapeNet
from torch.autograd import Variable
import torch
import model.chamfer.dist_chamfer as ext
import matplotlib.pyplot as plt
import neuralnet_pytorch as nnt
distChamfer = ext.chamferDist()

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--num_points', type=int, default=2048, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--nepoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='',  help='model path')

def batch_pairwise_dist(x, y):
    # 32, 2500, 3
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P


def batch_NN_loss(x, y):
    bs, num_points, points_dim = x.size()
    dist1 = torch.sqrt(batch_pairwise_dist(x, y))
    values1, indices1 = dist1.min(dim=2)

    dist2 = torch.sqrt(batch_pairwise_dist(y, x))
    values2, indices2 = dist2.min(dim=2)
    a = torch.div(torch.sum(values1,1), num_points)
    b = torch.div(torch.sum(values2,1), num_points)
    sum = torch.div(torch.sum(a), bs) + torch.div(torch.sum(b), bs)

    return sum


def main():
   
    global args
    opt = parser.parse_args()
    print (opt)
    opt.dataroot = "/home/zou/img2obj/train_data.json"
    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    with open(opt.dataroot,'r') as load_f:
        load_dict = json.load(load_f)
    train_dict, test_dict = train_test_split(load_dict, test_size=0.2)

    dataset = ShapeNet(train_dict, type='train', n_points=opt.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    print("number of training data:"+ str(len(dataset)))

    test_dataset = ShapeNet(test_dict, type='train', n_points=opt.num_points)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,shuffle=True, num_workers=int(opt.workers))
    print("number of training data:"+ str(len(test_dataset)))

    # creat model
    print("model building...")
    model = pic2points(num_points=opt.num_points)
    model.cuda()
    
    # load pre-existing weights
    if opt.model != '':
        model.load_state_dict(torch.load(opt.model))

    optimizer = torch.optim.Adam(model.parameters(),lr = 5e-4, weight_decay=1e-4)
    num_batch = len(dataset) / opt.batchSize
    best_loss = 100
    

    visual_loss_train = []
    visual_loss_test = []
    print('training mode ------------------')
    for epoch in range(opt.nepoch):
        loss_epoch_train = 0
        loss_epoch_test = 0
        print("epoch:"+str(epoch))
        model.train()
        if epoch > 20:
            optimizer.param_groups[0]['lr'] = 1e-5
        for i, data in enumerate(dataloader, 0):
            _, im, points = data
            im, points = Variable(im), Variable(points)
            im, points = im.cuda(), points.cuda()
            pred = model(im)
            #loss = batch_NN_loss(pred, points).cuda()
            loss_1 =  nnt.chamfer_loss(pred, points, reduce='mean')
            loss_2 =  nnt.emd_loss(pred, points, reduce='mean')            

            loss = 100*loss_1 + loss_2/100
           # print('chamfer', loss_1, 'emd_loss', loss_2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            loss_epoch_train = loss_epoch_train + loss.cpu().detach().numpy()

            if i % 50 is 0:
                print("training loss is:", loss.cpu().detach().numpy())

        print("epoch {} training loss is : {}".format(epoch, loss_epoch_train/(len(dataloader))))

        visual_loss_train.append(loss_epoch_train/(len(dataloader)))
        loss_test = 0
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(testdataloader, 0):
                _, im_test, points_test = data
                im_test, points_test = Variable(im_test), Variable(points_test)
                im_test, points_test = im_test.cuda(), points_test.cuda()
                pred_test = model(im_test)
              #  loss_test = batch_NN_loss(pred_test, points_test).cuda()
                loss_1 =  nnt.chamfer_loss(pred_test, points_test, reduce='mean')
                loss_2 =  nnt.emd_loss(pred_test, points_test, reduce='mean') 
                #dist1, dist2 = distChamfer(points_test, pred_test)
                loss_test = 100*loss_1 + loss_2/100
                loss_epoch_test = loss_epoch_test + loss_test

        print("epoch {} testing loss is : {}".format(epoch, loss_epoch_test/(len(testdataloader))))
        visual_loss_test.append(loss_epoch_test/(len(testdataloader)))

        if visual_loss_test[-1] < best_loss:
            best_loss = visual_loss_test[-1]
            torch.save(model, './results/model_1.pkl')
            print('save model, the best loss is', best_loss)


    times = list(range(len(visual_loss_test)))
        
    fig = plt.figure()
    ax.plot(times, visual_loss_test)
    ax.plot(times, visual_loss_train)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    plt.savefig('./results/run1.png')
    plt.show()



if __name__ == '__main__':
    num_cuda = cuda.device_count()
    print("number of GPUs have been detected:" + str(num_cuda))
    with torch.cuda.device(0):
        main()