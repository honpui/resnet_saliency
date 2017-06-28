import sys
sys.path.append('./util')
sys.path.append('./model')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from model import resnet50
from torch.autograd import Variable
import numpy as np
import argparse
import cv2
from glob import glob
import os

parser = argparse.ArgumentParser(description='Creating saliency maps for VQA')
parser.add_argument('--weights', type=str, default='./checkpoints/model_epoch_0.pth', help='Specififying weights to be loaded')
args = parser.parse_args()

def get_map(input_):
    input_ = input_.view(input_.size(0), -1)
    input_ = input_.cpu().numpy()
    softmax = np.exp(input_-np.max(input_))/np.sum(np.exp(input_-np.max(input_)))

    return softmax

def main():
    mean = np.load('mean.npy')
    mean = mean.reshape((1,1,3))
    std = np.load('std.npy')
    std = std.reshape((1,1,3))
    img_rows,img_cols = 448, 448
    batch_size = 1
    #loading data and model
    model = resnet50()
    model.load_state_dict(torch.load(args.weights)) #loading pretrained weights
    model.cuda()
    train_dir = '/home/luoyan/project/dataset/mscoco/images/train2014'
    test_dir = '/home/luoyan/project/dataset/mscoco/images/val2014'
    train_dir = glob(os.path.join(train_dir,'*.jpg'))
    test_dir = glob(os.path.join(test_dir,'*.jpg'))
    #main loop for computing result
    model.eval()

    train_map = dict()
    test_map = dict()
    #saving training data
    print 'Start generating saliency maps for training data'
    for i,cur_train in enumerate(train_dir):
        data = cv2.imread(cur_train)
        data = cv2.resize(data, (448, 448),interpolation = cv2.INTER_LINEAR)
        data = (data-mean)/std
        data = data.reshape((1,data.shape[0],data.shape[1],data.shape[2]))
        data = data.transpose((0, 3, 1, 2))
        data = torch.from_numpy(data)
        data = Variable(data,volatile=True)
        data = data.cuda()
        output = model(data)

        pred = get_map(output.data)
        pred = pred.reshape((img_rows/32*img_cols/32,1))
        cur_name = os.path.basename(cur_train)[15:-4]
        train_map[cur_name] = pred

        if i%5000 == 0:
            print "%d out of %d saliency maps have been generated" %(i,len(train_dir))

    #saving test data
    print 'Start generating saliency maps for validation data'
    for i,cur_test in enumerate(test_dir):
        data = cv2.imread(str(cur_test))
        data = cv2.resize(data, (448, 448),interpolation = cv2.INTER_LINEAR)
        data = (data-mean)/std
        data = data.reshape((1,data.shape[0],data.shape[1],data.shape[2]))
        data = data.transpose((0, 3, 1, 2))
        data = torch.from_numpy(data)
        data = Variable(data,volatile=True)
        data = data.cuda()
        output = model(data)

        pred = get_map(output.data)
        pred = pred.reshape((img_rows/32*img_cols/32,1))
        cur_name = os.path.basename(cur_test)[13:-4]
        test_map[cur_name]= pred

        if i%5000 == 0:
            print "%d out of %d saliency maps have been generated" %(i,len(test_dir))

    np.save('mscoco_saliency_train',train_map)
    np.save('mscoco_saliency_val',test_map)

main()
