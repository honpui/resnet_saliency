import sys
sys.path.append('./util')
sys.path.append('./model')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from pre_processing import Batch_generator
from model import resnet50
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Saliency Prediction With ResNet-50')
parser.add_argument('--mode', type=str, default='train', help='Selecting running mode (default: train)')
parser.add_argument('--weights', type=str, default=None, help='Specififying the directory for the weights to be loaded (default: None)')
parser.add_argument('--save_result', type=bool, default=True, help='Enable to save predicted map and ground truth (default: True)')
parser.add_argument('--res_dir', type=str, default='./res', help='Directory to save results (default: ./res)')
args = parser.parse_args()

def logsoftmax_cross_entropy(input_,target):
    "original cross entropy loss only works with one-hot vetcor, write an own one"
    epsilon = 2.22044604925031308e-16
    input_ = input_.view(input_.size(0), -1)
    softmax = torch.exp(input_-torch.max(input_).expand_as(input_))/torch.sum(torch.exp(input_-torch.max(input_).expand_as(input_))).expand_as(input_)
    loss = -torch.mean(target*torch.log(torch.clamp(softmax,min=epsilon,max=1)))
    loss = torch.mean(loss)
    return loss

def log(file_name,msg):
    log_file = open(file_name,"a")
    log_file.write(msg+'\n')
    log_file.close()

def adjust_learning_rate(optimizer, epoch):
    "adatively adjust lr based on epoch"
    if epoch < 10:
        lr = 2.5e-3
    elif epoch <= 20:
        lr = 2.5e-3 * (10 ** (float(epoch-10) / 10))
    elif epoch<=40:
        lr = 2.5e-2 * (0.1 ** (float(epoch-20) / 20))
    else:
        lr = 2.5e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    #parameters
    nb_epoch = 100
    img_rows,img_cols = 800, 608
    batch_size = 1
    #initialize data loader
    train_DataLoader = Batch_generator(img_rows,img_cols,mode='train',batch_size=batch_size)
    val_DataLoader = Batch_generator(img_rows,img_cols,mode='val',batch_size=batch_size)

    #defining model and optimizer
    model = resnet50()
    model.load_state_dict(torch.load('resnet_pretrained.pth')) #loading pretrained weights
    model.cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=2.5e-4, weight_decay=5e-4,momentum=0.9,nesterov=True) #0.001

    #private function for training and evaluation
    def train(epoch):
        model.train()
        random = np.arange(train_DataLoader.iter_num)
        np.random.shuffle(random)

        for i,batch_idx in enumerate(random):
            data, target = train_DataLoader.get_batch(batch_idx)
            data, target = torch.from_numpy(data),torch.from_numpy(target)
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss=logsoftmax_cross_entropy(output,target)
            # loss = F.cross_entropy(output,target)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0 or i == train_DataLoader.iter_num-1:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), train_DataLoader.data_size,
                    100. * i * len(data)/ train_DataLoader.data_size, loss.data[0])
                print msg
                log('training_loss_v4.log',msg)

    def test(epoch):
        model.eval()
        total_loss = []
        for batch_idx in xrange(val_DataLoader.iter_num):
            data, target = val_DataLoader.get_batch(batch_idx)
            data, target = torch.from_numpy(data),torch.from_numpy(target)
            data, target = Variable(data,volatile=True), Variable(target,volatile=True)
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # loss=criterion(output,target.long())
            loss = logsoftmax_cross_entropy(output,target)
            total_loss.append(loss.data[0])
        total_loss = np.mean(total_loss)
        msg = 'Validation loss for Epoch %d is %f' %(epoch,total_loss)
        print msg
        log('validation_loss_v4.log',msg)


    #main loop for training:
    print 'Start training model'
    for epoch in xrange(nb_epoch):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
        #save check point
        checkpoint = './checkpoints_v4/model_epoch_' + str(epoch) +'.pth'
        torch.save(model.state_dict(),checkpoint)

def eval_():
    img_rows,img_cols = 800, 608
    batch_size = 1
    #loading data and model
    val_DataLoader = Batch_generator(img_rows,img_cols,mode='val',batch_size=batch_size)
    model = resnet50()
    model.load_state_dict(torch.load(args.weights)) #loading pretrained weights
    model.cuda()

    if args.save_result == True:
        if not os.path.exists(args.res_dir):
            os.mkdir(args.res_dir)

    #main loop for computing result
    model.eval()
    total_loss = []
    print 'Start evaluation'
    for batch_idx in xrange(val_DataLoader.iter_num):
        data, target = val_DataLoader.get_batch(batch_idx)
        data, target = torch.from_numpy(data),torch.from_numpy(target)
        data, target = Variable(data,volatile=True), Variable(target,volatile=True)
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = logsoftmax_cross_entropy(output,target)
        total_loss.append(loss.data[0])

        if args.save_result:
            pred = output.data.cpu().numpy()
            pred = pred.reshape((img_rows/32,img_cols/32))
            gt = target.data.cpu().numpy()
            gt = gt.reshape((img_rows/32,img_cols/32))

            #normalize the maps
            pred = (pred - np.min(pred))/(np.max(pred)-np.min(pred))
            gt = (gt - np.min(gt))/(np.max(gt)-np.min(gt))
            pred*=255
            gt*=255


            cv2.imwrite(os.path.join(args.res_dir,str(batch_idx)+'_pred.png'),pred)
            cv2.imwrite(os.path.join(args.res_dir,str(batch_idx)+'_gt.png'),gt)

    total_loss = np.mean(total_loss)
    print 'Testing loss is %f' %total_loss


if args.mode =='train':
    train()
else:
    eval_()
