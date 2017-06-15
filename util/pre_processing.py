from salicon.salicon import SALICON
import numpy as np
import os
import cv2

#generating data,target for batch training
class Batch_generator:
    def __init__(self,img_rows,img_cols,mode,batch_size):
        self.data=[]
        self.anno=[]
        self.batch_size=batch_size
        self.mode = mode
        if os.path.exists('/media/eric/New Volume1/Vision/saliency/features/'+mode+'_features.npy'):
            self.data = np.load('/media/eric/New Volume1/Vision/saliency/features/'+mode+'_features.npy')
            self.anno = np.load('/media/eric/New Volume1/Vision/saliency/features/'+mode+'_anno.npy')
        else:
            self.init_data(img_rows,img_cols,self.mode)
            np.save('/media/eric/New Volume1/Vision/saliency/features/'+mode+'_features',self.data)
            np.save('/media/eric/New Volume1/Vision/saliency/features/'+mode+'_anno',self.anno)
        self.data_size = len(self.data)
        self.iter_num = self.data_size/self.batch_size

    #loading data using salicon API
    def init_data(self,img_rows,img_cols,mode):
        print('reading training data')
        dataType = mode+'2014'
        annFile = '/home/eric/Desktop/experiment/salicon/salicon-api/annotations/fixations_%s.json' %dataType
        salicon = SALICON(annFile)
        imgIds = salicon.getImgIds();
        img = salicon.loadImgs(imgIds[:])

        data = []
        anno = []
        for i in xrange(len(img)):
            #loading img_data
            I = cv2.imread('/home/eric/Desktop/experiment/salicon/salicon-api/images/'+mode+'/%s'%img[i]['file_name'])
            I = cv2.resize(I, (img_cols, img_rows),interpolation = cv2.INTER_LINEAR)
            data.append(I)
            #loading annotations
            annIds = salicon.getAnnIds(imgIds=img[i]['id'])
            anns = salicon.loadAnns(annIds)
            sal_map = salicon.showAnns(anns)
            sal_map = cv2.resize(sal_map, (img_cols/32, img_rows/32),interpolation = cv2.INTER_LINEAR) #resize the saliency map
            anno.append(sal_map)
        data = np.array(data, dtype=np.float32)
        data = data.transpose((0, 3, 1, 2))
        anno = np.array(anno, dtype=np.float32)
        anno = anno.reshape(anno.shape[0],(img_rows/32)*(img_cols/32)) #converting spatial saliency map into vectors

        #preprocessing data
        #normalize the img
        for i in xrange(3):
            data[:,i,:,:] -= np.mean(data[:,i,:,:])
            data[:,i,:,:] /= np.std(data[:,i,:,:])
        #normalize the saliency map for training with cross entropy
        for i in xrange(len(anno)):
            # cur_map = (cur_map-np.min(cur_map))/(np.max(cur_map)-np.min(cur_map)) #data v2
            anno[i] = anno[i]/np.sum(anno[i]) #data v1

        self.data = data
        self.anno = anno

    def get_batch(self,batch_num):
        if batch_num == self.iter_num-1:
            current_data = self.data[batch_num*self.batch_size:]
            current_anno = self.anno[batch_num*self.batch_size:]
            # if self.mode == 'train':
            #     self.shuffle()
        else:
            current_data = self.data[batch_num*self.batch_size:(batch_num+1)*self.batch_size]
            current_anno = self.anno[batch_num*self.batch_size:(batch_num+1)*self.batch_size]

        if len(current_data.shape)<4:
            current_data = current_data.reshape(1,current_data.shape[0],current_data.shape[1],current_data.shape[2])
        if len(current_anno.shape)<2:
            current_anno = current_anno.reshape(1,current_anno.shape[0])

        return current_data, current_anno

    #randomly shuffle data for training at the end of each epoch
    def shuffle(self,):
        random = np.arange(self.data_size)
        np.random.shuffle(random)
        self.data = self.data[random]
        self.anno = self.anno[random]
