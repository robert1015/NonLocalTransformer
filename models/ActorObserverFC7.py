#coding=utf-8

"""
   Use ActorObserver model as first person classifier
"""


import torch
from models.ActorObserverBase import ActorObserverBase
from models.utils import dprint
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



class ActorObserverFC7(ActorObserverBase):
    def __init__(self, args):

        if isinstance(args, dict):
            super(ActorObserverFC7, self).__init__(args)
        else:
            if 'DataParallel' in args.__class__.__name__:
                args = args.module
            print('Initializing FC7 extractor with AOB instance')
            self.__dict__ = args.__dict__


    def showSingle(self,input,index,pic):  # 显示单通道图
        input = input[0,:,:,:]
        temp = input[0 ,  :, :]
        for i in range (len(input)-1):
            temp = temp + input[i+1,  :, :]
        temp = temp/len(input)
        input = temp.detach().numpy()
        ax = pic.add_subplot(2, 2, index)
        ax.imshow(input)
        return ax

    def showOrigin(self,input,index):  # 显示原图（3通道）
        img = input.cpu().detach().numpy()
        img = img [0,:,:,:]
        img = np.transpose(img, (1, 2, 0))
        ax = plt.subplot(2, 3, index)
        plt.imshow(img)
        return ax

    def backwoad_basenet(self,input):  # 反推卷积部分

        unMaxpooling = nn.UpsamplingBilinear2d(scale_factor=2)
        input = input.type(torch.FloatTensor)
        relu = nn.ReLU(inplace=True)
        self.showSingle(input,1) #1

        out = F.conv_transpose2d(input.cuda(), self.basenet.layer4[0].downsample[0].weight, bias=None) # transLayer4
        out = unMaxpooling(out)
        out = relu(out)
        self.showSingle(out.cpu(),2) #2

        out = F.conv_transpose2d(out.cuda(), self.basenet.layer3[0].downsample[0].weight,  bias=None) # transLayer3
        out = unMaxpooling(out)
        self.showSingle(out.cpu(),3) #3

        out = F.conv_transpose2d(out.cuda(), self.basenet.layer2[0].downsample[0].weight,bias=None) # transLayer2
        out = unMaxpooling(out)
        self.showSingle(out.cpu(),4) #4

        out = F.conv_transpose2d(out.cuda(), self.basenet.layer1[0].downsample[0].weight,  bias=None) # transLayer1
        out = unMaxpooling(out)
        out = relu(out)
        out = F.conv_transpose2d(out.cuda(), self.basenet.conv1.weight,stride=2, padding=3, bias=None)
        self.showSingle(out.cpu(),5) #5

        plt.show()

        return out


    def forward(self, x, y, z):
        """ assuming:
            x: first person positive
            y: third person
            z: first person negative
        """
        base_x = self.basenet(x).reshape([-1,2048,7,7])
        base_y = self.basenet(y).reshape([-1,2048,7,7])

       # cbam_x, _ = self.cbam_x(base_x)
        cbam_y, _ = self.cbam_y(base_y)

        map_y = F.softmax((self.relu_saliency(self.last_conv(cbam_y))), dim=2)

        res_x =self.avg(base_x).view(-1,2048)
        res_y = self.avg_third(map_y * base_y).view(-1, 2048)

        res_x_ = self.avg(base_x).view(-1, 2048)
        res_y_ = self.avg(base_y).view(-1, 2048)

        w_x = self.firstpos_fc(res_x_).view(-1) * torch.exp(self.firstpos_scale)
        w_y = self.third_fc(res_y_).view(-1)* torch.exp(self.third_scale)
       # dprint('fc7 norms: {}\t {}', base_x.data.norm(), base_y.data.norm())
       # self.verbose()

        return res_x, res_y, w_x, w_y ,base_x , map_y * base_y
