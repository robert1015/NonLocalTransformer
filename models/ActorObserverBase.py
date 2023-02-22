#coding=utf-8
"""
ActorObserver Base model
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.utils import dprint, load_sub_architecture, remove_last_layer
from CBAM import *
import matplotlib.pyplot as plt
import numpy as np
import cv2

plt.switch_backend('agg')

class ActorObserverModel(nn.Module):
    def __init__(self, basenet):
        super(ActorObserverModel, self).__init__()

        self.basenet = basenet
        self.avg = self.basenet.avgpool
        self.fc = self.basenet.fc
        self.basenet.avgpool = nn.Sequential()
        self.basenet.fc = nn.Sequential()
        self.avg_third = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.last_conv = nn.Conv2d(2048, 1, kernel_size=1, stride=1)
        self.relu_saliency = nn.ReLU(inplace=True)


        dim = basenet.outdim #2048 dim
        self.cbam_y = CBAM_Module(dim, reduction=16)  # get
        # self.cbam_x = CBAM_Module(dim, reduction=16)
        # self.cbam_z = CBAM_Module(dim, reduction=16)

        self.firstpos_fc = nn.Sequential(nn.Linear(dim, 1), nn.Tanh())
        self.third_fc = nn.Sequential(nn.Linear(dim, 1), nn.Tanh())
        self.firstneg_fc = self.firstpos_fc

        self.firstpos_scale = nn.Parameter(torch.Tensor([math.log(.5)]))
        self.third_scale = nn.Parameter(torch.Tensor([math.log(.5)]))
        self.firstneg_scale = nn.Parameter(torch.Tensor([math.log(.5)]))


    def showSingle(self,base, input,index,pic):  # 显示单通道图
        input = input[0,:,:,:]
        temp = input.sum(dim = 0)
        temp = temp
        input = temp.detach().numpy()
        ymax = 255
        ymin = 0
        xmax = np.max(input)
        xmin = np.min(input)
        input = np.round( (ymax - ymin) * (input - xmin) / (xmax - xmin) + ymin)
        input = 255-cv2.applyColorMap(input.astype(np.uint8), cv2.COLORMAP_OCEAN)
        base = base[0,...].permute(1,2,0).numpy()
        base = (base-base.min())/(base.max()-base.min())*255
        scale = .9
        input = (input * scale+ base * (1-scale)).astype(np.uint8)
        ax = pic.add_subplot(1, 3, index)
        ax.imshow(input)
        return ax


    def base(self, x, y, z, visualized=False):
        """ assuming:
            x: first person positive
            y: third person
            z: first person negative
        """
        #get 卷积层输出的特征
        base_x =self.basenet(x).reshape([-1,2048,7,7])
        base_y =self.basenet(y).reshape([-1,2048,7,7])
        base_z =self.basenet(z).reshape([-1,2048,7,7])

        #get attention 输出
        # cbam_x, _ = self.cbam_x(base_x)
        cbam_y, cbam_weight_y = self.cbam_y(base_y)
        # cbam_z, _ = self.cbam_z(base_z)

        # get attention map & softmax
        map_y = F.softmax((self.relu_saliency(self.last_conv(cbam_y))), dim=2)

        # 用attention的输出去计算dist
        res_x = self.avg(base_x).view(-1,2048)
        res_y = self.avg_third(map_y * base_y).view(-1, 2048)
        res_z = self.avg(base_z).view(-1, 2048)

        # 用basenet的输出去计算weights
        res_x_ = self.avg(base_x).view(-1, 2048)
        res_y_ = self.avg(base_y).view(-1, 2048)
        res_z_ = self.avg(base_z).view(-1, 2048)


        # 可视化结果
        if visualized:
            unMaxpooling = nn.UpsamplingBilinear2d(scale_factor=32)
            pic = plt.figure()
            ax2 = self.showSingle(x.cpu(), unMaxpooling(base_x).cpu(), 1,pic)
            ax2.set_axis_off()
            ax3 = self.showSingle(y.cpu(), unMaxpooling(base_y).cpu(), 2,pic)
            ax3.set_axis_off()
            ax4 = self.showSingle(z.cpu(), unMaxpooling(base_z).cpu(), 3,pic)
            ax4.set_axis_off()
        #ax4 = self.showSingle((map_y * base_y).cpu(), 3,pic)
        #ax4.set_title('map_y * base_y')
        #plt.savefig('/home/yhy/nerner.jpg')
        # del base_x, base_y, base_z, cbam_x, cbam_y, cbam_z

        dist_a = F.pairwise_distance(res_x, res_y, 2).view(-1)
        dist_b = F.pairwise_distance(res_y, res_z, 2).view(-1)


        # dprint('fc7 norms: {} \t {} \t {}', base_x.data.norm(), base_y.data.norm(), base_z.data.norm())
        # dprint('pairwise dist means: {} \t {}\t {}', dist_a.data.mean(), dist_b.data.mean(),dist_c.data.mean())
        if visualized:
           return res_x_, res_y_, res_z_, dist_a, dist_b, pic
        return res_x_, res_y_, res_z_, dist_a, dist_b

    def verbose(self):
        dprint('scales:{}\t{}\t{}',
               math.exp(self.firstpos_scale.item()),
               math.exp(self.third_scale.item()),
               math.exp(self.firstneg_scale.item()))

    def forward(self, x, y, z, visualized=False, feature=False):
        """ assuming:
            x: first person positive
            y: third person
            z: first person negative
            dist_a : x y
            dist_b : y z
        """
        if feature:
            return self.feature(x, y, z)
        if visualized:
            base_x, base_y, base_z, dist_a, dist_b, pic= self.base(x, y, z, True)
        else:
            base_x, base_y, base_z, dist_a, dist_b= self.base(x, y, z)
        w_x = self.firstpos_fc(base_x).view(-1) * torch.exp(self.firstpos_scale)  # fc
        w_y = self.third_fc(base_y).view(-1) * torch.exp(self.third_scale)
        w_z = self.firstneg_fc(base_z).view(-1) * torch.exp(self.firstneg_scale)
        # self.verbose()
        if visualized:
            return dist_a, dist_b,  w_x, w_y, w_z, pic
        return dist_a, dist_b,  w_x, w_y, w_z

    def feature(self, x, y, z):
        base_x, base_y, base_z, dist_a, dist_b= self.base(x, y, z)
        return base_x, base_y, base_z, dist_a, dist_b

class ActorObserverBase(ActorObserverModel):
    def __init__(self, args):
        model = load_sub_architecture(args)
        remove_last_layer(model)
        super(ActorObserverBase, self).__init__(model)

