#coding=utf-8

"""
    Defines tasks for evaluation
"""

import torch
import numpy as np
import random
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from itertools import groupby
import torchvision.transforms as transforms
from train import parse
from utils.utils import MedianMeter, Timer
from models.ActorObserverFC7 import ActorObserverFC7
import torch.nn.functional as F
import itertools



def fc7list2mat(grp, dist=lambda x, y: np.linalg.norm(x - y)):
    ids, fc7s = zip(*list(grp))
    third, first = zip(*fc7s)
    n = len(ids)
    mat = np.zeros((n, n))
    for i, x in enumerate(third):
        for j, y in enumerate(first):
            mat[i, j] = dist(x,y)
    return ids, mat


def matsmooth(mat, winsize):
    m, n = mat.shape
    out = mat.copy()
    for aa in range(m):
        for bb in range(n):
            a = max(0, aa - winsize)
            a2 = min(m, aa + winsize + 1)
            b = max(0, bb - winsize)
            b2 = min(n, bb + winsize + 1)
            out[aa, bb] = mat[a:a2, b:b2].mean()
    return out


def best_one_sec_moment(mat, winsize=6):
    # assuming 6 fps
    m, n = mat.shape
    mat = matsmooth(mat, winsize)
    i, j = random.choice(np.argwhere(mat == mat.min()))
    print ('best moment',i,j)
    gt = i / float(m) * n / 6.   # gt 是 groudtruth
    return i / float(m), j / float(n), i / 6., j / 6., gt

def showSingle(input):  # 显示单通道图
    temp = input.sum(dim = 0)
    output = temp.detach().numpy()
    return output


def vis_match_res(x, index, first_fetru, third_fetru):
    path = '/home/yhy/actor-observer-master/datasets/CharadesEgo_v1_rgb/'
    save_path = '/home/yhy/alignment_pic/'
    tran1 = transforms.Resize(256)
    tran2 = transforms.CenterCrop(224)

    for i in range(first_fetru.size()[0]):
        video_id = x[2].items()[5][1][i]
        firsttime_pos = str(x[2].items()[1][1].numpy()[i] + 1).zfill(6)
        thirdtime = str(x[2].items()[0][1].numpy()[i] + 1).zfill(6)
        firstframe_path_pos = path + video_id + 'EGO' + '/' + video_id + 'EGO-' + firsttime_pos + '.jpg'
        thirdframe_path = path + video_id + '/' + video_id + '-' + thirdtime + '.jpg'
        firstimg_pos = tran2(tran1(Image.open(firstframe_path_pos))).convert('RGBA')
        thirdimg = tran2(tran1(Image.open(thirdframe_path))).convert('RGBA')

        first_feature = showSingle(first_fetru[i, :, :, :].cpu())
        third_feature = showSingle(third_fetru[i, :, :, :].cpu())

        pic = plt.figure()
        ax1 = pic.add_subplot(2, 2, 1)
        ax1.imshow(firstimg_pos)
        ax2 = pic.add_subplot(2, 2, 2)
        ax2.imshow(thirdimg)
        ax3 = pic.add_subplot(2, 2, 3)
        ax3.imshow(first_feature)
        ax3 = pic.add_subplot(2, 2, 4)
        ax3.imshow(third_feature)

        pic.savefig(save_path + video_id + '_' + str(index) +'_' +str(i)+ '.png')
        plt.cla()
        plt.close("all")


    return 0


def alignment(loader, model, epoch, args, task=best_one_sec_moment):
    timer = Timer()
    abssec = MedianMeter()
    abssec0 = MedianMeter()
    randsec = MedianMeter()
    model = ActorObserverFC7(model)

    # switch to evaluate mode
    model.eval()
    def part(x):
        return itertools.islice(x, int(len(x) * 0.3))
    def fc7_generator():
        for i, x in enumerate(part(loader)):  # loader = tester pic number
            inputs, target, meta = parse(x)
            target = target.long().cuda(async=True)

            input_vars = [torch.autograd.Variable(inp.cuda(),requires_grad=False)
                          for inp in inputs]
            # torch.autograd zip the training info
            # torch.autograd.data get tensor value
            # torch.autograd.grad get training grad
            first_fc7, third_fc7, w_x, w_y ,vis_x,vis_y = model(*input_vars)
            vis_match_res(x,i, vis_x, vis_y)

           # print (i,first_fc7.size())

            timer.tic()
            """
            if i % args.print_freq == 0:
                print('Alignment: [{0}/{1}]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f})'.format(
                          i, len(loader), timer=timer))"""
            for vid, o1, o2 in zip(meta['id'], first_fc7, third_fc7):
                yield vid, (o1.data.cpu().numpy(), o2.data.cpu().numpy())

    for key, grp in groupby(fc7_generator(), key=lambda x: x[0]):

        print('processing id: {}'.format(key))
        _, mat = fc7list2mat(grp)
        _, _, _, j, gt = task(mat, winsize=3)
        _, _, _, j0, gt0 = task(mat, winsize=0)
        _, _, _, jr, gtr = task(np.random.randn(*mat.shape), winsize=3)
        abssec.update(abs(j - gt))
        abssec0.update(abs(j0 - gt0))
        randsec.update(abs(jr - gtr))

        print('  abs3: {abs3.val:.3f} ({abs3.avg:.3f}) [{abs3.med:.3f}]'
              '  abs0: {abs0.val:.3f} ({abs0.avg:.3f}) [{abs0.med:.3f}]'
              '\n'
              '  absr: {absr.val:.3f} ({absr.avg:.3f}) [{absr.med:.3f}]'.format(
                  abs3=abssec, abs0=abssec0, absr=randsec))


    return abssec.med


