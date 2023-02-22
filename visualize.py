import sys
import os
import torch
import numpy as np
import random
import train
from opts import parse
from models import create_model
from datasets.get import get_dataset
import itertools
import matplotlib.pyplot as plt
import importlib

import copy
plt.switch_backend('agg')

name = __file__.split('/')[-1].split('.')[0]  # name is filename

global args, best_top1
args = [
    '--name', name, 
    '--print-freq', '50',
    # '--train-file', '/home/yhy/actor-observer-master/datasets/labels/CharadesEgo_v1_train_new.csv',
    # '--val-file', '/home/yhy/actor-observer-master/datasets/labels/CharadesEgo_v1_test_new.csv',
    '--dataset', 'charadesego',
    # '--data', '/home/yhy/actor-observer-master/datasets/CharadesEgo_v1_rgb/',
    '--arch', 'ActorObserverBaseNoShare',
    '--subarch', 'resnet152',
    # '--pretrained-subweights', '/home/zzhehao/JointAttention/exp/resnet152-b121ed2d.pth',
    '--loss', 'ActorObserverLossAll',
    '--subloss', 'DistRatio',
    '--subloss2', 'AttentionDist',
    '--decay', '0.95',
    '--lr', '4e-5',
    '--lr-decay-rate', '15',
    '--batch-size', '1',
    '--train-size', '0.1',
    '--val-size', '0.2',
    '--cache-dir', '/home/zzhehao/JointAttention/result/',
    '--epochs', '20',
    # '--evaluate',
    '--alignment',
    # '--usersalignment',
]
sys.argv.extend(args)

def parse_sample(x):
    return x[0], x[1], x[2] if len(x) > 2 else {'id': [1] * x[1].shape[0]}

def forward(inputs, target, model, criterion, ids, train=True, visualized = True):
    target = target.float().cuda(async=True)
    # volatile
   # torch.set_grad_enabled(train)
    input_vars = [torch.autograd.Variable(inp.cuda(), requires_grad=train)
                          for inp in inputs]
    target_var = torch.autograd.Variable(target, requires_grad=train)
    output = model(input_vars[0], input_vars[1], input_vars[2], visualized= visualized) # dist_a, dist_b,  w_x, w_y, w_z,pic


    loss, weights = criterion(*(list(output[0:5]) + [target_var, ids]))
    if visualized:
        return output[:2], loss, weights, output[-1]
    else:
        return output[:2], loss, weights

def get_original_image(meta):
    video_id = meta['id'][0]
    firsttime_neg = str(meta['firsttime_neg'].numpy()[0] + 1).zfill(6)
    firsttime_pos = str(meta['firsttime_pos'].numpy()[0] + 1).zfill(6)
    thirdtime = str(meta['thirdtime'].numpy()[0] + 1).zfill(6)
    path = '/home/zzhehao/CharadesEgo/CharadesEgo_v1_rgb/'
    firstframe_path_pos = path + video_id + 'EGO' + '/' + video_id + 'EGO-' + firsttime_pos+ '.jpg'
    firstframe_path_neg = path + video_id + 'EGO' + '/' + video_id + 'EGO-' + firsttime_neg + '.jpg'
    thirdframe_path = path + video_id + '/' + video_id + '-' + thirdtime + '.jpg'
    pic1 = plt.figure()
    ax1 = pic1.add_subplot(1, 3, 1)
    ax1.imshow(plt.imread(firstframe_path_pos))
    ax1.set_axis_off()
    ax2 = pic1.add_subplot(1, 3, 2)
    ax2.imshow(plt.imread(thirdframe_path))
    ax2.set_axis_off()
    ax3 = pic1.add_subplot(1, 3, 3)
    ax3.imshow(plt.imread(firstframe_path_neg))
    ax3.set_axis_off()
    return pic1

def case_getattr(obj, attr):
    casemap = {}
    for x in obj.__dict__:
        casemap[x.lower()] = x
    return getattr(obj, casemap[attr.lower()])

def get_val_dataset(args):
    dataset = importlib.import_module('.' + args.dataset.lower(), package='datasets')
    datasets = case_getattr(dataset, args.dataset).get(args)
    val_dataset = datasets[0]
    return val_dataset

def preprocessing(x):
    inputs, target, meta = parse_sample(x)
    inputs = [i.unsqueeze(0) for i in inputs]
    target = torch.tensor([target])
    meta1=copy.deepcopy(meta)
    for k in meta1.keys():
        v = meta1[k]
        meta1[k] = torch.tensor([v]) if isinstance(v, int) else [v]
    return inputs, target, meta1

if __name__ == '__main__':
    
    args = parse()
    model, criterion, optimizer = create_model(args)


    # val_dataset = get_val_dataset(args)
    # for i, x in enumerate(val_dataset):
    #     inputs, target, meta = preprocessing(x)
    #     if i == 0:
    #         print(inputs[0].shape) 
    #         print(target)
    #         print(meta)
    #     if i > 10:
    #         break
    
    # for i, x in enumerate(val_dataset):
    #     inputs, target, meta = preprocessing(x)
    #     if i == 0:
    #         print(inputs[0].shape) 
    #         print(target)
    #         print(meta)
    #     if i > 10:
    #         break

    # exit()
    # print(len(val_dataset))
    # inputs, target, meta = parse_sample(val_dataset[0])
    # inputs = [i.unsqueeze(0) for i in inputs]
    # target = torch.tensor([target])
    # for k in meta.keys():
    #     v = meta[k]
    #     meta[k] = torch.tensor([v]) if isinstance(v, int) else [v]
    # print(inputs[0][0,0,0,0:10]) 
    # print(target)
    # print(meta)
    # print("\n\n\n")
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=0, pin_memory=True)
    # for x in val_loader:
    #     inputs, target, meta = parse_sample(x)
    #     print(inputs[0][0,0,0,0:10]) 
    #     print(target)
    #     print(meta)
    #     break
    # exit()



    model.load_state_dict(torch.load('70model.pth.tar')['state_dict'])
    model.eval()

    # losses = AverageMeter()
    # top1 = AverageMeter()
    # wtop1 = AverageMeter()
    allweights = []

    val_dataset = get_val_dataset(args)
    print(len(val_dataset))
    # def part(x):
    #         return itertools.islice(x, int(len(x) * args.val_size))

    dataset_size =  int(len(val_dataset) * args.val_size)
    print("total: ", dataset_size)
    sample_size = 5000
    topk5 = int(np.ceil(.02 * sample_size))
    # for i, x in enumerate(val_dataset):
    #     print(i)
    #     if i >= sample_size:
    #         break
    #     inputs, target, meta = preprocessing(x)
    #     output, loss, weights = forward(inputs, target, model, criterion, meta['id'], train=False, visualized=False)
    #     # print(output, loss, weights)
    #     # break
    #     allweights.append(weights.item())
    #     x0, y0 = output
    #     correct = x0 < y0 if target[0] > 0 else y0 < x0
        
    # allweights = np.array(allweights)
    # ind = np.argsort(allweights)
    # ind = ind[-topk5:].tolist()
    # ind = set(ind)

    # print(ind)
    # with open('/home/zzhehao/JointAttention/result/third_to_first_person/visualize/indices.txt', 'w+') as f:
    #     f.write(str(ind))
    # with open('/home/zzhehao/JointAttention/result/third_to_first_person/visualize/indices.txt', 'r') as f:
    #     ind = f.read()
    #     ind = list(map(int, ind[5:][:-2].split(',')))
    ind=[405, 609, 762, 1574, 2422, 353, 4786, 7116, 8408, 11736, 13525, 13772, 15918, 23942, 24494, 27511]
    # ind=[405, 609, 762, 1574, 2422]

    def  norm(x):
        return (255*(x-x.min())/(x.max()-x.min())).astype(np.uint8)
    num = 0
    for i in ind:
        inputs, target, meta = preprocessing(val_dataset[i])
        pic_ori = get_original_image(meta)

        output, loss, weights, pic = forward(inputs, target, model, criterion, meta['id'], train=False)
        pic1 = plt.figure()
        ax1 = pic1.add_subplot(1, 3, 1)
        ax1.imshow(norm(inputs[0][0,...].permute(1,2,0).numpy()))
        ax1.set_axis_off()
        ax2 = pic1.add_subplot(1, 3, 2)
        ax2.imshow(norm(inputs[1][0,...].permute(1,2,0).numpy()))
        ax2.set_axis_off()
        ax3 = pic1.add_subplot(1, 3, 3)
        ax3.imshow(norm(inputs[2][0,...].permute(1,2,0).numpy()))
        ax3.set_axis_off()
        x, y = output
        correct = x < y if target[0] > 0 else y < x 

        # if correct:
        #     path = "/home/zzhehao/JointAttention/result/third_to_first_person/visualize/pos/"
        # else:
        #     path = "/home/zzhehao/JointAttention/result/third_to_first_person/visualize/neg/"
        path = '/home/zzhehao/JointAttention/result/temp_70_hot/'
        if not os.path.exists(path):
            os.makedirs(path)

        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        pic.savefig(path+"%03d_attention.jpg"%(i), bbox_inches='tight', pad_inches=0)
        pic_ori.savefig(path+"%03d_frame.jpg"%(i), bbox_inches='tight', pad_inches=0)
        pic1.savefig(path+"%03d_input.jpg"%(i), bbox_inches='tight', pad_inches=0)
        print(str(i)+' finished')
