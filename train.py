""" Defines the Trainer class which handles train/validation/validation_video
"""
import torch
import itertools
import numpy as np
import pdb
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils import map as meanap
from utils.utils import dump_gpumem, AverageMeter, submission_file, Timer
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math
import progressbar

plt.switch_backend('agg')

def adjust_learning_rate(startlr, decay_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = startlr * (0.1 ** (epoch // decay_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse(x):
    return x[0], x[1], x[2] if len(x) > 2 else {'id': [1] * x[1].shape[0]}


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def triplet_accuracy(output, target, weights=None):
    """
       if target>0 then first output should be smaller than right output
       optional weighted average
    """
    if type(output) is not list: 
        output = [(x.item(), y.item()) for x, y in zip(*output)]
    correct = [x < y if t > 0 else y < x for (x, y), t in zip(output, target)]
    if weights is None:
        return np.mean(correct)
    else:
        weights = weights.numpy()
        weights = weights / (1e-5 + np.sum(weights))
        return np.sum(np.array(correct).astype(float) * weights) # frame important


def triplet_topk(output, target, weights, topk=5):
    weights = np.array(weights)
    n = weights.shape[0]
    topkn = int(np.ceil(.01 * topk * n))
    ind = np.argsort(weights)
    ind = ind[-topkn:].tolist()
    return triplet_accuracy([output[x] for x in ind], [target[x] for x in ind])


def triplet_allk(output, target, weights):
    out = {}
    for k in (1, 2, 5, 10, 50):
        out['topk{}'.format(k)] = triplet_topk(output, target, weights, topk=k)
    return out


def forward(inputs, target, model, criterion, ids, train=True):
    target = target.float().cuda(async=True)
    # volatile
   # torch.set_grad_enabled(train)
    input_vars = [torch.autograd.Variable(inp.cuda(), requires_grad=train)
                          for inp in inputs]
    target_var = torch.autograd.Variable(target, requires_grad=train)
    output = model(*input_vars) # dist_a, dist_b,  w_x, w_y, w_z,pic

   # pic = output[-1]
    output = output[0:5]

    loss, weights = criterion(*(list(output) + [target_var, ids]))
    return output[:2], loss, weights

def ImgQuality(img_path):

    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    img = np.array(image)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if (tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    Sanno = res

    scharrx = cv2.Scharr(image, cv2.CV_64F, dx=1, dy=0)
    scharrx = cv2.convertScaleAbs(scharrx)
    scharry = cv2.Scharr(image, cv2.CV_64F, dx=0, dy=1)
    scharry = cv2.convertScaleAbs(scharry)
    grad = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
    grad = grad.mean()
    return  Sanno,grad

def FramesList(x):
    video_id = x[2].items()[5][1]
    path = '/home/yhy/actor-observer-master/datasets/CharadesEgo_v1_rgb/'
    Sanno = []
    Grad = []

    for i in range(len(video_id)):
        firsttime_pos = str(x[2].items()[1][1].numpy()[i] +1 ).zfill(6)
        firsttime_neg = str(x[2].items()[2][1].numpy()[i] +1).zfill(6)
        thirdtime = str(x[2].items()[0][1].numpy()[i] +1).zfill(6)

        firstframe_path_pos = path + video_id[i] + 'EGO' + '/' + video_id[i] + 'EGO-' + firsttime_pos + '.jpg'
        firstframe_path_neg = path + video_id[i] + 'EGO' + '/' + video_id[i] + 'EGO-' + firsttime_neg + '.jpg'
        thirdframe_path = path + video_id[i] + '/' + video_id[i] + '-' + thirdtime + '.jpg'

        Sanno_x_ ,Grad_x_ = ImgQuality(firstframe_path_pos)  #higher is better
        Sanno_y_ ,Grad_y_ = ImgQuality(thirdframe_path)
        Sanno_z_ ,Grad_z_ = ImgQuality(firstframe_path_neg)

        Sanno.append((Sanno_x_ + Sanno_y_ +  Sanno_z_)/3.0)
        Grad.append((Grad_x_ + Grad_y_ + Grad_z_)/3.0)

    Sanno = F.softmax(torch.tensor(Sanno),dim=0)+ 0.00001
    Grad = F.softmax(torch.tensor(Grad),dim=0)+ 0.00001

    return Sanno,Grad


class Trainer():
    def train(self, loader, model, criterion, optimizer, epoch, args):
        adjust_learning_rate(args.lr, args.lr_decay_rate, optimizer, epoch)
        timer = Timer()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        wtop1 = AverageMeter()

        #attentiontop1 = AverageMeter()
        metrics = {}

        # switch to train mode
        model.train()
        optimizer.zero_grad()

        def part(x):
            return itertools.islice(x, int(len(x) * args.train_size))

        widgets = ['Progress: ',progressbar.Percentage(), ' ', progressbar.Bar('#'),' ', progressbar.Timer(),    ' ']  
        p = progressbar.ProgressBar(widgets=widgets, maxval=args.print_freq).start()
        for i, x in enumerate(part(loader)):
            inputs, target, meta = parse(x)
            data_time.update(timer.thetime() - timer.end)
           # Sanno, Grad = FramesList(x)
            output, loss, weights  = forward(inputs, target, model, criterion, meta['id'])
            # here is the output
           # weights = weights * Sanno * Grad
            prec1 = triplet_accuracy(output, target)  # acc
            wprec1 = triplet_accuracy(output, target, weights)

            losses.update(loss.item(), inputs[0].size(0))
            top1.update(prec1, inputs[0].size(0))
            wtop1.update(wprec1, inputs[0].size(0))


           # attentiontop1.update(attention_score,inputs[0].size(0))

            loss.backward()
            if i % args.accum_grad == args.accum_grad - 1:
                # print('updating parameters')
                optimizer.step()
                optimizer.zero_grad()

            p.update(i % args.print_freq)
            timer.tic()
            if i % args.print_freq == 0:
                print('[{name}] Epoch: [{0}][{1}/{2}({3})]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'WAcc@1 {wtop1.val:.3f} ({wtop1.avg:.3f})\t'.format(
                          epoch, i, int(len(loader) * args.train_size), len(loader), name=args.name,
                          timer=timer, data_time=data_time, loss=losses, top1=top1, wtop1=wtop1))
        p.finish()

        metrics.update({'top1': top1.avg, 'wtop1': wtop1.avg })


        return metrics

    def validate(self, loader, model, criterion, epoch, args, ignore):
        tran1 = transforms.Resize(256)
        tran2 = transforms.CenterCrop(args.inputsize)
        timer = Timer()
        losses = AverageMeter()
        top1 = AverageMeter()
        wtop1 = AverageMeter()
        alloutputs = []
        metrics = {}


        # switch to evaluate mode
        model.eval()

        def part(x):
            return itertools.islice(x, int(len(x) * args.val_size))
        for i, x in enumerate(part(loader)):
            if(ignore):
                if i>3:
                    break
            inputs, target, meta = parse(x)
           # Sanno, Grad = FramesList(x)
            output, loss, weights = forward(inputs, target, model, criterion, meta['id'], train=False)
           # weights = weights * Sanno * Grad

            """
            if (i % 1 == 0):

                video_id = x[2].items()[5][1][0]
                firsttime_neg = str(x[2].items()[2][1].numpy()[0] + 1).zfill(6)
                firsttime_pos = str(x[2].items()[1][1].numpy()[0] + 1).zfill(6)
                thirdtime = str(x[2].items()[0][1].numpy()[0] + 1).zfill(6)
                path = '/home/yhy/actor-observer-master/datasets/CharadesEgo_v1_rgb/'
                firstframe_path_pos = path + video_id + 'EGO' + '/' + video_id + 'EGO-' + firsttime_pos+ '.jpg'
                firstframe_path_neg = path + video_id + 'EGO' + '/' + video_id + 'EGO-' + firsttime_neg + '.jpg'
                thirdframe_path = path + video_id + '/' + video_id + '-' + thirdtime + '.jpg'
                firstimg_pos = tran2(tran1(Image.open(firstframe_path_pos)))
                firstimg_neg = tran2(tran1(Image.open(firstframe_path_neg)))
                thirdimg = tran2(tran1(Image.open(thirdframe_path)))

                pic.savefig('/home/yhy/2.Attention-Maps/test/' + video_id + '_' + str(i) + 'vis.jpg')
                im_gray = cv2.imread('/home/yhy/2.Attention-Maps/test/' + video_id + '_' + str(i) + 'vis.jpg', cv2.IMREAD_GRAYSCALE)
                im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
                cv2.imwrite('/home/yhy/2.Attention-Maps/test/' + video_id + '_' + str(i) + 'vis.jpg', im_color)

                ax1 = pic.add_subplot(1, 3, 1)
                ax1.imshow(firstimg_pos)
                ax2 = pic.add_subplot(1, 3, 2)
                ax2.imshow(thirdimg)
                ax3 = pic.add_subplot(1, 3, 3)
                ax3.imshow(firstimg_neg)
                # plt.show()
                pic.savefig('/home/yhy/2.Attention-Maps/test/' + video_id + '_' + str(i) + '.jpg')
                plt.clf()"""

            prec1 = triplet_accuracy(output, target)
            wprec1 = triplet_accuracy(output, target, weights)
            losses.update(loss.item(), inputs[0].size(0))
            top1.update(prec1, inputs[0].size(0))
            wtop1.update(wprec1, inputs[0].size(0))
            alloutputs.extend(zip([(x.item(), y.item()) for x, y in zip(*output)], target, weights))
            timer.tic()

            if i % args.print_freq == 0:
                print('[{name}] Test [{epoch}]: [{0}/{1} ({2})]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'WAcc@1 {wtop1.val:.3f} ({wtop1.avg:.3f})\t'.format(
                          i, int(len(loader) * args.val_size), len(loader), name=args.name,
                          timer=timer, loss=losses, top1=top1, epoch=epoch, wtop1=wtop1))

        metrics.update(triplet_allk(*zip(*alloutputs)))  # acc
        metrics.update({'top1val': top1.avg, 'wtop1val': wtop1.avg})
        print(' * Acc@1 {top1val:.5f} \t WAcc@1 {wtop1val:.5f}'
              '\n   topk1: {topk1:.5f} \t topk2: {topk2:.5f} \t '
              'topk5: {topk5:.5f} \t topk10: {topk10:.5f} \t topk50: {topk50:.5f}'
              .format(**metrics))
        return metrics

    def validate_video(self, loader, model, epoch, args):
        """ Run video-level validation on the Charades test set"""
        timer = Timer()
        outputs, gts, ids = [], [], []
        metrics = {}

        # switch to evaluate mode
        model.eval()

        for i, x in enumerate(loader):
            inputs, target, meta = parse(x)
            target = target.long().cuda(async=True)
            assert target[0, :].eq(target[1, :]).all(), "val_video not synced"
            input_vars = [torch.autograd.Variable(inp.cuda(), volatile=True)
                          for inp in inputs]
            output = model(*input_vars)[-1]  # classification should be last output
            output = torch.nn.Softmax(dim=1)(output)

            # store predictions
            output_video = output.mean(dim=0)
            outputs.append(output_video.data.cpu().numpy())
            gts.append(target[0, :])
            ids.append(meta['id'][0])
            timer.tic()

            if i % args.print_freq == 0:
                print('Test2: [{0}/{1}]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f})'.format(
                          i, len(loader), timer=timer))
        # mAP, _, ap = meanap.map(np.vstack(outputs), np.vstack(gts))
        mAP, _, ap = meanap.charades_map(np.vstack(outputs), np.vstack(gts))
        metrics['mAP'] = mAP
        print(ap)
        print(' * mAP {:.3f}'.format(mAP))
        submission_file(
            ids, outputs, '{}/epoch_{:03d}.txt'.format(args.cache, epoch + 1))
        return metrics

    def validate_egovideo(self, loader, model, epoch, args):
        """ Run video-level validation on the Charades ego test set"""
        timer = Timer()
        outputs, gts, ids = [], [], []
        outputsw = []
        metrics = {}

        # switch to evaluate mode
        model.eval()
        for i, x in enumerate(loader):
            inp, target, meta = parse(x)
            target = target.long().cuda(async=True)
            assert target[0, :].eq(target[1, :]).all(), "val_video not synced"
            input_var = torch.autograd.Variable(inp.cuda(), volatile=True)
            output, w_x, w_z = model(input_var)
            output = torch.nn.Softmax(dim=1)(output)

            sw_x = torch.nn.Softmax(dim=0)(w_x) * w_x.shape[0]
            sw_x = (sw_x - sw_x.mean()) / sw_x.std()
            scale = torch.clamp(1 + (sw_x - 1) * 0.05, 0, 100)
            print('scale min: {}\t max: {}\t std: {}'.format(scale.min().item(), scale.max().item(), scale.std().item()))
            scale = torch.clamp(scale, 0, 100)
            scale *= scale.shape[0] / scale.sum()
            outputw = output * scale.unsqueeze(1)

            # store predictions
            output_video = output.mean(dim=0)
            outputs.append(output_video.data.cpu().numpy())
            outputsw.append(outputw.mean(dim=0).data.cpu().numpy())
            gts.append(target[0, :])
            ids.append(meta['id'][0])
            timer.tic()

            if i % args.print_freq == 0:
                print('Test2: [{0}/{1}]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f})'.format(
                          i, len(loader), timer=timer))
        # mAP, _, ap = meanap.map(np.vstack(outputs), np.vstack(gts))
        mAP, _, ap = meanap.charades_nanmap(np.vstack(outputs), np.vstack(gts))
        mAPw, _, _ = meanap.charades_nanmap(np.vstack(outputsw), np.vstack(gts))
        metrics['mAPego'] = mAP
        metrics['mAPegow'] = mAPw
        print(ap)
        print(' * mAPego {mAPego:.3f} \t mAPegow {mAPegow:.3f}'.format(**metrics))
        submission_file(
            ids, outputs, '{}/egoepoch_{:03d}.txt'.format(args.cache, epoch + 1))
        return metrics
