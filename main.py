#!/usr/bin/env python

import torch
import numpy as np
import random
import train
import tasks
import pdb
from models import create_model
from models.ActorObserverFirstPerson import ActorObserverFirstPerson
from datasets.get import get_dataset
from datasets.charadesegoalignment import get as get_alignment
# from datasets.charadesegousersalignment import get as get_usersalignment
import checkpoints
from opts import parse
from utils import tee
from test import test


def seed(manualseed):
    random.seed(manualseed)
    np.random.seed(manualseed)
    torch.manual_seed(manualseed)
    torch.cuda.manual_seed(manualseed)

best_top1 = 0


def validate(trainer, loaders, model, criterion, args, ignore, epoch=-1):
    _, val_loader, valvideo_loader = loaders[:3]
    scores = {}
    """
    if args.valvideoego:
        scores.update(trainer.validate_egovideo(
            loaders[3], ActorObserverFirstPerson(model), epoch, args))
    if args.alignment:
        alignment_loader = get_alignment(args)
        scores['alignment'] = tasks.alignment(alignment_loader, model, epoch, args)
    if args.usersalignment:
        alignment_loader = get_usersalignment(args)
        scores['usersalignment'] = tasks.alignment(alignment_loader, model, epoch, args)
    if args.valvideo:
        scores.update(trainer.validate_video(valvideo_loader, model, epoch, args))
"""
    scores.update(trainer.validate(val_loader, model, criterion, epoch, args,ignore))
    return scores

def score_file_main(scores, filename):
    with open(filename, 'w') as f:
        for key, val in sorted(scores.items()):
            f.write('{} {}\n'.format(key, val))

def main():
    global args, best_top1
    args = parse()
    if not args.no_logger:
        tee.Tee(args.cache + '/log.txt')
    print(vars(args))
    seed(args.manual_seed)

    model, criterion, optimizer = create_model(args)


    if args.resume:
        best_top1 = checkpoints.load(args, model, optimizer)
   # print(model)
    trainer = train.Trainer()
    loaders = get_dataset(args)
    train_loader = loaders[0]

    if args.evaluate:
        scores = validate(trainer, loaders, model, criterion, args)
        checkpoints.score_file(scores, "{}/model_000.txt".format(args.cache))
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            trainer.train_sampler.set_epoch(epoch)
        scores = {}
        ignore = True
        scores.update(trainer.train(train_loader, model, criterion, optimizer, epoch, args))
        scores.update(validate(trainer, loaders, model, criterion, args, ignore, epoch))
        #is_best = scores[args.metric] > best_top1
        #best_top1 = max(scores[args.metric], best_top1)
        is_best = True
        ignore = False
        model_path = checkpoints.save(epoch, args, model, optimizer, is_best, scores, args.metric)
        scores = test(model_path,ignore)
        score_file_main(scores, "{}/model_{:03d}.txt".format(args.cache, epoch + 1))
        #is_best = scores[args.metric] > best_top1
        best_top1 = max(scores[args.metric], best_top1)
        print('best:',best_top1)
    if not args.nopdb:
        pdb.set_trace()


if __name__ == '__main__':
    main()
