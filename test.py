#!/usr/bin/env python

"""Code for "Actor and Observer: Joint Modeling of First and Third-Person Videos", CVPR 2018
   Can be run directly or throught config scripts under exp/

   Gunnar Sigurdsson, 2018
"""
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

best_top1 = 0

def validate(trainer, loaders, model, criterion, args, ignore, epoch=-1 ):
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
    print ('alignment',scores)"""

    scores.update(trainer.validate(val_loader, model, criterion, epoch, args,ignore))
    return scores


def test(path,ignore):
    global args, best_top1
    args = parse()

    model, criterion, optimizer = create_model(args)

    args.resume = path
    if args.resume:
        best_top1 = checkpoints.load(args, model, optimizer)
        #print(model)
    print("model is done!")

    trainer = train.Trainer()
    loaders = get_dataset(args)
    print("loader is done!")

    scores = validate(trainer, loaders, model,criterion, args,ignore)
    print(scores)
    #checkpoints.score_file(scores, "{}/model_000.txt".format(args.cache))
    print('test data!')

    return scores





