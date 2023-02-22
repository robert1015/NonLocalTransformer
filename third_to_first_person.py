#!/usr/bin/env python
import sys
import os
import subprocess
import traceback
import pdb
from bdb import BdbQuit
"""
subprocess.Popen('find ./exp/.. -iname "*.pyc" -delete'.split())
sys.path.insert(0, '.')
os.nice(19)"""
print(sys.path)
from main import main
from test import test
name = __file__.split('/')[-1].split('.')[0]  # name is filename

args = [
    '--name', name, 
    '--print-freq', '50',
    # '--train-file', '/home/yhy/actor-observer-master/datasets/labels/CharadesEgo_v1_train_new.csv',
    # '--val-file', '/home/yhy/actor-observer-master/datasets/labels/CharadesEgo_v1_test_new.csv',
    '--dataset', 'charadesego',
    # '--data', '/home/yhy/actor-observer-master/datasets/CharadesEgo_v1_rgb/',
    '--arch', 'ActorObserverBaseNoShare',
    '--subarch', 'resnet152',
    '--pretrained-subweights', '/home/zzhehao/JointAttention/exp/resnet152-b121ed2d.pth',
    '--loss', 'ActorObserverLossAll',
    '--subloss', 'DistRatio',
    '--subloss2', 'AttentionDist',
    '--decay', '0.95',
    '--lr', '1e-4',
    '--lr-decay-rate', '15',
    '--batch-size', '16',
    '--train-size', '0.1',
    '--val-size', '0.2',
    '--cache-dir', '/home/zzhehao/JointAttention/result/',
    '--epochs', '50',
    # '--evaluate',
    '--alignment',
    # '--usersalignment',
]
sys.argv.extend(args)
try:
    main()
   # test("/home/yhy/1.MODEL/v1-add_CBAM/third_to_first_person_1224CBAM_part1/model_best.pth.tar",False)
except BdbQuit:
    sys.exit(1)
except Exception:
    traceback.print_exc()
    print ''
    pdb.post_mortem()
    sys.exit(1)
