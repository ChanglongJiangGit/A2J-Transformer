# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Tester
from utils.vis import vis_keypoints
import torch.backends.cudnn as cudnn
from utils.transforms import flip

import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='6,7', dest='gpu_ids')
    parser.add_argument('--test_set', type=str, default='test', dest='test_set')
    parser.add_argument('--test_epoch', type=str, default='0', dest='test_epoch')
    
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def test():
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True
    
    if cfg.dataset == 'InterHand2.6M':
        assert args.test_set, 'Test set is required. Select one of test/val'
    else:
        args.test_set = 'test'

    tester = Tester(args.test_epoch)
    tester._make_batch_generator(args.test_set)
    tester._make_model()
    
    preds = {'joint_coord': [], 'inv_trans': [], 'joint_valid': [] }

    timer = []
    

    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator,ncols=150)):
            
            # forward
            start = time.time()
            out = tester.model(inputs, targets, meta_info, 'test')
            end = time.time()

            joint_coord_out = out['joint_coord'].cpu().numpy()
            inv_trans = out['inv_trans'].cpu().numpy()
            joint_vaild = out['joint_valid'].cpu().numpy()

            preds['joint_coord'].append(joint_coord_out)
            preds['inv_trans'].append(inv_trans)
            preds['joint_valid'].append(joint_vaild)

            timer.append(end-start)
    
    
    # evaluate
    preds = {k: np.concatenate(v) for k,v in preds.items()}
    
    mpjpe_dict, hand_accuracy, mrrpe = tester._evaluate(preds)
    print(mpjpe_dict)
    print('time per batch is',np.mean(timer))

if __name__ == "__main__":
    test()
