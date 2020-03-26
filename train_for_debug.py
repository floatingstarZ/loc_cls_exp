from __future__ import division
import argparse
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2019)
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args(args=''):
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    if args == '':
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    #########################################################

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # args = ['./configs/fovea_r50_fpn_4gpu_1x.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/fovea_r50_baseline'
    #         ]
    # args = ['./configs/NMS_Net_r18.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/NMS_Net_ONLY_BACKBONE'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    # args = ['./configs/retinanet_r18_toy.py',
    #         '--gpus', '1',
    #         '--work_dir', './results/NMS_Net_test'
    #         ]

    ########################retina###########################
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # args = ['./configs/NMS_Net_r18.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/NMS_Net_test'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # args = ['./configs/NMS_Net_r18.py',
    #         '--gpus', '1',
    #         '--work_dir', './results/NMS_Net_test'
    #         ]
    ################## FOVEA ########################
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args = ['./configs/NMS_FoveaNet_r18.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/NMS_FoveaNet_r18'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # args = ['./configs/NMS_FoveaNet_r18.py',
    #         '--gpus', '1',
    #         '--work_dir', './results/NMS_FoveaNet_r18'
    #         ]
    ################### Fovea + NMS ##################
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args = ['./configs/NMS_Module_Fovea_r18.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/NMS_Module_Fovea_r18'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # args = ['./configs/NMS_Module_Fovea_r18.py',
    #         '--gpus', '1',
    #         '--work_dir', './results/NMS_Module_Fovea_r18'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # args = ['./configs/NMS_Module_Fovea_r50.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/NMS_Module_Fovea_r50'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # args = ['./configs/NMS_Module_Fovea_r50.py',
    #         '--gpus', '1',
    #         '--work_dir', './results/NMS_Module_Fovea_r50'
    #         ]
    ####################################################
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args = ['./configs/NMS_Module_Fovea_r50_deform.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/NMS_Module_Fovea_r50_deform'
    #         ]
    ##################### ID ###################################
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args = ['./configs/ID_Fovea_r50.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/ID_Fovea_r50'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    # args = ['./configs/ID_Fovea_r50.py',
    #         '--gpus', '1',
    #         '--work_dir', './results/ID_Fovea_r50'
    #         ]
    #################### refine(non local) ###################################
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args = ['./configs/Fovea_NMS_Head_Netr18.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_NMS_Head_Netr18'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    # args = ['./configs/Fovea_NMS_Head_Netr18.py',
    #         '--gpus', '1',
    #         '--work_dir', './results/Fovea_NMS_Head_Netr18'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # args = ['./configs/Fovea_NMS_Head_Net_r50.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_NMS_Head_Net_r50'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    # args = ['./configs/Fovea_NMS_Head_Net_r50.py',
    #         '--gpus', '1',
    #         '--work_dir', './results/Fovea_NMS_Head_Net_r50'
    #         ]
    #########################################
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args = ['./configs/Fovea_iou_r18.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_iou_r18'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # args = ['./configs/Fovea_iou_r18.py',
    #         '--gpus', '1',
    #         '--work_dir', './results/Fovea_iou_r18'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
    # args = ['./configs/Fovea_iou_r18.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_iou_r18'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # args = ['./configs/Fovea_iou_r50.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_iou_r50_ver1'
    #         ]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    args = ['./configs/Fovea_predict_iou_r18.py',
            '--gpus', '4',
            '--work_dir', './results/Fovea_predict_iou_r18'
            ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args = ['./configs/Fovea_predict_iou_r50.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_predict_iou_r50'
    #         ]
    ####################################################
    # os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    # args = ['./configs/Fovea_var_r18.py',
    #         '--gpus', '1',
    #         '--work_dir', './results/Fovea_var_r18'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # args = ['./configs/Fovea_var_r50.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_var_r50_exp'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args = ['./configs/Fovea_predict_iou_r50.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_predict_iou_r50'
    #         ]
    #######################
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,8,9"
    # args = ['./configs/Fovea_bay_nms_r18.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_bay_nms_r18'
    #         ]

    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args = ['./configs/Fovea_var_r18_coco.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_var_r18_coco'
    #         ]
#############################################################
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args = ['./configs/Fovea_multi_r18.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_multi_r18'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # args = ['./configs/Fovea_multi_r50.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_multi_r50_r6_c3'
    #         ]
    #########################################################
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args = ['./configs/Fovea_boost_r18.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_boost_r18'
    #         ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # args = ['./configs/Fovea_multi_r50.py',
    #         '--gpus', '4',
    #         '--work_dir', './results/Fovea_multi_r50_r6_c3'
    #         ]
    args = parse_args(args)
    print(args)

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # print([name for name, value in model.named_parameters()])
    # print(model)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
