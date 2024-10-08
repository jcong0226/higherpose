# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import warnings
import json
import yaml
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

import _init_paths
import models

from config import cfg
from config import update_config
from core.loss import MultiLossFactory
from core.trainer import do_train
from dataset import make_dataloader
from fp16_utils.fp16util import network_to_half
from fp16_utils.fp16_optimizer import FP16_Optimizer
from utils.utils import create_logger
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import setup_logger
from utils.utils import get_model_summary
# from scheduler import WarmupMultiStepLR
from arch_manager import ArchManager

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    # change the resolution according to config
    fixed_arch = None
    # if args.superconfig is not None:
    #     with open(args.superconfig, 'r') as f:
    #        fixed_arch = json.load(f)
    
    cfg.defrost()
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    if cfg.MODEL.NAME == 'pose_mobilenet' or cfg.MODEL.NAME == 'pose_simplenet':
        arch_manager = ArchManager(cfg)
        cfg_arch = arch_manager.fixed_sample()
        if fixed_arch is not None:
            cfg_arch = fixed_arch
    else:
        cfg_arch = None

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function
    main_worker(
        ','.join([str(i) for i in cfg.GPUS]),
        ngpus_per_node,
        logger,
        args,
        final_output_dir,
        tb_log_dir,
        cfg_arch=cfg_arch
    )



def main_worker(
        gpu, ngpus_per_node, logger, args, final_output_dir, tb_log_dir, cfg_arch=None
):
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.FP16.ENABLED:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if cfg.FP16.STATIC_LOSS_SCALE != 1.0:
        if not cfg.FP16.ENABLED:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    update_config(cfg, args)

    # setup logger
    # logger, _ = setup_logger(final_output_dir, args.rank, 'train')
    have_teacher = False

    if cfg.MODEL.NAME == 'pose_mobilenet' or cfg.MODEL.NAME == 'pose_simplenet':
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=True, cfg_arch = cfg_arch
        )
        if args.teacher == True:
            with open('./mobile_configs/search-S.json', 'r') as f:
                teacher_arch = json.load(f)
            teacher = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
                cfg, is_train=True, cfg_arch = teacher_arch
            )
            state_dict = torch.load('./pretrain/teacher/coco-S.pth.tar', map_location='cpu')
            teacher.load_state_dict(state_dict, strict=False)
            have_teacher = True
        else:
            teacher = None
    else:
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=True
        )
        teacher = None

    #set super config
    # if args.superconfig is not None and (cfg.MODEL.NAME == 'pose_supermobilenet' or cfg.MODEL.NAME == 'pose_superresnet'):  # superconfig None
    #     model.arch_manager.is_search = True
    #     with open(args.superconfig, 'r') as f:
    #         model.arch_manager.search_arch = json.load(f)
    
    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, './lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir
    )


    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # if not cfg.MULTIPROCESSING_DISTRIBUTED or (
    #         cfg.MULTIPROCESSING_DISTRIBUTED
    #         and args.rank == 0
    # ):
    #     dump_input = torch.rand(
    #         (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    #     )
    #     logger.info(get_model_summary(cfg.DATASET.INPUT_SIZE, model, dump_input))


    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    model.cuda(0)

    # define loss function (criterion) and optimizer
    loss_factory = MultiLossFactory(cfg).cuda()

    # Data loading code
    train_loader = make_dataloader(
        cfg, is_train=True, distributed=False
    )
    logger.info(train_loader.dataset)

    best_perf = -1
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)

    if cfg.FP16.ENABLED:  # false
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=cfg.FP16.STATIC_LOSS_SCALE,
            dynamic_loss_scale=cfg.FP16.DYNAMIC_LOSS_SCALE
        )

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    if cfg.FP16.ENABLED:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer.optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
        # )

    begin_time = datetime.datetime.now()
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train one epoch
        heatmap_loss, push_loss, pull_loss = do_train(cfg, model, lr_scheduler, train_loader, loss_factory, optimizer, epoch,
                 final_output_dir, tb_log_dir, writer_dict, fp16=cfg.FP16.ENABLED, teacher=teacher)
        writer = writer_dict['writer']
        writer.add_scalar('train_epoch_heatmap_loss', heatmap_loss, epoch)
        writer.add_scalar('train_epoch_push_loss', push_loss, epoch)
        writer.add_scalar('train_epoch_pull_loss', pull_loss, epoch)
        # In PyTorch 1.1.0 and later, you should call `lr_scheduler.step()` after `optimizer.step()`.
        optimizer.step()
        perf_indicator = epoch
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}\n'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    end_time = datetime.datetime.now()
    logger.info("train begin in:{}".format(begin_time))
    logger.info("train finish in:{}".format(end_time))
    logger.info("total time is:{}".format((end_time - begin_time)))

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state{}.pth.tar'.format(gpu)
    )

    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()

# python dist_train.py --cfg experiments/coco/mobilenet/supermobile.yaml