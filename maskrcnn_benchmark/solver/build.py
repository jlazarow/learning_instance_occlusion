# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import pdb

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []

    order_only = cfg.MODEL.ORDER_ON and cfg.MODEL.ROI_ORDER_HEAD.ONLY_TRAIN
    for key, value in model.named_parameters():
        if order_only and not ("roi_heads.order" in key):
            print("turning off {0} due to order head only".format(key))
            value.requires_grad = False
        
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR

        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        add_params = {
            "params": [value],
            "lr": lr,
            "weight_decay": weight_decay
        }

        # FIX.
        if cfg.SOLVER.RESUME_ITER > 0:
            add_params["initial_lr"] = cfg.SOLVER.BASE_LR

        params += [add_params]

    optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer

def make_lr_scheduler(cfg, optimizer):
    last_epoch = -1
    if cfg.SOLVER.RESUME_ITER > 0:
        last_epoch = cfg.SOLVER.RESUME_ITER
    
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
        last_epoch=last_epoch
    )
