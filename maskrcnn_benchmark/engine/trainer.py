# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from expdf.render import *
from expdf.statistic import *

# build the osmosis "web" API.
from osmosis.membranes.expdf.render import ScalarPlotRenderable

import pdb

def numpyize(t):
    # probably check device first.
    return t.detach().cpu().numpy()

PanopticTrainingStatistic = IndexedStatistic(
    "PanopticTrainingStatistic",
    [
        ScalarFieldDefinition("semantic"),
        ScalarFieldDefinition("objectness"),
        ScalarFieldDefinition("rpn_box_reg"),
        ScalarFieldDefinition("mask"),
        ScalarFieldDefinition("classifier"),
        ScalarFieldDefinition("box_reg")
    ],
    renderables=[
        ScalarPlotRenderable(
            "semantic",
            "semantic segmentation xentropy",
            x_value_name="step",
            y_value_name="semantic"),
        ScalarPlotRenderable(
            "objectness",
            "RPN objectness",
            x_value_name="step",
            y_value_name="objectness"),
        ScalarPlotRenderable(
            "rpn_box_reg",
            "RPN box regression",
            x_value_name="step",
            y_value_name="rpn_box_reg"),
        ScalarPlotRenderable(
            "mask",
            "mask xentropy",
            x_value_name="step",
            y_value_name="mask"),
        ScalarPlotRenderable(
            "classifier",
            "box head classification",
            x_value_name="step",
            y_value_name="classifier"),
        ScalarPlotRenderable(
            "box_reg",
            "box head box regression",
            x_value_name="step",
            y_value_name="box_reg")
    ])

def reduce_loss_dict(loss_dict, weights=None):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k] if weights is None else loss_dict[k] * weights[k])
            
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    experiment=None
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start ({0}) training".format(cfg.TASK.KIND.lower()))
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    # todo, enum>?
    is_detection = False
    is_semantic = False
    is_panoptic = False
    
    if cfg.TASK.KIND.lower() == "detection":
        is_detection = True
    elif cfg.TASK.KIND.lower() == "semantic":
        is_semantic = True
    elif cfg.TASK.KIND.lower() == "panoptic":
        is_panoptic = True

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        
        loss_dict = model(images, targets)

        weights = None
        if is_detection:
            losses = sum(loss for loss in loss_dict.values())
        elif is_semantic:
            raise Exception('not supported yet')
        elif is_panoptic:
            # NOTE: only supports Mask RCNN for now.
            # Facebook has an aversion to constants so I'll just repeat these here.
            weights = {
                "loss_semantic": cfg.MODEL.PANOPTIC.SEMANTIC_WEIGHT,
                "loss_objectness": cfg.MODEL.PANOPTIC.INSTANCE_WEIGHT,
                "loss_rpn_box_reg": cfg.MODEL.PANOPTIC.INSTANCE_WEIGHT,
                "loss_mask": cfg.MODEL.PANOPTIC.INSTANCE_WEIGHT,
                "loss_classifier": cfg.MODEL.PANOPTIC.INSTANCE_WEIGHT,
                "loss_box_reg": cfg.MODEL.PANOPTIC.INSTANCE_WEIGHT,
            }

            if cfg.MODEL.ORDER_ON:
                weights["loss_order"] = cfg.MODEL.ROI_ORDER_HEAD.WEIGHT

            if cfg.MODEL.FUSION_ON:
                weights["loss_rank"] = cfg.MODEL.PANOPTIC.INSTANCE_WEIGHT

            losses = sum(weights[name] * loss for name, loss in loss_dict.items())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict, weights=weights)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

            # add the results if in the main process.
            if experiment and is_panoptic:
                experiment.file.stats.add(
                    step=iteration,
                    semantic=numpyize(loss_dict_reduced["loss_semantic"]),
                    objectness=numpyize(loss_dict_reduced["loss_objectness"]),
                    rpn_box_reg=numpyize(loss_dict_reduced["loss_rpn_box_reg"]),
                    mask=numpyize(loss_dict_reduced["loss_mask"]),
                    classifier=numpyize(loss_dict_reduced["loss_classifier"]),
                    box_reg=numpyize(loss_dict_reduced["loss_box_reg"]),
                    writable_type=PanopticTrainingStatistic)
                                
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        torch.cuda.empty_cache()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
