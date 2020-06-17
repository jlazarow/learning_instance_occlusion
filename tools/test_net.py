# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import torch

import maskrcnn_benchmark
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.experiment import RunningLockFile, SaveCodeChanges
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

import pdb

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # replace the output directory based on the "task" kind.
    cfg.OUTPUT_DIR = os.path.join(
        os.path.dirname(cfg.OUTPUT_DIR),
        cfg.TASK.KIND.lower(),
        cfg.NAME,
        os.path.basename(cfg.OUTPUT_DIR))
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    running_lock = None
    if is_main_process():
        # save the config to the output directory.
        save_config_path = os.path.join(output_dir, "config.yaml")

        with open(save_config_path, "w") as cf:
            cf.write(cfg.dump())
            print("wrote (merged) config to {0}".format(save_config_path))

        running_lock = RunningLockFile(output_dir).start()
        save_code = SaveCodeChanges([os.path.dirname(maskrcnn_benchmark.__path__[0])])
        save_code(output_dir)

        print("saved code changes (against HEAD) to {0}".format(output_dir))

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
        
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    chunk_predictions = cfg.TEST.CHUNK_PREDICTIONS

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)

    if cfg.TASK.KIND.lower() == "panoptic":
        # at some point we should run all bbox/segm, etc.
        if cfg.MODEL.PANOPTIC.COMPUTE_CC_RESULTS and not chunk_predictions:
            iou_types = ("ccpan", "pan")
        else:
            iou_types = ("pan",)


    if cfg.TEST.ORDER_ONLY:
        iou_types = ("order",)
            
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    # allow the user to override with existing .pth files.
    # unsure how this will work in the distributed setting.
    if not distributed:
        existing_predictions = cfg.TEST.PREDICTION_PATHS
    else:
        print("distributed... ignoring existing predictions if given")
        existing_predictions = []

    chunk_predictions = cfg.TEST.CHUNK_PREDICTIONS
    if not distributed:
        # only makes sense for multiple GPUS when doing actual prediction.
        chunk_predictions = chunk_predictions and existing_predictions
        
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    dataset_index = 0

    try:
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            predictions = None
            if existing_predictions is not None and dataset_index < len(existing_predictions):
                # todo, check for "chunk" predictions" and delay the load.
                predictions_path = existing_predictions[dataset_index]

                if not chunk_predictions:
                    predictions = torch.load(predictions_path)
                else:
                    # this should be a list of "chunks".
                    predictions = predictions_path
                    print("using chunks: {0}".format(predictions))

            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                predictions=predictions,
                working_directory=cfg.TEMPORARY_DIR,
                chunk_predictions=chunk_predictions,
                compute_pre_results=cfg.MODEL.PANOPTIC.COMPUTE_PRE_RESULTS,
                panoptic_confidence_thresh=cfg.MODEL.FUSION.CONFIDENCE_THRESHOLD,
                panoptic_overlap_thresh=cfg.MODEL.FUSION.OVERLAP_THRESHOLD,
                panoptic_stuff_min_area=cfg.MODEL.FUSION.STUFF_MINIMUM_AREA)
            
            synchronize()
            dataset_index += 1
    finally:
        if running_lock:
            running_lock.end()

if __name__ == "__main__":
    main()
