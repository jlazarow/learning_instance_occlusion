# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
import maskrcnn_benchmark
import pycocotools
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.segmenter import build_segmentation_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.experiment import SaveCodeChanges, ExperimentRun
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from expdf.file import ExperimentRunFile
from expdf.hierarchical import *
from expdf.statistic import *

import pdb

kind_builder = {
    'Detection': build_detection_model,
    'Segmentation': build_segmentation_model,
    "Panoptic": build_detection_model,
}

class PanopticExperimentRunFile(ExperimentRunFile):
    def read(self):
        # split each epoch into its own file.
        self.stats = SetOfStatistics(self, self.root, "stats")
        if self.writable and self.multiple_reader:
            self.enter_multiple_reader()

    def refresh(self):
        super(PanopticExperimentRunFile, self).refresh()
        self.stats.refresh()

    def serialize(self):
        return {
            "number_iterations": self.stats.size
        }

    def close(self):
        super(PanopticExperimentRunFile, self).close()

class PanopticExperimentRun(ExperimentRun):
    def _create_experiment_file(self):
        return PanopticExperimentRunFile(
            os.path.join(self.data_path, "index.h5"),
            mode="w",
            multiple_reader=True)

def train(cfg, local_rank, distributed, experiment=None):
    if not cfg.TASK.KIND in kind_builder:
        raise Exception('unknown task: {0}'.format(cfg.TASK.KIND))
        
    model_builder = kind_builder[cfg.TASK.KIND]
    model = model_builder(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR    
        
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    # todo, make this relative?
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        experiment=experiment
    )

    return model

def test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
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
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # replace the output directory based on the "task" kind.
    cfg.OUTPUT_DIR = os.path.join(
        os.path.dirname(cfg.OUTPUT_DIR),
        cfg.TASK.KIND.lower(),
        cfg.NAME,
        os.path.basename(cfg.OUTPUT_DIR))

    # fix LRs based on number of gpus.
    batch_factor = float(cfg.SOLVER.IMS_PER_BATCH) / float(cfg.SOLVER.BASE_IMS_PER_BATCH)
    print("batch factor: {0}. adjusting.".format(batch_factor))

    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * batch_factor
    print("new LR: {0}".format(cfg.SOLVER.BASE_LR))
    
    current_decay_steps = cfg.SOLVER.STEPS
    cfg.SOLVER.STEPS = tuple([int(s / batch_factor) for s in current_decay_steps])
    print("new steps: {0}".format(cfg.SOLVER.STEPS))
    
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER / batch_factor)
    print("new max iter: {0}".format(cfg.SOLVER.MAX_ITER))

    cfg.SOLVER.RESUME_ITER = int(cfg.SOLVER.RESUME_ITER / batch_factor)
    print("new resume iter: {0}".format(cfg.SOLVER.RESUME_ITER))
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    # note, for ease, the experiment structure is already
    # set up.
    experiment = None
    if is_main_process():
        # save the config to the output directory.
        save_config_path = os.path.join(output_dir, "config.yaml")

        with open(save_config_path, "w") as cf:
            cf.write(cfg.dump())
            print("wrote (merged) config to {0}".format(save_config_path))

        save_code = SaveCodeChanges([os.path.dirname(maskrcnn_benchmark.__path__[0])])
        experiment = PanopticExperimentRun(
            output_dir, cfg.NAME, cfg.DESCRIPTION, startup_tasks=[save_code], epitaph="")
        experiment.__enter__()
        
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))

    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
        
    logger.info("Running with config:\n{}".format(cfg))

    try:
        print("about to train with rank: {0}".format(args.local_rank))
        model = train(cfg, args.local_rank, args.distributed, experiment=experiment)

        if not args.skip_test:
            test(cfg, model, args.distributed)
    finally:
        if is_main_process():
            if experiment:
                experiment.end()

if __name__ == "__main__":
    main()
