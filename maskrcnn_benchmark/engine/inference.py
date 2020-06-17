# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import shutil
import time

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize

import pdb

def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)

        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
            
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
            
    return results_dict

# special, passes "targets".
def compute_order_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images, targets=targets)
            output = [o.to(cpu_device) for o in output]
            
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
            
    return results_dict

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, working_directory="/tmp", skip_gather=False):
    all_predictions = scatter_gather(predictions_per_gpu, working_directory, skip_gather=skip_gather)
    if not is_main_process():
        return

    if skip_gather:
        return all_predictions
    
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        predictions=None,
        working_directory="/tmp",
        chunk_predictions=False,
        compute_pre_results=True,
        panoptic_confidence_thresh=0.6,
        panoptic_overlap_thresh=0.5,
        panoptic_stuff_min_area=(64 * 64)):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset

    # for now, short circuit for "order".
    if "order" in iou_types:
        predictions = compute_order_on_dataset(model, data_loader, device)
        return

    given_predictions = not (predictions is None)
    if not given_predictions:
        # make sure these are divisible.
        if chunk_predictions:
            print("chunking predictions")
        
        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
        start_time = time.time()
        predictions = compute_on_dataset(model, data_loader, device)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        logger.info(
            "Total inference time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )

        predictions = _accumulate_predictions_from_multiple_gpus(predictions, working_directory=working_directory, skip_gather=chunk_predictions)
        # if we decided to keep the chunks, we only get filenames back.

    if not is_main_process():
        return

    # OK, should all be done within a single process now.
    if output_folder and not given_predictions:
        if chunk_predictions:
            parent_path = os.path.dirname(predictions[0])
            
            for i, chunk_path in enumerate(predictions):
                chunk_save_path = os.path.join(output_folder, os.path.basename(chunk_path))
                shutil.move(chunk_path, chunk_save_path)

                predictions[i] = chunk_save_path

            # remove the parent.
            os.rmdir(parent_path)

            print("chunks:")
            print(predictions)
        else:
            torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        working_directory=working_directory,
        save_panoptic_results=True,
        save_pre_results=compute_pre_results,
        panoptic_confidence_thresh=panoptic_confidence_thresh,
        panoptic_overlap_thresh=panoptic_overlap_thresh,
        panoptic_stuff_min_area=panoptic_stuff_min_area)

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
