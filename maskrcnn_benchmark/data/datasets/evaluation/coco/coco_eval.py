import copy
import json
import logging
import numpy as np
import tempfile
import os
import math
import shutil
import time
import torch
import torch.multiprocessing as multiprocessing
from collections import OrderedDict, defaultdict
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.coco import COCODataset, COCOTestDataset
from maskrcnn_benchmark.data.datasets.panoptic_coco import PanopticCOCODataset
from maskrcnn_benchmark.metrics.panoptic import PQStat, PQStatCat, OFFSET, VOID
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

from PIL import Image

import pdb

# NOTE: plenty of this code is inspired/steals from Alexander Kirillov's
# Panoptic COCO library.

def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    working_directory="/tmp",
    save_panoptic_results=False,
    save_pre_results=False,
    panoptic_confidence_thresh=0.6,
    panoptic_overlap_thresh=0.5,
    panoptic_stuff_min_area=(64 * 64)):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return
    logger.info("Preparing results for COCO format")
    coco_results = {}

    if ("bbox" in iou_types) or ("segm" in iou_types):
        instance_predictions = predictions
        instance_dataset = dataset

        if isinstance(dataset, PanopticCOCODataset):
            instance_predictions = [p.region for p in predictions]
            instance_dataset = COCODataset(dataset.instances_annotation_path, dataset.root, remove_images_without_annotations=False)
        
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        coco_results["bbox"] = prepare_for_coco_detection(instance_predictions, instance_dataset)

    if "segm" in iou_types:
        logger.info("Preparing segm results")
        coco_results["segm"] = prepare_for_coco_segmentation(instance_predictions, instance_dataset)

    if "ccpan" in iou_types:
        logger.info("Preparing panoptic connected components results")

        save_cc_panoptic_results_path = None
        if save_panoptic_results and output_folder:
            save_cc_panoptic_results_path = os.path.join(output_folder, "cc")

        coco_results["ccpan"] = prepare_connected_components_panoptic(
            predictions,
            dataset,
            working_directory=working_directory,            
            save_panoptic_results_path=save_cc_panoptic_results_path)

        if save_panoptic_results:
            print("wrote connected components panoptic results to {0}".format(save_cc_panoptic_results_path))        
        
    if "pan" in iou_types:
        logger.info("Preparing panoptic results")

        save_panoptic_results_path = None
        save_pre_results_path = None
        if save_panoptic_results and output_folder:
            save_panoptic_results_path = os.path.join(output_folder, "results")

        if save_pre_results and output_folder:            
            save_pre_results_path = os.path.join(output_folder, "pre")
            
        coco_results["pan"] = prepare_for_coco_panoptic(
            predictions,
            dataset,
            working_directory=working_directory,
            save_panoptic_results_path=save_panoptic_results_path,
            save_pre_results_path=save_pre_results_path,
            confidence_thresh=panoptic_confidence_thresh,
            overlap_thresh=panoptic_overlap_thresh,
            stuff_min_area=panoptic_stuff_min_area)
        if save_panoptic_results:
            print("wrote combined panoptic results to {0}".format(save_panoptic_results_path))

    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        # separate panoptic path for now.
        if iou_type == "pan":
            annotations, masks = coco_results[iou_type]

            # map predictions to by ID.
            mapped_predictions = { dataset.id_to_img_map[i]: p for i, p in enumerate(predictions) }
            res = evaluate_panoptic(
                mapped_predictions,
                dataset,
                annotations,
                masks,
                working_directory=working_directory,
                save_panoptic_results_path=save_panoptic_results_path,
                save_pre_results_path=save_pre_results_path)
        elif iou_type == "ccpan":
            annotations, masks = coco_results[iou_type]

            # map predictions to by ID.
            mapped_predictions = { dataset.id_to_img_map[i]: p for i, p in enumerate(predictions) }
            res = evaluate_panoptic(
                mapped_predictions,
                dataset,
                annotations,
                masks,
                working_directory=working_directory,
                save_panoptic_results_path=save_cc_panoptic_results_path,
                save_pre_results_path=None,
                kind="ccpan")
        else:
            with tempfile.NamedTemporaryFile() as f:
                file_path = f.name
                if output_folder:
                    file_path = os.path.join(output_folder, iou_type + ".json")
                res = evaluate_predictions_on_coco(
                    instance_dataset.coco, coco_results[iou_type], file_path, iou_type)

        results.update(res)
        
    logger.info(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))
    return results, coco_results

def prepare_for_coco_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

def prepare_for_coco_segmentation(predictions, dataset):
    import pycocotools.mask as mask_util    

    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # rles = prediction.get_field('mask')

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results

class IdGenerator(object):
    def __init__(self, categories):
        self.taken_colors = set([0, 0, 0])
        self.categories = categories
        for category in self.categories.values():
            if category['isthing'] == 0:
                self.taken_colors.add(tuple(category['color']))

    def get_color(self, cat_id):
        def random_color(base, max_dist=30):
            new_color = base + np.random.randint(low=-max_dist,
                                                 high=max_dist+1,
                                                 size=3)
            return tuple(np.maximum(0, np.minimum(255, new_color)))

        category = self.categories[cat_id]
        if category['isthing'] == 0:
            return category['color']

        base_color_array = category['color']
        base_color = tuple(base_color_array)
        if base_color not in self.taken_colors:
            self.taken_colors.add(base_color)
            return base_color
        else:
            while True:
                color = random_color(base_color_array)
                if color not in self.taken_colors:
                     self.taken_colors.add(color)
                     return color

    def get_id(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color)

    def get_id_and_color(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color), color

def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for i in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color    
    
def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.uint32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return color[0] + 256 * color[1] + 256 * 256 * color[2]

class PanopticFromSemanticSegmentation(object):
    def __init__(self, categories, category_thing_label_map, category_stuff_label_map, stuff_start_id, instance_min_area=(64 **2), stuff_min_area=(64 ** 2)):
        self.categories = categories
        self.category_thing_label_map = category_thing_label_map
        self.category_stuff_label_map = category_stuff_label_map
        self.stuff_start_id = stuff_start_id
        self.id_generator = IdGenerator(self.categories)
        self.instance_min_area = stuff_min_area
        self.stuff_min_area = stuff_min_area

    def __call__(self, prediction, image_height, image_width):
        import scipy.ndimage.measurements as mlabel

        # resize to gt.
        prediction = prediction.resize((image_width, image_height))        
        semantic_mask = prediction.segmentation.mask.numpy()
        contiguous_ids, contiguous_counts = np.unique(
            semantic_mask, return_counts=True)

        if self.stuff_start_id == 1:
            # no go.
            raise ValueError("can only produce instance segmentation from dense ontology")

        # the resulting _panoptic_ mask.
        panoptic_mask = np.zeros((image_height, image_width), dtype=np.uint32)
        segments_info = []
        
        thing_contiguous_ids = []
        for i, contiguous_id in enumerate(contiguous_ids):
            # not collapsed.
            count = contiguous_counts[i]
            is_thing = contiguous_id < self.stuff_start_id

            # unsure if this can be correct.
            if is_thing:
                thing_contiguous_ids.append(contiguous_id)
            else:
                category_id = self.category_stuff_label_map[contiguous_id]

                # "stuff", see if it's big enough.
                if count >= self.stuff_min_area:
                    # generate a color/ID.
                    segment_id = rgb2id(self.id_generator.get_color(category_id))
                    panoptic_mask[semantic_mask == contiguous_id] = segment_id

                    segment_info = {
                        "id": int(segment_id),
                        "category_id": int(category_id),
                    }
                    
                    segments_info.append(segment_info)
                else:
                    print("dropping stuff of kind {0} for being too small".format(
                        self.id_generator.categories[category_id]["name"]))

        # decide to place "things".
        thing_contiguous_ids = np.array(thing_contiguous_ids)
        for thing_contiguous_id in thing_contiguous_ids:
            category_id = self.category_thing_label_map[thing_contiguous_id]
            mask_of_instances, number_instances = mlabel.label(semantic_mask == thing_contiguous_id)

            # go through each instance and "kill" them if they're too small.
            instance_ids, instance_counts = np.unique(
                mask_of_instances, return_counts=True)

            for instance_index in range(number_instances):
                instance_id = instance_ids[instance_index + 1]
                instance_count = instance_counts[instance_index + 1]

                if instance_count >= self.instance_min_area:
                    segment_id = rgb2id(self.id_generator.get_color(category_id))
                    
                    panoptic_mask[mask_of_instances == instance_id] = segment_id
                    segment_info = {
                        "id": int(segment_id),
                        "category_id": int(category_id),
                    }
                    
                    segments_info.append(segment_info)
                    
                else:
                    print("dropping thing of kind {0} for being too small: {1}".format(
                        self.id_generator.categories[category_id]["name"], instance_count))

        return panoptic_mask, segments_info

class PanopticCombinationDebug(object):
    def consider_segment(self, box, mask, category_id, current):
        pass
    
    def rejected_segment(self, reason):
        pass

    def accepted_segment(self, intersect, reclaim_segments=None):
        pass
                    
class PanopticNaiveCombinationMethod(object):
    def __init__(self, categories, category_thing_label_map, category_stuff_label_map, stuff_start_id, confidence_thresh=0.5, overlap_thresh=0.5, stuff_min_area=(64 ** 2),
                 compute_iou_agreements=False):
        self.categories = categories
        self.category_thing_label_map = category_thing_label_map
        self.category_stuff_label_map = category_stuff_label_map
        self.stuff_start_id = stuff_start_id
        self.id_generator = IdGenerator(self.categories)
        self.confidence_thresh = confidence_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_min_area = stuff_min_area
        self.masker = Masker(threshold=0.5, padding=1)
        self.compute_iou_agreements = compute_iou_agreements
        
    def __call__(self, prediction, image_height, image_width, combination_debug=None):
        import pycocotools.mask as mask_util

        # resize to gt.
        prediction = prediction.resize((image_width, image_height))        
        
        # handle instance segmentations first.
        instance_prediction = prediction.region
        # instance_prediction = instance_prediction.sorted_by_confidence()
        
        boxes = instance_prediction.bbox.tolist()
        scores = instance_prediction.get_field("scores")

        ranks = None
        if instance_prediction.has_field("rank"):
            ranks = instance_prediction.get_field("rank")

        labels = instance_prediction.get_field("labels").tolist()
        masks = instance_prediction.get_field("mask")

        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = self.masker(masks.expand(1, -1, -1, -1, -1), instance_prediction)
    
            # this should be the mask corresponding to the most confident detection class?
            masks = masks[0]

        # compute an "agreement" between the masks and the semantic segmentation only if ranks
        # were not provided.
        if ranks is None:
            agreements = torch.ones((len(instance_prediction),))
            if self.compute_iou_agreements:
                for mask_index in range(masks.shape[0]):
                    mask = masks[mask_index, 0]
                    bbox = boxes[mask_index]
                    label = labels[mask_index]
    
                    # for now, round. assume these stay on the image?
                    x_1, y_1, x_2, y_2 = int(np.ceil(bbox[0])), int(np.ceil(bbox[1])), int(np.floor(bbox[2])), int(np.floor(bbox[3]))
                    area = (x_2 - x_1) * (y_2 - y_1)
                    #print(area)
    
                    # pull these out from the mask.
                    cropped_mask = (mask[y_1:(y_2 + 1), x_1:(x_2 + 1)]).numpy() # 
                    cropped_semantic = prediction.segmentation.mask[y_1:(y_2 + 1), x_1:(x_2 + 1)].numpy().astype(np.uint8)
    
                    semantic_with_label = cropped_semantic == label
                    mask_with_label = cropped_mask > 0
                    intersection_with_label = np.logical_and(semantic_with_label, mask_with_label)
                    union_with_label = np.logical_or(semantic_with_label, mask_with_label)
            
                    intersection = np.count_nonzero(intersection_with_label)
                    union = np.count_nonzero(union_with_label)
                    if union == 0:
                        agreements[mask_index] = 0.0
                        print("found a zero union, skipping")
                        continue
            
                    iou = float(intersection) / float(union)
                    agreements[mask_index] = iou
        else:
            agreements = ranks

        # todo, let this be optional to use the scores * ranks.
        instance_prediction = instance_prediction.sorted_by(-agreements  * scores)
        
        boxes = instance_prediction.bbox.tolist()
        scores = instance_prediction.get_field("scores").tolist()
        labels = instance_prediction.get_field("labels").tolist()
        masks = instance_prediction.get_field("mask")

        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = self.masker(masks.expand(1, -1, -1, -1, -1), instance_prediction)
            # this should be the "most confident" mask per detection (check?)
            masks = masks[0]

        overlaps = None
        overlaps_by_index = defaultdict(list)
        if instance_prediction.has_field("overlap"):
            overlaps = {}

            overlap_pairs = instance_prediction.get_field("overlap_pairs")
            overlap_predictions = instance_prediction.get_field("overlap")
            
            for pair_index in range(overlap_pairs.shape[0]):
                overlap_pair = overlap_pairs[pair_index]
                overlap_prediction = int(overlap_predictions[pair_index])
                
                from_index, to_index = int(overlap_pair[0]), int(overlap_pair[1])

                # enforce the anti symmetry.
                overlaps[(from_index, to_index)] = overlap_prediction
                #overlaps[(to_index, from_index)] = 1 if (overlap_prediction == 0) else 0

                overlaps_by_index[from_index].append((from_index, to_index))
                #overlaps_by_index[to_index].append((from_index, to_index))

        panoptic_mask = np.zeros((image_height, image_width), dtype=np.uint32)
        segments_info = []        
            
        # sort by score and drop those not hitting the threshold.
        used = None
        index_to_segments = {}
        for k, box in enumerate(boxes):
            # bit of debug for now.
            if not (overlaps is None):
                this_name = self.categories[self.category_thing_label_map[labels[k]]]["name"]

                for j in range(k + 1, len(boxes)):
                    if k == j:
                        continue

                    other_name = self.categories[self.category_thing_label_map[labels[j]]]["name"]

                    # check if overlap.
                    if (k, j) in overlaps:
                        value = overlaps[(k, j)]
                        if value == 0:
                            print("{0} below {1}".format(this_name, other_name))
                        else:
                            print("{0} on top of {1}".format(this_name, other_name))
                    else:
                        pass
                        #print("no overlap connection between {0} and {1}".format(this_name, other_name)

            if not (combination_debug is None):
                combination_debug.consider_segment(
                    box=box,
                    mask=masks[k],
                    category_id=self.category_thing_label_map[labels[k]],
                    score=scores[k],
                    current=panoptic_mask)
            
            if scores[k] < self.confidence_thresh:
                if not (combination_debug is None):
                    combination_debug.rejected_segment("segment has too small of confidence {0} < {1}".format(scores[k], self.confidence_thresh), low_confidence=True)
                
                continue
            
            # generate the RLE.
            mask = masks[k]
            rle = mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")

            area = mask_util.area(rle)
            if area == 0:
                # discard degenerate segmentations.
                if not (combination_debug is None):
                    combination_debug.reject_segment("segment has no area")
                    
                print("degenerate instance segmentation, ignoring")
                continue
        
            if used is None:
                intersect = 0
                used = copy.deepcopy(rle)
            else:
                intersect_mask = mask_util.merge([used, rle], intersect=True)                
                intersect = mask_util.area(intersect_mask)

                if not (combination_debug is None):
                    combination_debug.before_reclaim(
                        mask, intersect)

            intersect_before_reclaim = intersect

                # if (intersect / area > self.overlap_thresh):
                #     # highly occluded but nothing to reclaim from, surpress entirely.                
                #     continue

            # Suppose instances have been placed that highly occluded
            # this instance. We need to decide whether this object should
            # , in fact, reclaim those pixels from each instance. We only check
            # the "above" relationship because placing first handles this.
            # 1. If we want to reclaim pixels from an object, we should
            #    only take back those in the intersection.
            reclaim_segments = []

            # TODO: check if this greedy logic is OK.
            overlaps_for_object = overlaps_by_index[k]
            if not (combination_debug is None):
                debug_masks = []
                debug_bbox = []
                debug_category_ids = []
                debug_predicted = []

                debug_predicted_under = []
                debug_predicted_over = []

                for from_index, to_index in overlaps_for_object:
                    debug_masks.append(masks[to_index])
                    debug_bbox.append(boxes[to_index])
                    debug_category_ids.append(labels[to_index])
                    debug_predicted.append(overlaps[(from_index, to_index)])
                    
                    # if we haven't placed the "to" object, nothing to do for that object.
                    if not (to_index in index_to_segments):
                        print("to object has not been placed, skipping.")
                        continue

                    # to object has been placed before us.
                    overlap_value_debug = overlaps[(from_index, to_index)]
                    this_lies_above_debug = overlap_value_debug == 1
                    this_lies_below_debug = overlap_value_debug == 0

                    against_segment_debug = index_to_segments[to_index]

                    if this_lies_above_debug:
                        debug_predicted_over.append(against_segment_debug)
                    elif this_lies_below_debug:
                        debug_predicted_under.append(against_segment_debug)
                    else:
                        # hmm
                        pdb.set_trace()

                combination_debug.has_overlaps(debug_bbox, debug_masks, debug_category_ids, debug_predicted)
                combination_debug.has_segment_relationships(debug_predicted_under, debug_predicted_over)
            
            for from_index, to_index in overlaps_for_object:
                # if we haven't placed the "to" object, nothing to do for that object.
                if not (to_index in index_to_segments):
                    #print("to object has not been placed, skipping.")
                    continue

                # to object has been placed before us.
                overlap_value = overlaps[(from_index, to_index)]            
                this_lies_above = overlap_value == 1

                if this_lies_above:
                    # this should be exclusive, sanity check?
                    #print("this lies above the other object, we should reclaim")       
                    reclaim_segments.append(index_to_segments[to_index])
                 
            # # OK, we're highly occluded (more than half).
            # if (intersect / area > self.overlap_thresh) and not reclaim_segments:
            #     # highly occluded but nothing to reclaim from, surpress entirely.                
            #     continue
            #     # check if we're overlapping someone.
            #     #print("{0} might be skipped because of already claimed pixels".format(this_name))
            
            verbatim_mask = mask_util.decode(rle) == 1
            if (intersect != 0):
                # remove pixels already covered in the panoptic segmentation.
                final_mask = np.logical_and(panoptic_mask == 0, verbatim_mask)
                final_rle = mask_util.encode(np.array(final_mask.astype(np.uint8), order="F"))

                if reclaim_segments:
                    # go through each segment, find the intersection form the original mask
                    # and union it to the final mask.
                    for reclaim_segment in reclaim_segments:
                        reclaim_segment_id = reclaim_segment["id"]
                        claimed_mask = mask_util.encode(np.array((panoptic_mask == reclaim_segment_id).astype(np.uint8), order="F"))
                        overlap_mask = mask_util.merge([claimed_mask, rle], intersect=True)

                        # remove these from the overlap.
                        intersect -= mask_util.area(overlap_mask)
                        final_rle = mask_util.merge([overlap_mask, final_rle], intersect=False)

                    final_mask = mask_util.decode(final_rle) == 1

                if not (combination_debug is None):
                    intersect_mask = mask_util.merge([used, final_rle], intersect=True)
                    combination_debug.after_reclaim(mask, intersect_mask)
                    
                # OK, we're highly occluded (more than half).
                if (intersect / area > self.overlap_thresh):
                    # highly occluded but nothing to reclaim from, surpress entirely.
                    if not (combination_debug is None):
                        combination_debug.rejected_segment("segment has intersect / area of {0}/{1} = {2} > {3} (despite reclaiming from {4} segments)".format(
                            intersect, area, intersect / area, self.overlap_thresh, len(reclaim_segments)))

                    continue

                used = mask_util.merge([used, final_rle], intersect=False)
            else:
                final_mask = verbatim_mask
                used = mask_util.merge([used, rle], intersect=False)            
                
            category_id = self.category_thing_label_map[labels[k]]

            # generate an RGB color that is "close" (in RGB value) to the
            # base color for the category and not already taken by another
            # instance.
            segment_id = self.id_generator.get_id(category_id)
            segment_info = {
                "id": int(segment_id),
                "category_id": int(category_id),
            }

            index_to_segments[k] = segment_info
            
            panoptic_mask[final_mask] = segment_id
            segments_info.append(segment_info)

            if not (combination_debug is None):
                combination_debug.accepted_segment(intersect=intersect, area=area, reclaim_segments=reclaim_segments)

        # go through this one more time to get rid of "killed" segments. todo: make this unnecessary.
        surviving_segment_ids = np.unique(panoptic_mask).tolist()
        segments_info = [s for s in segments_info if s["id"] in surviving_segment_ids]

        # "stuff".
        stuff_masks = {}

        semantic_prediction = prediction.segmentation.mask.numpy()
        for y in range(image_height):
            for x in range(image_width):
                label = semantic_prediction[y, x]

                # bit of a hack for now.
                if self.stuff_start_id == 1:
                    # things are collapsed, we can't assign them, so skip them.
                    is_thing = not (label in self.category_stuff_label_map)
                    if is_thing:
                        continue

                    # only stuff can remain:
                    category_id = self.category_stuff_label_map[label]
                else:
                    # not collapsed.
                    is_thing = label < self.stuff_start_id

                    # unsure if this can be correct.
                    if is_thing:
                        continue
                        #category_id = self.category_thing_label_map[label]
                    else:
                        category_id = self.category_stuff_label_map[label]
                        
                stuff_mask = None
                if not (category_id in stuff_masks):
                    stuff_mask = np.zeros((image_height, image_width), dtype=np.uint32)
                    stuff_masks[category_id] = stuff_mask
                else:
                    stuff_mask = stuff_masks[category_id]
                    
                stuff_mask[y, x] = 1

        for category_id, segmentation_mask in stuff_masks.items():
            segmentation_mask = segmentation_mask == 1

            # take what hasn't been aassigned to an instance.
            mask_left = np.logical_and(panoptic_mask == 0, segmentation_mask)

            # if what's left isn't big enough, forget about it.
            if mask_left.sum() < self.stuff_min_area:
                continue

            # todo, really hope this is never going to output > int32 max.
            segment_id = self.id_generator.get_id(category_id)            
            segment_info = {
                "id": int(segment_id),
                "category_id": int(category_id),
            }

            panoptic_mask[mask_left] = segment_id
            segments_info.append(segment_info)

        # TODO, what if there are "unlabeled" pixels?.
        return panoptic_mask, segments_info

def combine_connected_components_pool(
        pid,
        predictions_path,
        categories,
        category_thing_label_map,
        category_stuff_label_map,        
        stuff_start_id,
        image_ids,
        image_dimensions):        
    #, dataset_path, dataset_root, predictions, image_ids, confidence_thresh, overlap_thresh, stuff_min_area):
    predictions = torch.load(predictions_path)
    print("cpu: {0} loaded {1} predictions".format(pid, len(predictions)))

    annotations = {}
    masks = {}

    segmenter = PanopticFromSemanticSegmentation(
        categories,
        category_thing_label_map,
        category_stuff_label_map,
        stuff_start_id)

    for index, image_id in enumerate(image_ids):
        if index % 5 == 0:
            print("cpu {0}: {1}/{2} processed".format(pid, index, len(image_ids)))
            
        prediction = predictions[index]
    
        image_width, image_height = image_dimensions[index]
        panoptic_mask, segments_info = segmenter(prediction, image_height, image_width)
        annotation = {
            "image_id": int(image_id),
            "segments_info": segments_info,
        }

        annotations[image_id] = annotation
        masks[image_id] = panoptic_mask

    write_path = os.path.dirname(predictions_path)

    annotations_path = os.path.join(write_path, "annotations.pth")
    masks_path = os.path.join(write_path, "masks.pth")

    print("cpu {0}: saving work".format(pid))
    torch.save(annotations, annotations_path)
    torch.save(masks, masks_path)

    return annotations_path, masks_path
    
# strategy to perform some sweet MP:
#   1. split ID array _once_into given parallelism factor.
#   2. pass this array and ID to the subprocesses. they will need to re-read the dataset, combiner, etc.
#   3. write the results to a temporary file and return the path to that.
def combine_panoptic_pool(
        pid,
        predictions_path,
        categories,
        category_thing_label_map,
        category_stuff_label_map,        
        stuff_start_id,
        image_ids,
        image_dimensions,
        confidence_thresh,
        overlap_thresh,
        stuff_min_area):
    #, dataset_path, dataset_root, predictions, image_ids, confidence_thresh, overlap_thresh, stuff_min_area):
    predictions = torch.load(predictions_path)
    print("cpu: {0} loaded {1} predictions".format(pid, len(predictions)))

    annotations = {}
    masks = {}

    combiner = PanopticNaiveCombinationMethod(
        categories,
        category_thing_label_map,
        category_stuff_label_map,
        stuff_start_id,
        confidence_thresh=confidence_thresh,
        overlap_thresh=overlap_thresh,
        stuff_min_area=stuff_min_area)        

    for index, image_id in enumerate(image_ids):
        if index % 5 == 0:
            print("cpu {0}: {1}/{2} processed".format(pid, index, len(image_ids)))
            
        prediction = predictions[index]
    
        image_width, image_height = image_dimensions[index]
        panoptic_mask, segments_info = combiner(prediction, image_height, image_width)
        annotation = {
            "image_id": int(image_id),
            "segments_info": segments_info,
        }

        annotations[image_id] = annotation
        masks[image_id] = panoptic_mask

    write_path = os.path.dirname(predictions_path)

    annotations_path = os.path.join(write_path, "annotations.pth")
    masks_path = os.path.join(write_path, "masks.pth")

    print("cpu {0}: saving work".format(pid))
    torch.save(annotations, annotations_path)
    torch.save(masks, masks_path)

    return annotations_path, masks_path

def prepare_connected_components_panoptic(predictions, dataset, working_directory="/tmp", save_panoptic_results_path=None):
    assert isinstance(dataset, PanopticCOCODataset)

    import pycocotools.mask as mask_util
    import numpy as np

    annotations = {}
    masks = {}

    cpu_count = 8
    print("restricting parallelism to {0} cpus".format(cpu_count))

    workload_size = int(math.ceil(len(predictions) / float(cpu_count)))
    print("workload size is about {0} per cpu".format(workload_size))

    image_ids = range(len(predictions))
    workload_starts = np.arange(0, len(image_ids), step=workload_size)
    number_workloads = len(workload_starts)

    print("{0} workloads being created".format(number_workloads))

    with multiprocessing.Pool(processes=cpu_count) as pool:    
        processes = []
        working_directories = []

        for workload_index in range(number_workloads):
            # last one.
            if workload_index == (number_workloads - 1):
                workload_predictions = predictions[workload_starts[workload_index]:]
                workload_image_ids = image_ids[workload_starts[workload_index]:]
            else:
                workload_predictions = predictions[workload_starts[workload_index]:workload_starts[workload_index + 1]]
                workload_image_ids = image_ids[workload_starts[workload_index]:workload_starts[workload_index + 1]]

            workload_original_image_ids = [dataset.id_to_img_map[i] for i in workload_image_ids]
            workload_image_dimensions = [(dataset.coco.imgs[i]["width"], dataset.coco.imgs[i]["height"]) for i in workload_original_image_ids]
                
            predictions_folder = tempfile.mkdtemp(dir=working_directory)
            working_directories.append(predictions_folder)
            
            predictions_path = os.path.join(predictions_folder, "predictions.pth")
            torch.save(workload_predictions, predictions_path)
                
            # process = combine_connected_components_pool(
            #     workload_index,
            #     predictions_path,
            #     dataset.coco.cats,
            #     dataset.contiguous_category_id_to_json_thing_id,
            #     dataset.contiguous_category_id_to_json_stuff_id,
            #     dataset.stuff_start,
            #     workload_original_image_ids,
            #     workload_image_dimensions)
            process = pool.apply_async(combine_connected_components_pool,
                                       (workload_index,
                                        predictions_path,
                                        dataset.coco.cats,
                                        dataset.contiguous_category_id_to_json_thing_id,
                                        dataset.contiguous_category_id_to_json_stuff_id,
                                        dataset.stuff_start,
                                        workload_original_image_ids,
                                        workload_image_dimensions))
            processes.append(process)            
        
        print("{0} processes running".format(len(processes)))

        for i, process in enumerate(processes):
            annotations_path, masks_path = process.get()

            process_annotations = torch.load(annotations_path)
            process_masks = torch.load(masks_path)

            for id in process_annotations.keys():
                annotations[id] = process_annotations[id]
                masks[id] = process_masks[id]
            
        print("removing working directories")
        for working_directory in working_directories:
            shutil.rmtree(working_directory)

    if not (save_panoptic_results_path is None):
        if not os.path.exists(save_panoptic_results_path):
            os.mkdir(save_panoptic_results_path)

        # pull out the dataset name.
        dataset_name = os.path.basename(dataset.annotation_mask_path)
        panoptic_annotations_path = os.path.join(save_panoptic_results_path, dataset_name + ".json")
        panoptic_masks_path = os.path.join(save_panoptic_results_path, dataset_name)        
                
        if not os.path.exists(panoptic_masks_path):
            os.mkdir(panoptic_masks_path)
            
        # write these out in the format panopticapi expects for verification.
        # just to make sure the order is consistent.
        list_of_annotations = []
        for id, annotation in annotations.items():
            mask = masks[id]

            padded_filename = str(annotation["image_id"]).zfill(12) + ".png"
            annotation["file_name"] = padded_filename

            Image.fromarray(id2rgb(mask)).save(
                os.path.join(panoptic_masks_path, padded_filename))
            list_of_annotations.append(annotation)

        panoptic_annotations = {
            "annotations": list_of_annotations,
            "categories": list(dataset.coco.cats.values())
        }

        with open(panoptic_annotations_path, "w") as f:
            json.dump(panoptic_annotations, f)
            
    return annotations, masks
    
IGNORE_PRE_SCORE_THRESHOLD = 0.01
SAVE_PRE_LOWEST_COUNT = 50
        
def prepare_for_coco_panoptic(
        predictions, dataset, confidence_thresh=0.5, overlap_thresh=0.5, stuff_min_area=(64 ** 2),
        working_directory="/tmp", save_panoptic_results_path=None, save_pre_results_path=None):
    assert (isinstance(dataset, PanopticCOCODataset) or isinstance(dataset, COCOTestDataset))

    import pycocotools.mask as mask_util
    import numpy as np

    annotations = {}
    masks = {}

    ## todo, generalize this?
    #max_workload = 128
    #print("restricting workload to {0} per cpu".format(max_workload))

    cpu_count = 8

    number_chunks = 1
    is_chunked = False

    if isinstance(predictions[0], str):
        number_chunks = len(predictions)
        is_chunked = True

    for chunk_number in range(number_chunks):
        print("processing chunk {0}/{1}".format(chunk_number + 1, number_chunks))
        if is_chunked:
            # load the predictions.
            chunk_predictions_path = predictions[chunk_number]
            print("loading chunk predictions: {0}".format(chunk_predictions_path))

            # split manually
            chunk_predictions = torch.load(chunk_predictions_path)

            image_ids = list(chunk_predictions.keys())
            #image_ids = [dataset.id_to_img_map[i] for i in image_indexes]
            chunk_predictions = [chunk_predictions[i] for i in image_ids]
        else:
            # just some sanity checking.
            if number_chunks > 1:
                raise ValueError("unexpected")

            chunk_predictions = predictions

            # after gathering, these are sequential.
            image_ids = range(len(chunk_predictions))
    
        print("restricting parallelism to {0} cpus".format(cpu_count))

        workload_size = int(math.ceil(len(chunk_predictions) / float(cpu_count)))
        print("workload size is about {0} per cpu".format(workload_size))

        workload_starts = np.arange(0, len(image_ids), step=workload_size)
        number_workloads = len(workload_starts)

        print("{0} workloads being created".format(number_workloads))

        with multiprocessing.Pool(processes=cpu_count) as pool:
            processes = []
            working_directories = []
            for workload_index in range(number_workloads):
                # last one.
                if workload_index == (number_workloads - 1):
                    workload_predictions = chunk_predictions[workload_starts[workload_index]:]
                    workload_image_ids = image_ids[workload_starts[workload_index]:]
                else:
                    workload_predictions = chunk_predictions[workload_starts[workload_index]:workload_starts[workload_index + 1]]
                    workload_image_ids = image_ids[workload_starts[workload_index]:workload_starts[workload_index + 1]]

                workload_original_image_ids = [dataset.id_to_img_map[i] for i in workload_image_ids]
                workload_image_dimensions = [(dataset.coco.imgs[i]["width"], dataset.coco.imgs[i]["height"]) for i in workload_original_image_ids]
                
                predictions_folder = tempfile.mkdtemp(dir=working_directory)
                working_directories.append(predictions_folder)
            
                predictions_path = os.path.join(predictions_folder, "predictions.pth")
                torch.save(workload_predictions, predictions_path)
                
                # result = combine_panoptic_pool(workload_index,
                #                                predictions_path,
                #                                dataset.coco.cats,
                #                                dataset.contiguous_category_id_to_json_thing_id,
                #                                dataset.contiguous_category_id_to_json_stuff_id,
                #                                dataset.stuff_start,
                #                                workload_original_image_ids,
                #                                workload_image_dimensions,
                #                                confidence_thresh,
                #                                overlap_thresh,
                #                                stuff_min_area)
                # pdb.set_trace()
                process = pool.apply_async(combine_panoptic_pool,
                                           (workload_index,
                                            predictions_path,
                                            dataset.coco.cats,
                                            dataset.contiguous_category_id_to_json_thing_id,
                                            dataset.contiguous_category_id_to_json_stuff_id,
                                            dataset.stuff_start,
                                            workload_original_image_ids,
                                            workload_image_dimensions,
                                            confidence_thresh,
                                            overlap_thresh,
                                            stuff_min_area))
                processes.append(process)            
            
            print("{0} processes running".format(len(processes)))

            for i, process in enumerate(processes):
                annotations_path, masks_path = process.get()

                process_annotations = torch.load(annotations_path)
                process_masks = torch.load(masks_path)

                for id in process_annotations.keys():
                    annotations[id] = process_annotations[id]
                    masks[id] = process_masks[id]
            
            print("removing working directories")
            for working_directory_remove in working_directories:
                shutil.rmtree(working_directory_remove)

    if not (save_panoptic_results_path is None):
        if not os.path.exists(save_panoptic_results_path):
            os.mkdir(save_panoptic_results_path)

        # pull out the dataset name.
        dataset_name = os.path.basename(dataset.annotation_mask_path)
        panoptic_annotations_path = os.path.join(save_panoptic_results_path, dataset_name + ".json")
        panoptic_masks_path = os.path.join(save_panoptic_results_path, dataset_name)        
                
        if not os.path.exists(panoptic_masks_path):
            os.mkdir(panoptic_masks_path)
            
        # write these out in the format panopticapi expects for verification.
        # just to make sure the order is consistent.
        list_of_annotations = []
        for id, annotation in annotations.items():
            mask = masks[id]

            padded_filename = str(annotation["image_id"]).zfill(12) + ".png"
            annotation["file_name"] = padded_filename

            Image.fromarray(id2rgb(mask)).save(
                os.path.join(panoptic_masks_path, padded_filename))
            list_of_annotations.append(annotation)

        panoptic_annotations = {
            "annotations": list_of_annotations,
            "categories": list(dataset.coco.cats.values())
        }

        with open(panoptic_annotations_path, "w") as f:
            json.dump(panoptic_annotations, f)

    return annotations, masks

def evaluate_panoptic_single(pq, categories, ground_truth_annotation, ground_truth_mask, predicted_annotation, predicted_mask, debug=False):
    # notion of consistency of segments in annotations and those actually present in the RGB mask.
    ground_truth_segments = { seg["id"]: seg for seg in ground_truth_annotation["segments_info"] }
    predicted_segments = { seg["id"]: seg for seg in predicted_annotation["segments_info"] }
    predicted_segment_ids_json = set(seg["id"] for seg in predicted_annotation["segments_info"])
    
    if len(predicted_segments) != len(predicted_segment_ids_json):
        raise Exception("segment ID must be duplicated in a prediction annotation")

    predicted_segment_ids_mask, predicted_segment_ids_mask_count = np.unique(predicted_mask, return_counts=True)
    for predicted_segment_id_mask, predicted_segment_id_mask_count in zip(predicted_segment_ids_mask, predicted_segment_ids_mask_count):
        if not (predicted_segment_id_mask in predicted_segments):
            if predicted_segment_id_mask == VOID:
                continue

            raise KeyError("In the image with ID {0} segment with ID {1} is presented in PNG and not presented in JSON.".format(
                ground_truth_annotation["image_id"], predicted_segment_id_mask))

        # fill in the area.
        predicted_segments[predicted_segment_id_mask]["area"] = predicted_segment_id_mask_count
        predicted_segment_ids_json.remove(predicted_segment_id_mask)

        if not (predicted_segments[predicted_segment_id_mask]["category_id"] in categories):
            raise KeyError("In the image with ID {0} segment with ID {1} has unknown category_id {2}.".format(
                ground_truth_annotation["image_id"], predicted_segment_id_mask, predicted_segments[predicted_segment_id_mask]["category_id"]))

    if len(predicted_segment_ids_json) != 0:
        raise KeyError("In the image with ID {0} the following segment IDs {1} are presented in JSON and not presented in PNG.".format(
            ground_truth_annotation["image_id"], list(predicted_segment_ids_json)))
        
        # # for now, just remove these segments.
        # for segment_id in predicted_segment_ids_json:
        #     del predicted_segments[segment_id]         

    # ground truth segment ID * (256 ** 3) + predicted segment ID
    # note: OFFSET > segment ID so this number mod OFFSET is the predicted ID and
    #       integer division must be the ground truth ID.
    # seems like a vectorized way of creating an image with "tuples".
    intersection_of_segments = ground_truth_mask.astype(np.uint64) * OFFSET + predicted_mask.astype(np.uint64)
    intersection_map = {}
    
    intersection_ids, intersection_id_counts = np.unique(intersection_of_segments, return_counts=True)
    for intersection_id, intersection_count in zip(intersection_ids, intersection_id_counts):
        # not really sure why Python 3 makes these non-ints.
        ground_truth_segment_id = int(intersection_id // OFFSET)
        predicted_segment_id = int(intersection_id % OFFSET)
        intersection_map[(ground_truth_segment_id, predicted_segment_id)] = intersection_count
        
    # count all matched pairs.
    ground_truth_matched = set()
    predicted_matched = set()
    for id_tuple, intersection in intersection_map.items():
        ground_truth_segment_id, predicted_segment_id = id_tuple

        # not really sure what this is safeguarding against.
        if not (ground_truth_segment_id in ground_truth_segments):
            # seems to skip VOID usually.
            continue

        if not (predicted_segment_id in predicted_segments):
            continue

        ground_truth_segment = ground_truth_segments[ground_truth_segment_id]
        predicted_segment = predicted_segments[predicted_segment_id]
        
        if ground_truth_segment["iscrowd"] == 1:
            continue

        if ground_truth_segment["category_id"] != predicted_segment["category_id"]:
            continue

        # segments are about the same category.
        union = (predicted_segment["area"] + ground_truth_segment["area"] -
                 intersection -
                 intersection_map.get((VOID, predicted_segment_id), 0))
        iou = float(intersection) / union
        if iou > 0.5:
            if debug:
                print("matched category {0} from gt {1} to pred {2}".format(ground_truth_segment["category_id"], ground_truth_segment_id, predicted_segment_id))
            # matched.
            pq[ground_truth_segment["category_id"]].tp += 1
            pq[ground_truth_segment["category_id"]].iou += iou

            ground_truth_matched.add(ground_truth_segment_id)
            predicted_matched.add(predicted_segment_id)
        else:
            if debug:
                print("IoU match of {0} ({1}) failed between gt {2} to pred {3}".format(iou, ground_truth_segment["category_id"], ground_truth_segment_id, predicted_segment_id))

    # count false negatives.
    crowd_segments_per_category = {}
    for ground_truth_segment_id, ground_truth_segment in ground_truth_segments.items():
        if ground_truth_segment_id in ground_truth_matched:
            # matched.
            continue

        ground_truth_category_id = ground_truth_segment["category_id"]
        # crowd segments don't need to be matched.
        if ground_truth_segment["iscrowd"] == 1:
            # not quite sure why this can't just overwrite?
            crowd_segments_per_category[ground_truth_category_id] = ground_truth_segment_id
            continue

        pq[ground_truth_category_id].fn += 1

    # count false positives
    for predicted_segment_id, predicted_segment in predicted_segments.items():
        if predicted_segment_id in predicted_matched:
            continue

        predicted_category_id = predicted_segment["category_id"]
            
        # intersection of the segment with VOID
        intersection = intersection_map.get((VOID, predicted_segment_id), 0)

        # plus intersection with corresponding CROWD region if it exists            
        if predicted_category_id in crowd_segments_per_category:
            intersection += intersection_map.get((crowd_segments_per_category[predicted_category_id], predicted_segment_id), 0)

        # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
        if intersection / predicted_segment["area"] > 0.5:
            continue
            
        pq[predicted_category_id].fp += 1

    return pq

def save_pre_results_pool(
        pid,
        predictions_path,
        category_thing_label_map,
        category_stuff_label_map,
        stuff_start,
        collapse_thing_ontology,
        not_stuff_contiguous_id,
        image_ids,
        image_dimensions,        
        save_masks_path):
    import pycocotools.mask as mask_util
    
    predictions = torch.load(predictions_path)
    print("cpu: {0} loaded {1} predictions".format(pid, len(predictions)))

    masker = Masker(threshold=0.5, padding=1)
        
    # we'll put instance mask annotations here (up to some confidence).
    annotations = {}
    overlaps = {}
    
    # we need: dataset, image ids to process, predictions map. image data
    for index, image_id in enumerate(image_ids):
        if index % 5 == 0:
            print("cpu {0}: {1}/{2} processed".format(pid, index, len(image_ids)))
        
        prediction = predictions[image_id]
        image_width, image_height = image_dimensions[index]
        prediction = prediction.resize((image_width, image_height))
        base_annotation = {
            "image_id": image_id,
        }
            
        # write out the semantic mask as a PNG with category IDs (like COCO-Stuff).
        contiguous_mask = prediction.segmentation.mask.numpy()
        semantic_mask = np.zeros_like(contiguous_mask, dtype=np.uint8)

        for y in range(image_height):
            for x in range(image_width):
                label = contiguous_mask[y, x]

                # void.
                if label == 0:
                    continue

                # thing (collapsed)
                if collapse_thing_ontology and (label == not_stuff_contiguous_id):
                    continue

                # thing (not collapsed):
                if label < stuff_start:
                    semantic_mask[y, x] = category_thing_label_map[label]
                else:
                    # stuff.
                    semantic_mask[y, x] = category_stuff_label_map[label]

        semantic_annotation = dict(base_annotation)
        padded_filename = str(image_id).zfill(12) + ".png"
        semantic_annotation["file_name"] = padded_filename

        Image.fromarray(semantic_mask).save(
            os.path.join(save_masks_path, padded_filename))
        annotations[image_id] = [semantic_annotation]

        # now add annotations for each instance. this should be at maximum 100 (COCO).
        #region = prediction.region.sorted_by_confidence()
        region = prediction.region
        masks = region.get_field("mask")
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), region)
            # this should be the "most confident" mask per detection (check?)
            masks = masks[0]
            
        region = region.convert("xywh")
        boxes = region.bbox.numpy()
        box_labels = region.get_field("labels").numpy()
        box_scores = region.get_field("scores").numpy()
        box_areas = region.area().numpy()            

        for box_index in range(len(prediction.region)):
            # not sure if we need to set an ID.
            box_annotation = dict(base_annotation)

            box = boxes[box_index]
            box_label = box_labels[box_index]
            box_score = box_scores[box_index]

            if box_score < IGNORE_PRE_SCORE_THRESHOLD:
                break
                
            box_area = box_areas[box_index]
            mask = masks[box_index]
            rle = mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")

            box_annotation.update({
                "category_id": int(category_thing_label_map[box_label]),
                "segmentation": rle,
                "score": float(box_score),
                "area": float(box_area),
                "bbox": [float(b) for b in box.tolist()]
            })
            annotations[image_id].append(box_annotation)

        # now save the overlaps if they exist.
        if prediction.region.has_field("overlap"):
            # create an overlap matrix.
            overlap_matrix = -1 * np.ones((len(prediction.region), len(prediction.region)), dtype=np.int8)

            overlap_pairs = prediction.region.get_field("overlap_pairs")
            overlap_predictions = prediction.region.get_field("overlap")

            for pair_index in range(overlap_pairs.shape[0]):
                overlap_pair = overlap_pairs[pair_index]
                overlap_prediction = int(overlap_predictions[pair_index])
                
                from_index, to_index = int(overlap_pair[0]), int(overlap_pair[1])

                # enforce the anti symmetry.
                overlap_matrix[from_index, to_index] = overlap_prediction

            overlaps[image_id] = overlap_matrix
            
    write_path = os.path.dirname(predictions_path)
    annotations_path = os.path.join(write_path, "annotations.pth")

    print("cpu {0}: saving work".format(pid))
    torch.save(annotations, annotations_path)
    
    if len(overlaps) > 0:
        overlaps_path = os.path.join(write_path, "overlaps.pth")
        torch.save(overlaps, overlaps_path)
    else:
        overlaps_path = None

    return annotations_path, overlaps_path

def save_pre_results(predictions, dataset, save_pre_results_path, sorted_by_pq_image_ids, working_directory="/tmp"):
    import pycocotools.mask as mask_util
    import numpy as np

    if not os.path.exists(save_pre_results_path):
        os.mkdir(save_pre_results_path)

    dataset_name = os.path.basename(dataset.annotation_mask_path)
    pre_annotations_path = os.path.join(save_pre_results_path, dataset_name + ".json")
    pre_overlaps_path = os.path.join(save_pre_results_path, dataset_name + "_overlaps.pth")
    semantic_masks_path = os.path.join(save_pre_results_path, dataset_name)

    if not os.path.exists(semantic_masks_path):
        os.mkdir(semantic_masks_path)

    annotations_map = {}
    overlaps_map = {}

    cpu_count = 8
    print("restricting parallelism to {0} cpus".format(cpu_count))

    workload_size = int(math.ceil(len(predictions) / float(cpu_count)))
    print("workload size is about {0} per cpu".format(workload_size))

    image_ids = list(predictions.keys())
    workload_starts = np.arange(0, len(image_ids), step=workload_size)
    number_workloads = len(workload_starts)

    print("{0} workloads being created".format(number_workloads))

    with multiprocessing.Pool(processes=cpu_count) as pool:    
        processes = []
        working_directories = []

        for workload_index in range(number_workloads):
            # last one.
            if workload_index == (number_workloads - 1):
                workload_image_ids = image_ids[workload_starts[workload_index]:]
            else:
                workload_image_ids = image_ids[workload_starts[workload_index]:workload_starts[workload_index + 1]]

            workload_predictions = {i: predictions[i] for i in workload_image_ids}
            workload_image_dimensions = [(dataset.coco.imgs[i]["width"], dataset.coco.imgs[i]["height"]) for i in workload_image_ids]
                
            predictions_folder = tempfile.mkdtemp(dir=working_directory)
            working_directories.append(predictions_folder)

            # todo, don't save twice.
            predictions_path = os.path.join(predictions_folder, "predictions.pth")
            torch.save(workload_predictions, predictions_path)

            process = pool.apply_async(save_pre_results_pool,
                                       (workload_index,
                                        predictions_path,
                                        dataset.contiguous_category_id_to_json_thing_id,
                                        dataset.contiguous_category_id_to_json_stuff_id,
                                        dataset.stuff_start,
                                        dataset.collapse_thing_ontology,
                                        dataset.not_stuff_contiguous_id if dataset.collapse_thing_ontology else -1, 
                                        workload_image_ids,
                                        workload_image_dimensions,
                                        semantic_masks_path))
            processes.append(process)            
        
        print("{0} processes running".format(len(processes)))

        for i, process in enumerate(processes):
            # masks should already be written.
            annotations_path, overlaps_path = process.get()

            process_annotations = torch.load(annotations_path)
            if not (overlaps_path is None):
                process_overlaps = torch.load(overlaps_path)

                for id in process_overlaps.keys():
                    overlaps_map[id] = process_overlaps[id]

            for id in process_annotations.keys():
                annotations_map[id] = process_annotations[id]

            
        print("removing working directories")
        for working_directory in working_directories:
            shutil.rmtree(working_directory)

        # re-order the annotations by the sorted IDs.
        annotations = []
        for image_id in sorted_by_pq_image_ids:
            annotations.extend(annotations_map[image_id])
        
        pre_data = {
            "annotations": annotations,
            "categories": list(dataset.coco.cats.values())
        }

        with open(pre_annotations_path, "w") as f:
            json.dump(pre_data, f)

        if len(overlaps_map) > 0:
            torch.save(overlaps_map, pre_overlaps_path)

def evaluate_panoptic(predictions, dataset, predicted_annotations, predicted_masks,
                      save_panoptic_results_path=None,
                      save_pre_results_path=None,
                      working_directory="/tmp",
                      kind="pan"):
    import pycocotools.mask as mask_util    
    print("evaluating panoptic")

    coco = dataset.coco
    categories = coco.cats
    
    # match these up.
    ground_truth_image_ids = coco.getImgIds()

    # same number.
    if len(predicted_annotations) != len(ground_truth_image_ids):
        raise Exception("mismatch between number of predictions ({0}) and ground truths ({1})".format(
            len(predicted_annotations), len(ground_truth_image_ids)))

    # same image IDs.
    for image_id in ground_truth_image_ids:
        if (not (image_id in predicted_annotations)) or (not (image_id in predicted_masks)):
            raise Exception("missing prediction annotation or mask for image ID {0}".format(image_id))

    # actually compare now.
    pqs = []
    pq_overall = PQStat()
    
    start_time = time.time()
    for image_id in ground_truth_image_ids:
        ground_truth_annotation = coco.loadAnns(image_id)
        if len(ground_truth_annotation) != 1:
            raise Exception("expected only a single panoptic annotation")

        ground_truth_annotation = ground_truth_annotation[0]
        ground_truth_mask = rgb2id(np.array(Image.open(
            os.path.join(dataset.annotation_mask_path, ground_truth_annotation["file_name"])),
            dtype=np.uint32))
        
        predicted_annotation = predicted_annotations[image_id]
        predicted_mask = predicted_masks[image_id]

        pq = PQStat()
        evaluate_panoptic_single(
            pq,
            categories,
            ground_truth_annotation,
            ground_truth_mask,
            predicted_annotation,
            predicted_mask)

        quality = {}
        
        pq_results, _ = pq.pq_average(categories, isthing=None)
        quality["pq"] = 100.0 * pq_results["pq"]
        
        # sometimes these throw exceptions.
        try:
            pq_stuff, _ = pq.pq_average(categories, isthing=False)
            quality["pq_st"] = 100.0 * pq_stuff["pq"]
        except:
            quality["pq_st"] = 0.0

        try:
            pq_thing, _ = pq.pq_average(categories, isthing=True)
            quality["pq_th"] = 100.0 * pq_thing["pq"]
        except:
            quality["pq_th"] = 0.0

        pqs.append(quality)
        pq_overall += pq

    # now that we have PQ per image, fill this in if desired in the JSON.
    if not (save_panoptic_results_path is None):
        dataset_name = os.path.basename(dataset.annotation_mask_path)
        panoptic_annotations_path = os.path.join(save_panoptic_results_path, dataset_name + ".json")

        with open(panoptic_annotations_path, "r") as f:
            panoptic_data = json.load(f)
            panoptic_annotations = panoptic_data["annotations"]
            id_to_annotation = {
                a["image_id"]: a for a in panoptic_annotations
            }
            
            for i, image_id in enumerate(ground_truth_image_ids):
                panoptic_annotation = id_to_annotation[image_id]
                panoptic_annotation["quality"] = pqs[i]

        with open(panoptic_annotations_path, "w") as f:
            json.dump(panoptic_data, f)

    # sort these by lowest PQ to highest.
    indexes = range(len(ground_truth_image_ids))
    sorted_by_pq_indexes = sorted(indexes, key=lambda i: pqs[i]["pq"])
    sorted_by_pq_image_ids = [ground_truth_image_ids[i] for i in sorted_by_pq_indexes]

    if not (save_pre_results_path is None):
        save_pre_results(predictions, dataset, save_pre_results_path, sorted_by_pq_image_ids, working_directory=working_directory)

    total_delta = time.time() - start_time
    print("total time elapsed: {:0.2f} seconds".format(total_delta))

    # package these into a COCO eval.
    from pycocotools.cocoeval import COCOeval
    metrics = [("PQ", None), ("PQ_Th", True), ("PQ_St", False)]

    eval = COCOeval(iouType=kind)
    for name, isthing in metrics:
        metric_results, _ = pq_overall.pq_average(categories, isthing=isthing)
        eval.stats.append(100 * metric_results["pq"])

    return eval

# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }

def evaluate_predictions_on_coco(
    coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval

class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "pan": ["PQ", "PQ_Th","PQ_St"],
        "ccpan": ["PQ", "PQ_Th","PQ_St"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoint": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "ccpan", "pan")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)

def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)
