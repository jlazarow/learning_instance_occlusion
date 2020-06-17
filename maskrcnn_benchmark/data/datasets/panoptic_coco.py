import numpy as np
import os
import pickle
import torch
import torchvision
from PIL import Image

from collections import OrderedDict

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.panoptic import PanopticTarget, SemanticSegmentation
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

import pdb

RGB_BASE = 256

class PanopticCOCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, remove_images_without_annotations,
            transforms=None, collapse_thing_ontology=True,
            overlaps_path=None,
            remove_empty=True):
        super(PanopticCOCODataset, self).__init__(root, ann_file)

        self.annotation_path = ann_file
        self.collapse_thing_ontology = collapse_thing_ontology
        self.overlaps_path = overlaps_path
        self.remove_empty = remove_empty

        # not the best way.
        from pycocotools.coco import COCO
        annotations_path = os.path.dirname(self.annotation_path)
        panoptic_annotation_filename = os.path.basename(self.annotation_path)

        self.annotation_mask_path = os.path.join(
            annotations_path, os.path.splitext(panoptic_annotation_filename)[0])
        self.instances_annotation_path = os.path.join(
            annotations_path, panoptic_annotation_filename.replace("panoptic", "instances"))
        self.coco_instances = COCO(self.instances_annotation_path)

        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # this is interesting because it appears that there are always _panoptic annotations_
        # but not necessarily instance ones (do we need to compute these on the fly)?
        if remove_images_without_annotations:
            before_removal_count = len(self.ids)
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco_instances.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]
            
            after_removal_count = len(self.ids)
            print("NOTE: removed {0} images because they lack instance annotations".format(
                before_removal_count - after_removal_count))

        self.overlaps = None
        if self.overlaps_path:
            self.overlaps_path = os.path.join(annotations_path, overlaps_path)
                    
            with open(self.overlaps_path, "rb") as handle:
                # ID -> overlap matrix.
                self.overlaps = pickle.load(handle)

        # 0 is void.
        self.stuff_start = 1
        self.json_thing_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds(kind="thing"))
        }
        if not self.collapse_thing_ontology:
            self.stuff_start = max(self.json_thing_category_id_to_contiguous_id.values()) + 1
        
        self.json_stuff_category_id_to_contiguous_id = {
            v: i + self.stuff_start for i, v in enumerate(self.coco.getCatIds(kind="stuff"))
        }
        self.contiguous_category_id_to_json_thing_id = {
            v: k for k, v in self.json_thing_category_id_to_contiguous_id.items()
        }
        self.contiguous_category_id_to_json_stuff_id = {
            v: k for k, v in self.json_stuff_category_id_to_contiguous_id.items()
        }

        # for segmenting non-stuff pixels.
        if self.collapse_thing_ontology:
            # don't define this otherwise on purpose.
            self.not_stuff_contiguous_id = max(self.json_stuff_category_id_to_contiguous_id.values()) + 1
        else:
            print("dense ontology!")
            
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    # anno is a set of instances annotations.
    def _get_instance_box_list(self, img, anno, image_id):
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes                
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        # assume these are consistent.
        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_thing_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        if not (self.overlaps is None):
            overlaps = np.copy(self.overlaps[image_id])

            # todo: remove once we know datasets are "anti-symmetric".
            number_objects = overlaps.shape[0]

            # quick sanity check for now.
            #if number_objects != len(boxes):
            #    raise ValueError("image id {0} has mismatch of boxes {1} to overlaps {2}".format(image_id, len(boxes), number_objects))
            
            for y in range(number_objects):
                for x in range(y + 1, number_objects):        
                    current = overlaps[y, x]
                    if current == 0:
                        overlaps[x, y] = 1
                    elif current == 1:
                        overlaps[x, y] = 0
            
            overlaps = torch.tensor(overlaps).float()
            target.add_field("overlaps", overlaps)

        return target

    # anno is a panoptic annotation.
    def _get_semantic_segmentation(self, anno):
        # load the image mask.
        mask_img_filename = anno['file_name']
        mask_img = Image.open(os.path.join(
            os.path.join(self.annotation_mask_path, mask_img_filename))).convert('RGB')
        mask_img.load()

        # H x W x C -> C x H x W and convert to IDs so that we can match these to segments.
        id_mask = np.array(mask_img)
        id_mask = id_mask[:, :, 0] + RGB_BASE * id_mask[:, :, 1] + (RGB_BASE ** 2) * id_mask[:, :, 2]

        return self._get_semantic_segmentation_from_id(anno, id_mask)

    def _get_semantic_segmentation_from_id(self, anno, id_mask):
        # build an ID to "class" mapping. if not "stuff", map to special "other ID".
        id_map = {}
        for segment in anno["segments_info"]:
            segment_id = segment["id"]
            category_id = segment["category_id"]

            if category_id in self.json_stuff_category_id_to_contiguous_id:
                # stuff!
                id_map[segment_id] = self.json_stuff_category_id_to_contiguous_id[category_id]
            else:
                # thing! collapse it or not?
                id_map[segment_id] = (self.not_stuff_contiguous_id if self.collapse_thing_ontology else
                                      self.json_thing_category_id_to_contiguous_id[category_id])

        # apply this map to the image.
        segmentation_mask = np.zeros_like(id_mask).astype(np.int32)
        has_label_mask = np.zeros_like(id_mask).astype(np.int32)
        height, width = id_mask.shape

        for y in range(height):
            for x in range(width):
                pixel_id = id_mask[y, x]

                # this pixel might be unlabeled.
                if pixel_id > 0:
                    segmentation_mask[y, x] = id_map[id_mask[y, x]]
                    has_label_mask[y, x] = 1
        
        return SemanticSegmentation(mask=torch.tensor(segmentation_mask), has_label_mask=torch.tensor(has_label_mask))

    def _create_region_from_prediction(self, img, anno, image_id):
        import pycocotools.mask as mask_util
        
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes                
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        # assume these are consistent.
        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_thing_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = torch.tensor(np.expand_dims(np.array([mask_util.decode(obj["segmentation"]) for obj in anno]), axis=1))
        target.add_field("mask", masks)

        scores = [obj["score"] for obj in anno]
        scores = torch.tensor(scores)
        target.add_field("scores", scores)
        
        if not (self.overlaps is None):
            overlaps = np.copy(self.overlaps[image_id])

            # todo: remove once we know datasets are "anti-symmetric".
            number_objects = overlaps.shape[0]

            # quick sanity check for now.
            #if number_objects != len(boxes):
            #    raise ValueError("image id {0} has mismatch of boxes {1} to overlaps {2}".format(image_id, len(boxes), number_objects))
            
            for y in range(number_objects):
                for x in range(y + 1, number_objects):        
                    current = overlaps[y, x]
                    if current == 0:
                        overlaps[x, y] = 1
                    elif current == 1:
                        overlaps[x, y] = 0
            
            overlaps = torch.tensor(overlaps).float()
            target.add_field("overlaps", overlaps)

        return target
        

    def _category_to_contiguous_segmentation_mask(self, segmentation_mask):
        has_label_mask = np.zeros_like(segmentation_mask).astype(np.int32)
        height, width = segmentation_mask.shape

        for y in range(height):
            for x in range(width):
                category_id = segmentation_mask[y, x]
                if category_id == 0:
                    continue

                if category_id in self.json_stuff_category_id_to_contiguous_id:
                    # stuff!
                    contiguous_id = self.json_stuff_category_id_to_contiguous_id[category_id]
                else:
                    # thing! collapse it or not?
                    contiguous_id = (self.not_stuff_contiguous_id if self.collapse_thing_ontology else
                                     self.json_thing_category_id_to_contiguous_id[category_id])

                # this pixel might be unlabeled.
                segmentation_mask[y, x] = contiguous_id
                has_label_mask[y, x] = 1
        
        return SemanticSegmentation(mask=torch.tensor(segmentation_mask), has_label_mask=torch.tensor(has_label_mask))

    def prepare_image(self, img, panoptic_annotations, instance_annotations, image_id):
        # note there should only be one "anno".
        # original code filtered "crowsds", should we for segment infos?
        if len(panoptic_annotations) != 1:
            raise Exception('expected only a single annotation for panoptic')

        panoptic_annotation = panoptic_annotations[0]

        # for Mask-RCNN.
        region = self._get_instance_box_list(img, instance_annotations, image_id=image_id)

        # for Semantic Segmentation.
        segmentation = self._get_semantic_segmentation(panoptic_annotation)

        target = PanopticTarget(region=region, segmentation=segmentation)
        target = target.clip_to_image(remove_empty=self.remove_empty)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __getitem__(self, idx):
        # load both for now.
        img, panoptic_annotations = super(PanopticCOCODataset, self).__getitem__(idx)
        instance_annotation_ids = self.coco_instances.getAnnIds(imgIds=self.ids[idx])
        instance_annotations = self.coco_instances.loadAnns(instance_annotation_ids)

        image_id = self.id_to_img_map[idx]
        img, target = self.prepare_image(img, panoptic_annotations, instance_annotations, image_id=image_id)
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
