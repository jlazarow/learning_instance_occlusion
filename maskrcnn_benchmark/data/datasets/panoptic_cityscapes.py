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

# note that this uses "COCO" style annotations for instances.
class PanopticCityscapesDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, remove_images_without_annotations,
            transforms=None, collapse_thing_ontology=True,
            remove_empty=True):
        super(PanopticCityscapesDataset, self).__init__(root, ann_file)

        # this should be the path to "cityscapes_fine_instanceonly_seg_*_cocostyle"
        self.annotation_path = ann_file

        annotations_root = os.path.dirname(self.annotation_path)
        dataset_type = os.path.basename(self.root)

        self.segmentation_root = os.path.join(annotations_root, "panoptic_{0}".format(dataset_type))
        
        self.collapse_thing_ontology = collapse_thing_ontology
        self.remove_empty = remove_empty

        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # this is interesting because it appears that there are always _panoptic annotations_
        # but not necessarily instance ones.
        if remove_images_without_annotations:
            before_removal_count = len(self.ids)
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]
            
            after_removal_count = len(self.ids)
            print("NOTE: removed {0} images because they lack instance annotations".format(
                before_removal_count - after_removal_count))

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

    def _get_semantic_segmentation(self, image_data):
        # read the semantic segmentation (hack for now).
        segmentation_labels_filename = image_data["seg_file_name"].replace("instanceIds", "labelIds")

        label_image = Image.open(os.path.join(self.segmentation_root, segmentation_labels_filename))
        label_image.load()

        segmentation_labels = np.array(label_image)

        segmentation_mask = np.zeros_like(segmentation_labels).astype(np.int32)
        has_label_mask = np.zeros_like(segmentation_labels).astype(np.int32)
        height, width = segmentation_mask.shape

        for y in range(height):
            for x in range(width):
                label_id = segmentation_labels[y, x]

                # this pixel might be unlabeled or unused.
                if label_id in self.json_thing_category_id_to_contiguous_id:
                    segmentation_mask[y, x] = self.json_thing_category_id_to_contiguous_id[label_id]
                    has_label_mask[y, x] = 1
                elif label_id in self.json_stuff_category_id_to_contiguous_id:
                    segmentation_mask[y, x] = self.json_stuff_category_id_to_contiguous_id[label_id]
                    has_label_mask[y, x] = 1

        return SemanticSegmentation(mask=torch.tensor(segmentation_mask), has_label_mask=torch.tensor(has_label_mask))

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

        return target

    def prepare_image(self, img, instance_annotations, image_id):
        image_data = self.coco.loadImgs(image_id)[0]

        # for Mask-RCNN.
        region = self._get_instance_box_list(img, instance_annotations, image_id=image_id)

        # for Semantic Segmentation.
        segmentation = self._get_semantic_segmentation(image_data)

        target = PanopticTarget(region=region, segmentation=segmentation)
        target = target.clip_to_image(remove_empty=self.remove_empty)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target        

    def __getitem__(self, idx):
        # read the instance annotations.
        img, instance_annotations = super(PanopticCityscapesDataset, self).__getitem__(idx)
        image_id = self.id_to_img_map[idx]

        img, target = self.prepare_image(img, instance_annotations, image_id=image_id)
        
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
