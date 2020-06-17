# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

import pdb


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def prepare_image(self, img, anno):
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        img, target = self.prepare_image(img, anno)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


# I guess we need this.
class FakeTarget(object):
    def to(self, device):
        return self

    def clip_to_image(self, remove_empty=True):
        return self

    def resize(self, size, *args, **kwargs):
        return self

    def transpose(self, method):
        return self

    def crop(self, box):
        return self

# ideally should be agnostic to underlying dataset but for
# now it will be panoptic specific.
class COCOTestDataset(torchvision.datasets.coco.CocoTest):
    def __init__(self, ann_file, root, transforms=None, collapse_thing_ontology=False):
        super(COCOTestDataset, self).__init__(root, ann_file)

        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        self.collapse_thing_ontology = collapse_thing_ontology

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

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

        self.annotation_mask_path = "testdev"
        
        self.transforms = transforms
        self.fake_target = FakeTarget()        

    def __getitem__(self, idx):
        img = super(COCOTestDataset, self).__getitem__(idx)

        if self.transforms is not None:
            img, _ = self.transforms(img, self.fake_target)
 
        return img, None, idx        
        
    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
