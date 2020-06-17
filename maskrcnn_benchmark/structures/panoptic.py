import numpy as np
import PIL
import torch

import pdb

from PIL import Image

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
RGB_BASE = 256

class SemanticSegmentation(object):
    # mask is a mask of class IDs, has_label_mask is binary for whether
    # the pixel is labeled.
    def __init__(self, mask, has_label_mask):
        self.mask = mask
        self.has_label_mask = has_label_mask
        # will Mask RCNN put the previous on the GPU and then call augmentation methods?

    def to(self, device):
        return SemanticSegmentation(
            mask=self.mask.to(device),
            has_label_mask=self.has_label_mask.to(device) if not (self.has_label_mask is None) else None)

    def resize(self, size, *args, **kwargs):
        if (self.mask.device.type != 'cpu') and (self.has_label_mask.device.type != 'cpu'):
            raise Exception('expected tensors to be on the CPU for resizing')

        # todo, put this all on the GPU/CPU without going back to PIL.
        mask_np = self.mask.numpy().astype(np.uint8)
        mask_img = Image.fromarray(mask_np)
        resized_mask_img = mask_img.resize(size, resample=PIL.Image.NEAREST)

        if not (self.has_label_mask is None):
            has_label_mask_np = self.has_label_mask.numpy().astype(np.uint8)
            has_label_mask_img = Image.fromarray(has_label_mask_np)
            resized_has_label_mask_img = has_label_mask_img.resize(size, resample=PIL.Image.NEAREST)
            resized_has_label_mask = torch.tensor(np.array(resized_has_label_mask_img).astype(np.int32))
        else:
            resized_has_label_mask = None
            
        # seems like this should be consistent, but a bit unsure.
        return SemanticSegmentation(
            mask=torch.tensor(np.array(resized_mask_img).astype(np.int32)),
            has_label_mask=resized_has_label_mask)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        if method == FLIP_LEFT_RIGHT:
            flipped_mask = torch.flip(self.mask, [1])
            flipped_has_label_mask = torch.flip(self.has_label_mask, [1])
        elif method == FLIP_TOP_BOTTOM:
            flipped_mask = torch.flip(self.mask, [0])
            flipped_has_label_mask = torch.flip(self.has_label_mask, [0])

        return SemanticSegmentation(mask=flipped_mask, has_label_mask=flipped_has_label_mask)
        
    def crop(self, box):
        cropped_mask = self.mask[box[1]:box[3], box[0]:box[2]]
        cropped_has_label_mask = self.has_label_mask[box[1]:box[3], box[0]:box[2]]

        return SemanticSegmentation(mask=cropped_mask, has_label_mask=cropped_has_label_mask)

class PanopticTarget(object):
    # region: Mask RCNN object information
    # segmentation: Semantic segmentation class map.
    def __init__(self, region, segmentation):
        self.region = region
        self.segmentation = segmentation

    def to(self, device):
        return PanopticTarget(
            region=self.region.to(device),
            segmentation=self.segmentation.to(device))

    def clip_to_image(self, remove_empty=True):
        # clip the bounding box.
        self.region = self.region.clip_to_image(remove_empty=remove_empty)
        
        # doesn't seem like there should be anything to clip for the
        # segmentation.
        return self

    def resize(self, size, *args, **kwargs):
        return PanopticTarget(
            region=self.region.resize(size, *args, **kwargs),
            segmentation=self.segmentation.resize(size, *args, **kwargs))

    def transpose(self, method):
        return PanopticTarget(
            region=self.region.transpose(method),
            segmentation=self.segmentation.transpose(method))

    def crop(self, box):
        return PanopticTarget(
            region=self.region.crop(box),
            segmentation=self.segmentation.crop(box))
        
        
