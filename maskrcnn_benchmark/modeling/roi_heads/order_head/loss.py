import numpy as np

import pycocotools.mask as mask_util
import torch

import pdb

def prepare_mask_intersection_matrix(boxes, hard_masks):
    import pycocotools.mask as mask_util
    
    rles = [
        mask_util.encode(np.array(hard_mask[0, :, :, np.newaxis], order="F"))[0]
        for hard_mask in hard_masks
    ]

    iscrowd = []
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")
        iscrowd.append(0)
    
    # quickly find those that intersect at all.
    iou = mask_util.iou(rles, rles, iscrowd)
    number_masks = len(rles)

    # zeros prevents diagonals from ever being considered.
    intersect_ratio = np.zeros_like(iou, dtype=np.float32)

    # note, these masks are variable size. unsure whether to do this altogether.
    #intersect_bbox = np.zeros((number_masks, number_masks, 4), dtype=np.int32)
    
    for from_index in range(intersect_ratio.shape[0]):
        from_rle = rles[from_index]
        from_area = mask_util.area(from_rle)

        # only need the upper triangle.
        for to_index in range(from_index + 1, intersect_ratio.shape[1]):
            to_rle = rles[to_index]
            to_area = mask_util.area(to_rle)

            # this is symmetric.
            merged = mask_util.merge([from_rle, to_rle], intersect=True)

            # do we compute the bbox here or wait for later? add some padding?
            #merged_bbox = mask_util.toBbox(merged)
            #intersect_bbox[from_index, to_index] = merged_bbox[0]
            #intersect_bbox[to_index, from_index] = merged_bbox[0]
            
            from_to_intersect = float(mask_util.area(merged))

            from_ratio = from_to_intersect / (from_area + 0.0001)
            to_ratio = from_to_intersect / (to_area + 0.0001)

            # from -> to.
            intersect_ratio[from_index, to_index] = from_ratio
            intersect_ratio[to_index, from_index] = to_ratio

    intersect_ratio = torch.tensor(intersect_ratio).to(boxes.bbox.device)
    #intersect_bbox = torch.tensor(intersect_bbox).to(boxes.bbox.device)
    
    return intersect_ratio#, intersect_bbox

def filter_actual_overlaps(target, matching, proposal_pairs):
    if not target.has_field("overlaps"):
        raise ValueError("overlaps do not exist on target")

    # from the "overlap" ground truth, we can infer the ground truth class that survives the overlap.
    overlaps = target.get_field("overlaps")

    first_idxs = torch.unsqueeze(matching[proposal_pairs[:, 0]], dim=1)
    second_idxs = torch.unsqueeze(matching[proposal_pairs[:, 1]], dim=1)
                
    selected_overlaps = overlaps[first_idxs, second_idxs]
            
    actual_overlaps = selected_overlaps >= 0            
    mask_of_overlaps = torch.nonzero(actual_overlaps)[:, 0]
    
    return mask_of_overlaps, selected_overlaps

def subsample_actual_overlaps(mask_of_overlaps, maximum_per_image=None):
    if maximum_per_image is None:
        return mask_of_overlaps
    
    number_masked = mask_of_overlaps.shape[0]
    subsample_size = min(maximum_per_image, number_masked)
    subsample_perm = torch.randperm(number_masked, device=mask_of_overlaps.device)[:subsample_size]
        
    return mask_of_overlaps[subsample_perm]

def compute_overlap_matrix(target):
    width, height = target.size
    masks = target.get_field("masks")

    rles = []
    iscrowd = []
    for poly in masks.polygons:
        encoded = mask_util.frPyObjects(poly.polygons, height, width)
        
        rles.append(encoded[0])
        iscrowd.append(0)

    return prepare_mask_intersection_matrix(target, rles, iscrowd)
    
    
    
