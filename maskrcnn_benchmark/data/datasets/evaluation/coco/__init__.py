from .coco_eval import do_coco_evaluation


def coco_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    working_directory,
    save_panoptic_results,
    save_pre_results,
    panoptic_confidence_thresh,
    panoptic_overlap_thresh,
    panoptic_stuff_min_area):
    return do_coco_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        working_directory=working_directory,
        save_panoptic_results=save_panoptic_results,
        save_pre_results=save_pre_results,
        panoptic_confidence_thresh=panoptic_confidence_thresh,
        panoptic_overlap_thresh=panoptic_overlap_thresh,
        panoptic_stuff_min_area=panoptic_stuff_min_area)
