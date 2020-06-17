# Learning Instance Occlusion

This is the code for the CVPR 2020 [paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Lazarow_Learning_Instance_Occlusion_for_Panoptic_Segmentation_CVPR_2020_paper.pdf) "Learning Instance Occlusion for Panoptic Segmentation".

This project is based off of the excellent [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). We extend it to:

1. Support [Panoptic Segmentation/Panoptic FPN](https://arxiv.org/abs/1901.02446)
2. Support learning an instance-wise relationship to determine occlusion
3. Integrate this into the [existing greedy merging heuristic](https://arxiv.org/abs/1801.00868)

If you make use of the ideas or code in this project, please consider citing:

```
@InProceedings{Lazarow_2020_CVPR,
author = {Lazarow, Justin and Lee, Kwonjoon and Shi, Kunyu and Tu, Zhuowen},
title = {Learning Instance Occlusion for Panoptic Segmentation},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
