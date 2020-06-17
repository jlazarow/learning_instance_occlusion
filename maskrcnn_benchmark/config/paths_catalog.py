# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/mnt/cube/datasets"
    DATASETS = {
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },                
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "stuff_coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/stuff_train2017.json"
        },                
        "stuff_coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/stuff_val2017.json"
        },                
        "panoptic_coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/panoptic_train2017.json"
        },
        "panoptic_coco_2017_train_overlap005": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/panoptic_train2017.json",
            "overlaps_path": "panoptic_train_overlap_005.npy",
        },                        
        "panoptic_coco_2017_train_overlap010": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/panoptic_train2017.json",
            "overlaps_path": "panoptic_train_overlap_01.npy",
        },                        
        "panoptic_coco_2017_train_overlap02": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/panoptic_train2017.json",
            "overlaps_path": "panoptic_train_overlap_02.npy",
        },                
        "panoptic_coco_2017_train_overlap035": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/panoptic_train2017.json",
            "overlaps_path": "panoptic_train_overlap_035.npy",
        },                
        "panoptic_coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/panoptic_val2017.json"
        },        
        "panoptic_coco_2017_val_overlap02": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/panoptic_val2017.json",
            "overlaps_path": "panoptic_val_overlap_02.npy",
        },        
        "panoptic_coco_2017_val_overlap02": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/panoptic_val2017.json",
            "overlaps_path": "panoptic_val_overlap_02.npy",
        },        
        "panoptic_coco_2017_tiny": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/panoptic_tiny2017.json",
        },
        # DO NOT TRAIN, this is for leaderboard (nor could you, there are no ground truths).
        "panoptic_coco_2017_test_dev": {
            "img_dir": "coco/test2017",
            "ann_file": "coco/annotations/panoptic_testdev2017.json"
        },                
        "panoptic_coco_2017_test_dev_tiny": {
            "img_dir": "coco/test2017",
            "ann_file": "coco/annotations/panoptic_testdev2017_tiny.json"
        },                
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "panoptic_cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/fine/train",
            "ann_file": "cityscapes/coco-annotations/panoptic_fine_train.json",            
        },
        "panoptic_cityscapes_fine_instanceonly_seg_train_cocostyle_overlap02": {
            "img_dir": "cityscapes/fine/train",
            "ann_file": "cityscapes/coco-annotations/panoptic_fine_train.json",
            "overlaps_path": "panoptic_fine_train_overlap_02_precedence_fixed.npy",
        },
        "panoptic_cityscapes_fine_instanceonly_seg_train_cocostyle_overlap005": {
            "img_dir": "cityscapes/fine/train",
            "ann_file": "cityscapes/coco-annotations/panoptic_fine_train.json",
            "overlaps_path": "panoptic_fine_train_overlap_005_fixed.npy",
        },
        "panoptic_cityscapes_fine_instanceonly_seg_train_cocostyle_overlap010": {
            "img_dir": "cityscapes/fine/train",
            "ann_file": "cityscapes/coco-annotations/panoptic_fine_train.json",
            "overlaps_path": "panoptic_fine_train_overlap_010.npy",
        },
        "panoptic_cityscapes_fine_instanceonly_seg_train_cocostyle_overlap015": {
            "img_dir": "cityscapes/fine/train",
            "ann_file": "cityscapes/coco-annotations/panoptic_fine_train.json",
            "overlaps_path": "panoptic_fine_train_overlap_015.npy",
        },
        "panoptic_cityscapes_fine_instanceonly_seg_train_cocostyle_overlap020": {
            "img_dir": "cityscapes/fine/train",
            "ann_file": "cityscapes/coco-annotations/panoptic_fine_train.json",
            "overlaps_path": "panoptic_fine_train_overlap_020.npy",
        },
        "panoptic_cityscapes_fine_instanceonly_seg_val_cocostyle_overlap005": {
            "img_dir": "cityscapes/fine/val",
            "ann_file": "cityscapes/coco-annotations/panoptic_fine_val.json",
            "overlaps_path": "panoptic_fine_val_overlap_005_fixed.npy",
        },
        "panoptic_cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/fine/val",
            "ann_file": "cityscapes/coco-annotations/panoptic_fine_val.json"
        },
        "panoptic_cityscapes_fine_instanceonly_seg_tiny_cocostyle": {
            "img_dir": "cityscapes/fine/val",
            "ann_file": "cityscapes/coco-annotations/panoptic_fine_tiny.json"
        },
        "panoptic_cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/fine/test",
            "ann_file": "cityscapes/coco-annotations/panoptic_fine_test.json"
        }
    }

    @staticmethod
    def get(name):
        if "panoptic_coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )

            if "overlaps_path" in attrs:
                args["overlaps_path"] = attrs["overlaps_path"]

            # test-dev set.
            if "test" in name:
                return dict(
                    factory="COCOTestDataset",
                    args=args,
                )
            
            return dict(
                factory="PanopticCOCODataset",
                args=args,
            )
        elif "panoptic_cityscapes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )

            if "overlaps_path" in attrs:
                args["overlaps_path"] = attrs["overlaps_path"]            

            return dict(
                factory="PanopticCOCODataset",
                args=args,
            )
        elif "stuff_coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="StuffCOCODataset",
                args=args,
            )
        elif "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
