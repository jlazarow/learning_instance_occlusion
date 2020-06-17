TASK:
  KIND: "Panoptic"
MODEL:
  META_ARCHITECTURE: "PanopticFPN"
  WEIGHT: "catalog://ImageNetPretrained/MSRA/R-50"
  BACKBONE:
    CONV_BODY: "R-50-FPN"
    OUT_CHANNELS: 256
  RPN:
    USE_FPN: True
    ANCHOR_STRIDE: (4, 8, 16, 32, 64)
    PRE_NMS_TOP_N_TRAIN: 2000
    PRE_NMS_TOP_N_TEST: 1000
    POST_NMS_TOP_N_TEST: 1000
    FPN_POST_NMS_TOP_N_TEST: 1000
    USE_SEMANTIC_FEATURES: True
    RPN_HEAD: "SingleConvRPNHead"
  ROI_HEADS:
    USE_FPN: True
    USE_SEMANTIC_FEATURES: True
  ROI_BOX_HEAD:
    POOLER_RESOLUTION: 7
    POOLER_SCALES: (0.25, 0.125, 0.0625, 0.03125)
    POOLER_SAMPLING_RATIO: 2
    FEATURE_EXTRACTOR: "FPN2MLPFeatureExtractor"
    PREDICTOR: "FPNPredictor"
  ROI_MASK_HEAD:
    POOLER_SCALES: (0.25, 0.125, 0.0625, 0.03125)
    FEATURE_EXTRACTOR: "MaskRCNNFPNFeatureExtractor"
    PREDICTOR: "MaskRCNNC4Predictor"
    POOLER_RESOLUTION: 14
    POOLER_SAMPLING_RATIO: 2
    RESOLUTION: 28
    SHARE_BOX_FEATURE_EXTRACTOR: False
  SEMANTIC:
    USE_FPN: True
    SEMANTIC_HEAD: "UpsampleConvSumCombineScales"
    CONV_HEAD_DIM: 128
    UPSAMPLE_MODULE: "One3x3ReLU"
    POOLER_SCALES: (0.25, 0.125, 0.0625, 0.03125)
    COLLAPSE_THING_ONTOLOGY: False
    UPSAMPLE_METHOD: "bilinear"
  PANOPTIC:
    INSTANCE_WEIGHT: 1.0
    SEMANTIC_WEIGHT: 0.5
  MASK_ON: True
DATASETS:
  TRAIN: ("panoptic_coco_2017_train",)
  TEST: ("panoptic_coco_2017_val",)
DATALOADER:
  SIZE_DIVISIBILITY: 32
SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  WEIGHT_DECAY: 0.0001
  STEPS: (120000, 160000)
  MAX_ITER: 180000
TEST:
  # nearest: /mnt/cube/jlazarow/experiments/02042019_112833/inference/panoptic_coco_2017_val/predictions.pth
  # bilinear: /mnt/cube/jlazarow/experiments/02082019_110113/inference/panoptic_coco_2017_val/predictions.pth
  PREDICTION_PATHS: []
TEMPORARY_DIR: "/media/data/jlazarow"
NAME: "prime_rpn_roi"
DESCRIPTION: "priming the RPN and ROI heads with residual semantic features 4GPU"
