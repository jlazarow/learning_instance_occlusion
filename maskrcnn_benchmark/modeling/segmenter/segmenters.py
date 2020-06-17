from .semantic_fpn import SemanticFPN

_SEGMENTATION_META_ARCHITECTURES = {
    "SemanticFPN": SemanticFPN
}

def build_segmentation_model(cfg):
    meta_arch = _SEGMENTATION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
