# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# import torch
# from .nms import nms
# from .roi_align import ROIAlign
# from .roi_align import roi_align
# from .roi_pool import ROIPool
# from .roi_pool import roi_pool
from torchvision.ops import nms
from torchvision.ops import RoIAlign as ROIAlign
from torchvision.ops import roi_align
from torchvision.ops import RoIPool as ROIPool
from torchvision.ops import roi_pool

__all__ = ["nms", "roi_align", "ROIAlign", "roi_pool", "ROIPool"]
