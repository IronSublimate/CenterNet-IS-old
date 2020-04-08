from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import torch.utils.data as data
import numpy as np


# All classes in dataset folder and sample folder inherit from this
class BaseDataset(data.dataset):
    num_classes = 0
    default_resolution = [512, 512]
    mean = np.array([0, 0, 0],
                    dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0, 0, 0],
                   dtype=np.float32).reshape((1, 1, 3))

    def __init__(self):
        # super().__init__()
        self.data_dir = ""
        self.img_dir = ""
        self.annot_path = ""
        self.max_objs = 0
        self.class_name = []
        self._valid_ids = np.array()
        self.cat_ids = {}
        self._data_rng = 0
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self.split = ""
        self.opt = None

        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.coco = coco.COCO()
        self.images = []
        self.num_samples = 0
