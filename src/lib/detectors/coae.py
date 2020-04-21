from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector


class CoAEDetector(BaseDetector):
    def __init__(self, opt):
        super().__init__(opt)
