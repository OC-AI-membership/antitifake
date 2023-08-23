# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro
"""
from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4

# Cudnn parameters
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Model parameters
_C.MODEL = CN()
_C.MODEL.NAME = 'detconfcmx'
_C.MODEL.PRETRAINED = ''
_C.MODEL.MODS = ('RGB','NP++')
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.EXTRA.DETECTION = None
_C.MODEL.EXTRA.CONF = False

# Dataset parameters
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.TRAIN = []
_C.DATASET.VALID = []
_C.DATASET.NUM_CLASSES = 2

# Testing parameters
_C.TEST = CN()
_C.TEST.MODEL_FILE = ''


# def update_config(cfg, args):
#     cfg.defrost()
#     cfg.merge_from_file(f'trufor.yaml')
#     if args.opts:
#         cfg.merge_from_list(args.opts)
#     cfg.freeze()

import os

def update_config(cfg, args):
    cfg.defrost()
    # 현재 스크립트 파일의 디렉토리를 가져옵니다.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # trufor.yaml 파일의 절대 경로를 생성합니다.
    yaml_file_path = os.path.join(current_dir, 'trufor.yaml')
    # 절대 경로를 사용하여 파일을 읽습니다.
    cfg.merge_from_file(yaml_file_path)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()