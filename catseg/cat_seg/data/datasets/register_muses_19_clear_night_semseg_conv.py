# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

# The following code is a modified version of detectron2's Cityscapes Panoptic Segmentation Dataset Mapper
# The original code can be found at
# https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/detectron2/data/datasets/cityscapes.py

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from .muses import load_muses_semantic
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, CITYSCAPES_CATEGORIES

MUSES_CLEAR_DAY_SPLITS = [
    ("train", "muses/frame_camera/train/clear/night/", "muses/gt_semantic/train/clear/night/"),
    ("val", "muses/frame_camera/val/clear/night/", "muses/gt_semantic/val/clear/night/"),
]

def register_muses_clear_conv_night(root):
    ds_name = 'muses_conv-clear-night'
    for split, image_dir, gt_dir in MUSES_CLEAR_DAY_SPLITS:
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        full_name = f'{ds_name}_sem_seg_{split}'

        DatasetCatalog.register(
            full_name, lambda x=image_dir, y=gt_dir: load_muses_semantic(x, y)
        )

        MetadataCatalog.get(full_name).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="muses_sem_seg",
            ignore_label=255,
            thing_colors=[k["color"] for k in CITYSCAPES_CATEGORIES if k["name"] in meta["thing_classes"]],
            stuff_colors=[k["color"] for k in CITYSCAPES_CATEGORIES if k["name"] in meta["stuff_classes"]],
            **meta,
        )
        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_muses_clear_conv_night(_root)