# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved


import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

MSRS_SEM_SEG_CATEGORIES = [
    {
        "color": [0, 0, 0],
        "instances": False,
        "readable": "unlabelled",
        "name": "background--unlabelled",
        "evaluate": True,
    },
    {
        "color": [64, 0, 128],
        "instances": True,
        "readable": "car",
        "name": "vehicles--car",
        "evaluate": True,
    },
    {
        "color": [64, 64, 0],
        "instances": True,
        "readable": "person",
        "name": "person",
        "evaluate": True,
    },
    {
        "color": [0, 128, 192],
        "instances": True,
        "readable": "bike",
        "name": "vehicles--bike",
        "evaluate": True,
    },
    {
        "color": [0, 0, 192],
        "instances": False,
        "readable": "curve",
        "name": "construction--curve",
        "evaluate": True,
    },
    {
        "color": [128, 128, 0],
        "instances": False,
        "readable": "car_stop",
        "name": "construction--car-stop",
        "evaluate": True,
    },
    {
        "color": [64, 64, 128],
        "instances": False,
        "readable": "guardrail",
        "name": "construction--guardrail",
        "evaluate": True,
    },
    {
        "color": [192, 128, 128],
        "instances": False,
        "readable": "color_cone",
        "name": "construction--color-cone",
        "evaluate": True,
    },
    {
        "color": [192, 64, 0],
        "instances": False,
        "readable": "bump",
        "name": "construction--bump",
        "evaluate": True,
    },
]



def _get_msrs_meta():
    stuff_classes = [k["readable"] for k in MSRS_SEM_SEG_CATEGORIES if k["evaluate"]]
    assert len(stuff_classes) == 9

    stuff_colors = [k["color"] for k in MSRS_SEM_SEG_CATEGORIES if k["evaluate"]]
    assert len(stuff_colors) == 9

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_msrs_semseg(root):
    ds_name = 'msrs'
    root = os.path.join(root, "msrs")
    meta = _get_msrs_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    for split, image_dirname, sem_seg_dirname in [
        ('train', 'train/vi', 'train/labels'),
        #('val', 'images_detectron2/val', 'annotations_detectron2/val', CLASSES),
        ('val', 'test/vi', 'test/labels'),  #rgb
        #('test', 'ir/test', 'annotations_detectron2/test', CLASSES),  #ir
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        # print("_------____---____---____---___---_____---___----", image_dir, "*********8888****88*****888*****888*****")
        # print(gt_dir)
        # name = f"mv_sem_seg_{name}"
        name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='png', image_ext='png'
            ),
        )

        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=0,  # different from other datasets, Mapillary Vistas sets ignore_label to 65
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_msrs_semseg(_root)