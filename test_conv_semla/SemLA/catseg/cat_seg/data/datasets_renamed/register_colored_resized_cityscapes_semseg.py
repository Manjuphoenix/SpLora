# Copyright (c) Meta Platforms, Inc. All Rights Reserved

# The following code is a modified version of detectron2's Cityscapes Panoptic Segmentation Dataset Mapper
# The original code can be found at
# https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/detectron2/data/datasets/cityscapes.py

import os # -> original code

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes import load_cityscapes_semantic
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, CITYSCAPES_CATEGORIES
        
_RAW_CITYSCAPES_COLORED_RESIZED_SPLITS = {
    "cs-normal_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cs-normal_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
}

def register_colored_resized_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_COLORED_RESIZED_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="resized_cityscapes_sem_seg",
            ignore_label=255,
            thing_colors=[k["color"] for k in CITYSCAPES_CATEGORIES if k["name"] in meta["thing_classes"]],
            stuff_colors=[k["color"] for k in CITYSCAPES_CATEGORIES if k["name"] in meta["stuff_classes"]],
            **meta,
        )
        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_colored_resized_cityscapes(_root)

##########################################3

# import os
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets.cityscapes import load_cityscapes_semantic
# from collections import namedtuple

# # Define Label structure
# Label = namedtuple('Label', [
#     'name', 'id', 'trainId', 'category', 'catId', 'hasInstances', 'ignoreInEval', 'color'
# ])

# # ALL 34 Cityscapes classes - KEEP ORIGINAL trainIds
# ALL_CITYSCAPES_LABELS = [
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,154) ),
#     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
# ]

# # ==== CRITICAL CHANGE: Include ALL classes, reassign trainIds ====
# # Remove the filter - include everything except license plate
# ALL_CITYSCAPES_CATEGORIES = []
# new_train_id = 0

# # Sort by original ID to maintain order
# for label in sorted(ALL_CITYSCAPES_LABELS, key=lambda x: x.id):
#     ALL_CITYSCAPES_CATEGORIES.append({
#         "color": label.color,
#         "isthing": 1 if label.hasInstances else 0,
#         "id": label.id,
#         # "trainId": new_train_id,  # Sequential trainId: 0, 1, 2, ..., 33
#         "trainId": label.trainId,  # Sequential trainId: 0, 1, 2, ..., 33
#         "name": label.name,
#         "category": label.category,
#     })
#     new_train_id += 1

# _RAW_CITYSCAPES_COLORED_RESIZED_SPLITS = {
#     "cs-all34_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
#     "cs-all34_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
# }

# def register_all_cityscapes_classes(root):
#     for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_COLORED_RESIZED_SPLITS.items():
#         image_dir = os.path.join(root, image_dir)
#         gt_dir = os.path.join(root, gt_dir)
#         sem_key = key.format(task="sem_seg")
        
#         # Extract thing and stuff classes FROM ALL 34 CATEGORIES
#         thing_classes = [k["name"] for k in ALL_CITYSCAPES_CATEGORIES if k["isthing"] == 1]
#         stuff_classes = [k["name"] for k in ALL_CITYSCAPES_CATEGORIES if k["isthing"] == 0]
#         thing_colors = [k["color"] for k in ALL_CITYSCAPES_CATEGORIES if k["isthing"] == 1]
#         stuff_colors = [k["color"] for k in ALL_CITYSCAPES_CATEGORIES if k["isthing"] == 0]
        
#         DatasetCatalog.register(
#             sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
#         )
        
#         MetadataCatalog.get(sem_key).set(
#             image_dir=image_dir,
#             gt_dir=gt_dir,
#             evaluator_type="resized_cityscapes_sem_seg",
#             ignore_label=255,  # No class uses 255 now since we reassigned
#             thing_classes=thing_classes,
#             stuff_classes=stuff_classes,
#             thing_colors=thing_colors,
#             stuff_colors=stuff_colors,
#         )
        
#         print(f"✓ Registered {sem_key} with {len(ALL_CITYSCAPES_CATEGORIES)} classes")
#         print(f"  Thing classes ({len(thing_classes)}): {thing_classes}")
#         print(f"  Stuff classes ({len(stuff_classes)}): {stuff_classes}")

# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_all_cityscapes_classes(_root)

# # ```

# # **What changed:**
# # 1. **Removed this line:** `TRAINABLE_LABELS = [label for label in ALL_CITYSCAPES_LABELS if label.trainId != 255 and label.trainId != -1]` ← This was filtering to only 19 classes!
# # 2. **Added sequential trainId assignment:** Now all 34 classes get trainIds from 0-33
# # 3. Changed dataset name to `cs-all34_` to differentiate from the 19-class version

# # **Expected output:**
# # ```
# # ✓ Registered cs-all34_sem_seg_val with 34 classes
# #   Thing classes (10): ['ego vehicle', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']
# #   Stuff classes (24): ['unlabeled', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky']