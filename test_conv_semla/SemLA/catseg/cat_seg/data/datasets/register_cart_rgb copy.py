import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.colormap import colormap




    


CART_SEM_SEG_CATEGORIES = [
    {
        "color": [255, 36, 0],      #FF2400
        "instances": False,
        "readable": "unknown",
        "name": "unknown",
        "evaluate": False,
    },
    {
        "color": [0, 0, 0],         #000000
        "instances": False,
        "readable": "background",
        "name": "background",
        "evaluate": False,
    },
    {
        "color": [242, 216, 196],       #F2D8C4
        "instances": False,
        "readable": "bare-ground",
        "name": "bare-ground",
        "evaluate": True,
    },
    {
        "color": [89, 70, 54],          #594636
        "instances": False,
        "readable": "boulders",
        "name": "boulders",
        "evaluate": True,
    },
    {
        "color": [166, 166, 166],       #A6A6A6
        "instances": True,
        "readable": "human-made-structures",
        "name": "human-made-structures",
        "evaluate": True,
    },
    {
        "color": [82, 89, 90],          #52595A
        "instances": False,
        "readable": "road",
        "name": "road",
        "evaluate": True,
    },
    {
        "color": [155, 230, 0],         #9BE600
        "instances": True,
        "readable": "shrubs",
        "name": "shrubs",
        "evaluate": True,
    },
    {
        "color": [0, 138, 53],          #008A35
        "instances": True,
        "readable": "trees",
        "name": "trees",
        "evaluate": True,
    },
    {
        "color": [0, 216, 245],             #00D8F5
        "instances": False,
        "readable": "sky",
        "name": "sky",
        "evaluate": True,
    },
    {
        "color": [13, 127, 252],        #0D7FFC
        "instances": False,
        "readable": "water",
        "name": "water",
        "evaluate": True,
    },
    {
        "color": [255, 249, 0],         #FFF900
        "instances": True,
        "readable": "vehicles",
        "name": "vehicles",
        "evaluate": True,
    },
    {
        "color": [254, 0, 170],         #FE00AA
        "instances": True,
        "readable": "person",
        "name": "person",
        "evaluate": True,
    },
]
        
        
def _get_ie_meta():
    stuff_classes = [k["readable"] for k in CART_SEM_SEG_CATEGORIES if k["evaluate"]]

    print("-_____---____--____--_____-", len(stuff_classes), "___--_____---____--_____-")
    assert len(stuff_classes) == 10

    stuff_colors = [k["color"] for k in CART_SEM_SEG_CATEGORIES if k["evaluate"]]
    assert len(stuff_colors) == 10

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret



def register_cartr_rgb(root):
    ds_name = 'cartr'
    root = os.path.join(root, "cart")
    meta = _get_ie_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    for split, image_dirname, sem_seg_dirname in [
        ('train', 'train/color', 'train/annotations'),
        #('val', 'images_detectron2/val', 'annotations_detectron2/val', CLASSES),
        ('val', 'val/color', 'val/annotations'),  #rgb
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
                y, x, gt_ext='png', image_ext='jpg'
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
register_cartr_rgb(_root)