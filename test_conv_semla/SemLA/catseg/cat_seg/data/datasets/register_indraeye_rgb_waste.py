import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.colormap import colormap



IE_SEM_SEG_CATEGORIES = [
    {
        "color": [0, 24.7, 100],
        "instances": True,
        "readable": "car",
        "name": "car",
        "evaluate": True,
    },
    {
        "color": [0, 24.7, 49.8],
        "instances": True,
        "readable": "person",
        "name": "person",
        "evaluate": True,
    },
    {
        "color": [0, 0, 24.7],
        "instances": True,
        "readable": "motorcycle",
        "name": "motorcycle",
        "evaluate": True,
    },
    {
        "color": [0, 24.7, 24.7],
        "instances": True,
        "readable": "truck",
        "name": "vehicles--truck",
        "evaluate": True,
    },
    {
        "color": [0, 24.7, 74.9],
        "instances": True,
        "readable": "rickshaw",
        "name": "vehicles--rickshaw",
        "evaluate": True,
    },
    {
        "color": [0, 74.9, 49.8],
        "instances": True,
        "readable": "small-truck",
        "name": "vehicles--small-truck",
        "evaluate": True,
    },
    {
        "color": [0, 49.8, 24.7],
        "instances": True,
        "readable": "bus",
        "name": "vehicles--bus",
        "evaluate": True,
    },
    {
        "color": [0, 24.7, 0],
        "instances": True,
        "readable": "bicycle",
        "name": "vehicles--bicycle",
        "evaluate": True,
    },
    {
        "color": [0, 0,  49.8],
        "instances": True,
        "readable": "cargo_trike",
        "name": "vehicles--cargo_trike",
        "evaluate": True,
    },
    {
        "color": [0, 39.2, 60.8],
        "instances": True,
        "readable": "van",
        "name": "vehicles--van",
        "evaluate": True,
    },
    {
        "color": [0, 49.8, 100],
        "instances": True,
        "readable": "backhoe_loader",
        "name": "vehicles--backhoe_loader",
        "evaluate": True,
    },
    {
        "color": [0, 49.8, 74.9],
        "instances": True,
        "readable": "tractor",
        "name": "vehicles--tractor",
        "evaluate": True,
    },
]
        
        
def _get_ie_meta():
    stuff_classes = [k["readable"] for k in IE_SEM_SEG_CATEGORIES if k["evaluate"]]
    assert len(stuff_classes) == 12

    stuff_colors = [k["color"] for k in IE_SEM_SEG_CATEGORIES if k["evaluate"]]
    assert len(stuff_colors) == 12

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret



def register_ie_semseg(root):
    ds_name = 'IE_Segmentation'
    root = os.path.join(root, "IE_Segmentation")
    meta = _get_ie_meta()
    # for name, dirname in [("train", "train"), ("val", "val")]:
    for split, image_dirname, sem_seg_dirname in [
        ('train', 'IE_eo_ir_split/eo/train/images', 'IE_eo_ir_split/eo/train/segmentation/color'),
        #('val', 'images_detectron2/val', 'annotations_detectron2/val', CLASSES),
        ('val', 'IE_eo_ir_split/eo/test/images', 'IE_eo_ir_split/eo/test/segmentation/color'),  #rgb
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
register_ie_semseg(_root)