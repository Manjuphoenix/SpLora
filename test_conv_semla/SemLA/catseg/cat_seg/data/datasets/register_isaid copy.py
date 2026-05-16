import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# --- iSAID Dataset Label Definition (Modified from your provided information) ---
# We use the 'trainId' as the ID for Detectron2's contiguous labeling and
# ensure we include 'isthing' and 'color' which are necessary for segmentation tasks.

ISAID_CATEGORIES = [
    # {'name': 'unlabeled', 'id': 0, 'trainId': 255, 'category': 'void', 'isthing': 0, 'color': (0, 0, 0), 'ignoreInEval': True},
    {'name': 'ship', 'id': 1, 'trainId': 0, 'category': 'transport', 'isthing': 1, 'color': [0, 0, 63]},
    {'name': 'storage_tank', 'id': 2, 'trainId': 1, 'category': 'transport', 'isthing': 1, 'color': [0, 63, 63]},
    {'name': 'baseball_diamond', 'id': 3, 'trainId': 2, 'category': 'land', 'isthing': 1, 'color': [0, 63, 0]},
    {'name': 'tennis_court', 'id': 4, 'trainId': 3, 'category': 'land', 'isthing': 1, 'color': [0, 63, 127]},
    {'name': 'basketball_court', 'id': 5, 'trainId': 4, 'category': 'land', 'isthing': 1, 'color': [0, 63, 191]},
    {'name': 'Ground_Track_Field', 'id': 6, 'trainId': 5, 'category': 'land', 'isthing': 1, 'color': [0, 63, 255]},
    {'name': 'Bridge', 'id': 7, 'trainId': 6, 'category': 'land', 'isthing': 1, 'color': [0, 127, 63]},
    {'name': 'Large_Vehicle', 'id': 8, 'trainId': 7, 'category': 'transport', 'isthing': 1, 'color': [0, 127, 127]},
    {'name': 'Small_Vehicle', 'id': 9, 'trainId': 8, 'category': 'transport', 'isthing': 1, 'color': [0, 0, 127]},
    {'name': 'Helicopter', 'id': 10, 'trainId': 9, 'category': 'transport', 'isthing': 1, 'color': [0, 0, 191]},
    {'name': 'Swimming_pool', 'id': 11, 'trainId': 10, 'category': 'land', 'isthing': 1, 'color': [0, 0, 255]},
    {'name': 'Roundabout', 'id': 12, 'trainId': 11, 'category': 'land', 'isthing': 1, 'color': [0, 191, 127]},
    {'name': 'Soccer_ball_field', 'id': 13, 'trainId': 12, 'category': 'land', 'isthing': 1, 'color': [0, 127, 191]},
    {'name': 'plane', 'id': 14, 'trainId': 13, 'category': 'transport', 'isthing': 1, 'color': [0, 127, 255]},
    {'name': 'Harbor', 'id': 15, 'trainId': 14, 'category': 'transport', 'isthing': 1, 'color': [0, 100, 155]},
    # Note: If the iSAID dataset includes a 'stuff' category (like background, water, road, etc.)
    # those should be added with 'isthing': 0. The categories above are primarily 'things' (objects).
]


# def _get_isaid_meta():
#     """
#     Constructs the metadata dictionary for the iSAID dataset.
#     This replaces the _get_coco_stuff_meta function.
#     """
#     # Use 'trainId' for contiguous ID (0 to N-1) for training
#     train_ids = [k["trainId"] for k in ISAID_CATEGORIES]
    
#     # Check for contiguous and unique IDs starting from 0
#     expected_ids = list(range(len(ISAID_CATEGORIES)))
#     if sorted(train_ids) != expected_ids:
#         print(f"Warning: iSAID trainIds {train_ids} are not contiguous from 0 to {len(ISAID_CATEGORIES) - 1}. Check your labels.")

#     # Create map from the original 'id' to the contiguous 'trainId'
#     isaid_dataset_id_to_contiguous_id = {k["id"]: k["trainId"] for k in ISAID_CATEGORIES}
    
#     # Class names, ordered by contiguous ID
#     contiguous_classes = sorted(ISAID_CATEGORIES, key=lambda x: x['trainId'])
#     isaid_classes = [k["name"] for k in contiguous_classes]

#     # Handle thing/stuff split
#     thing_classes = [k["name"] for k in contiguous_classes if k.get("isthing", 1) == 1]
#     stuff_classes = [k["name"] for k in contiguous_classes if k.get("isthing", 1) == 0]
    
#     # Background/unlabeled/void label is typically handled by the ignore_label (255) in load_sem_seg

#     ret = {
#         "stuff_dataset_id_to_contiguous_id": isaid_dataset_id_to_contiguous_id,
#         "stuff_classes": isaid_classes, # Detectron2 often uses 'stuff_classes' to mean all classes.
#         "thing_classes": thing_classes, # For instance segmentation or panoptic segmentation
#         "stuff_classes_with_colors": ISAID_CATEGORIES,
#     }
#     return ret



def _get_isaid_meta():
    stuff_ids = [k["id"] for k in ISAID_CATEGORIES]
    

    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in ISAID_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret



# def register_all_isaid(root):
#     """
#     Registers the iSAID dataset for semantic segmentation.
#     This replaces the register_all_coco_stuff_10k function.
#     """
#     # Assuming your iSAID data structure is similar to the COCO-Stuff structure
#     # You will likely need to adjust 'isaid' to match your actual directory name
#     root = os.path.join(root, "isaid") # <-- **CHANGE THIS TO YOUR DATASET DIRECTORY NAME**
#     meta = _get_isaid_meta()
    
#     # Define your splits (train and val) and their relative directories
#     for name, image_dirname, sem_seg_dirname in [
#         ("train", "train/images", "train/annotations/Semantic_masks/"), # <-- **ADJUST THESE PATHS**
#         ("val", "val/images", "val/annotations/Semantic_masks/"),       # <-- **TO YOUR ISAID PATHS**
#     ]:

#         image_dir = os.path.join(root, image_dirname)
#         gt_dir = os.path.join(root, sem_seg_dirname)
        
#         # New dataset name: 'isaid_sem_seg_{split}'
#         dataset_name = f"isaid_sem_seg_{name}"
        
#         # Register the dataset
#         DatasetCatalog.register(
#             dataset_name, 
#             # Assuming your segmentation masks are PNGs and images are JPGs
#             lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
#         )
        
#         # Register the metadata
#         MetadataCatalog.get(dataset_name).set(
#             image_root=image_dir,
#             sem_seg_root=gt_dir,
#             evaluator_type="sem_seg",
#             ignore_label=255, # Matches the 'unlabeled' trainId
#             **meta,
#         )

# # --- Execution ---
# # Get the root path from environment variable or default to "datasets"
# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_all_isaid(_root)



def register_all_isaid(root):
    root = os.path.join(root, "isaid")
    meta = _get_isaid_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "train/images", "train/annotations/Semantic_masks/"),
        ("val", "val/images", "val/annotations/Semantic_masks/"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        # name = f"coco_2017_{name}_stuff_all_sem_seg"
        name = f"isaid_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_isaid(_root)