"""
Experimental script to generate embeddings using CATSeg's CLIP model.
Updated to take dataset and source_domains_file separately.
"""
from pathlib import Path
import os
import yaml
import argparse
import sys
import torch
from torch.nn import functional as F
from detectron2.config import get_cfg, CfgNode as CN
from domain_orchestrator.utils import get_domain_args

DETECTRON2_DATASET_PATH = os.getenv("DETECTRON2_DATASETS")


def add_minimal_catseg_config(cfg):
    """
    Add minimal CATSeg config structure without importing the config module.
    This avoids triggering dataset registration.
    """
    cfg.MODEL.MASK_FORMER = CN()
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    if not hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
        cfg.MODEL.SEM_SEG_HEAD = CN()

    # Define all keys from YAML to avoid KeyError
    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED = "ViT-B/16"
    cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH = 0
    cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH = 0
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = ""
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = ""
    cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE = "attention"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 150

    # CLIP pixel normalization
    cfg.MODEL.CLIP_PIXEL_MEAN = [122.7709383, 116.7460125, 104.09373615]
    cfg.MODEL.CLIP_PIXEL_STD = [68.5005327, 66.6321579, 70.3231630]

    # INPUT defaults
    cfg.INPUT.CROP = CN()
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"

    # SOLVER defaults
    cfg.SOLVER.OPTIMIZER = "ADAMW"


class MinimalCATSegWrapper:
    """
    Minimal wrapper that mimics CATSeg model interface for embedding extraction.
    Loads only CLIP model without triggering dataset registration.
    """
    def __init__(self, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_pretrained = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        self.clip_pixel_mean = torch.Tensor(cfg.MODEL.CLIP_PIXEL_MEAN).view(-1, 1, 1).to(self.device)
        self.clip_pixel_std = torch.Tensor(cfg.MODEL.CLIP_PIXEL_STD).view(-1, 1, 1).to(self.device)

        # Add catseg path
        catseg_path = Path(__file__).parent / "catseg"
        if str(catseg_path) not in sys.path:
            sys.path.insert(0, str(catseg_path))

        # Import CLIP module using importlib to avoid triggering cat_seg.__init__.py
        import importlib.util

        third_party_path = catseg_path / "cat_seg" / "third_party"

        # Load model_vpt
        model_vpt_path = third_party_path / "model_vpt.py"
        spec_vpt = importlib.util.spec_from_file_location("model_vpt", model_vpt_path)
        model_vpt_module = importlib.util.module_from_spec(spec_vpt)
        spec_vpt.loader.exec_module(model_vpt_module)

        # Load simple_tokenizer
        tokenizer_path = third_party_path / "simple_tokenizer.py"
        spec_tokenizer = importlib.util.spec_from_file_location("simple_tokenizer", tokenizer_path)
        simple_tokenizer_module = importlib.util.module_from_spec(spec_tokenizer)
        spec_tokenizer.loader.exec_module(simple_tokenizer_module)

        # Load clip.py
        clip_path = third_party_path / "clip.py"
        spec_clip = importlib.util.spec_from_file_location("clip", clip_path)
        clip_module = importlib.util.module_from_spec(spec_clip)

        # Patch relative imports
        class MockModule:
            def __init__(self, module):
                self._module = module
                self.build_model = module.build_model

        original_modules = sys.modules.copy()
        sys.modules['cat_seg.third_party.model_vpt'] = MockModule(model_vpt_module)
        sys.modules['cat_seg.third_party.simple_tokenizer'] = simple_tokenizer_module

        try:
            clip_module.__package__ = "cat_seg.third_party"
            spec_clip.loader.exec_module(clip_module)
        finally:
            for key in list(sys.modules.keys()):
                if key.startswith('cat_seg.third_party') and key not in original_modules:
                    del sys.modules[key]

        clip = clip_module
        import open_clip

        prompt_depth = getattr(cfg.MODEL.SEM_SEG_HEAD, 'PROMPT_DEPTH', 0)
        prompt_length = getattr(cfg.MODEL.SEM_SEG_HEAD, 'PROMPT_LENGTH', 0)

        if self.clip_pretrained in ["ViT-G", "ViT-H"]:
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if self.clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrain, device=self.device, force_image_size=336)
        else:
            clip_model, _ = clip.load(self.clip_pretrained, device=self.device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)

        self.clip_model = clip_model.float().eval()

        # Mock sem_seg_head.predictor
        class MockPredictor:
            def __init__(self, clip_model, clip_pretrained):
                self.clip_model = clip_model
                self.clip_pretrained = clip_pretrained

        class MockSemSegHead:
            def __init__(self, predictor):
                self.predictor = predictor

        self.sem_seg_head = MockSemSegHead(MockPredictor(self.clip_model, self.clip_pretrained))

    def eval(self):
        self.clip_model.eval()
        return self


def load_catseg_clip_from_config(args):
    """
    Load CLIP model from config without triggering dataset registration.
    Reads config directly from YAML file.
    """
    print("Loading CATSeg CLIP model...")
    cfg = get_cfg()
    add_minimal_catseg_config(cfg)

    config_path = Path(args.config_file)
    print(f"Using config: {config_path}")
    cfg.merge_from_file(str(config_path))

    return MinimalCATSegWrapper(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to CATSeg YAML config")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., a150)")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training images")
    parser.add_argument("--lora_library_path", type=str, required=True, help="Where to store LoRA statistics")
    args = parser.parse_args()

    print(f"Generating embeddings for dataset: {args.dataset}")
    print(f"Train path: {args.train_path}")

    from domain_orchestrator import embedding_catseg

    model = load_catseg_clip_from_config(args)
    embedding_manager = embedding_catseg.EmbeddingManager(catseg_model=model)

    domain_path = Path(args.lora_library_path) / args.dataset
    embedding_manager.calculate_statistics(
        domain_name=args.dataset,
        domain_path=domain_path,
        train_path=Path(args.train_path)
    )

    print("Finished generating embeddings.")
