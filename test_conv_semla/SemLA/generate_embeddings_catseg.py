"""
Experimental script to generate embeddings using CATSeg's CLIP model.
Based on the logic in cat_seg_model_embeddings.py
"""
from pathlib import Path
import os
import yaml
import argparse
import sys
import torch
from torch.nn import functional as F
from detectron2.structures import ImageList
from detectron2.config import get_cfg, CfgNode as CN
from domain_orchestrator.utils import get_domain_args

DETECTRON2_DATASET_PATH = os.getenv("DETECTRON2_DATASETS")


def add_minimal_catseg_config(cfg):
    """
    Add minimal CATSeg config structure without importing the config module.
    This avoids triggering dataset registration.
    """
    # Add MASK_FORMER config
    cfg.MODEL.MASK_FORMER = CN()
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32
    
    # Add SEM_SEG_HEAD config structure (minimal, will be overridden by YAML)
    if not hasattr(cfg.MODEL, 'SEM_SEG_HEAD'):
        cfg.MODEL.SEM_SEG_HEAD = CN()
    
    # Set minimal defaults that might be needed
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = ""
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = ""
    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED = "ViT-B/16"
    cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH = 0
    cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH = 0
    cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE = "attention"
    
    # Add CLIP pixel mean/std
    cfg.MODEL.CLIP_PIXEL_MEAN = [122.7709383, 116.7460125, 104.09373615]
    cfg.MODEL.CLIP_PIXEL_STD = [68.5005327, 66.6321579, 70.3231630]
    
    # Add other SEM_SEG_HEAD fields that might be in the YAML
    cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM = 128
    cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM = 128
    cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS = [64, 32]
    cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS = [256, 128]
    cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS = [32, 16]
    cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS = 4
    cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS = 4
    cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS = 128
    cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES = [6, 6]
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION = [24, 24]
    cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES = 12
    cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE = "linear"
    
    # Add PROMPT_ENSEMBLE config
    cfg.MODEL.PROMPT_ENSEMBLE = False
    cfg.MODEL.PROMPT_ENSEMBLE_TYPE = "single"
    
    # Add SOLVER config keys
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.CLIP_MULTIPLIER = 0.01
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    
    # Add INPUT config structure
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    cfg.INPUT.SIZE_DIVISIBILITY = -1
    
    # Add TEST config
    cfg.TEST.SLIDING_WINDOW = False


class MinimalCATSegWrapper:
    """
    Minimal wrapper that mimics CATSeg model interface for embedding extraction.
    Loads only CLIP model without triggering dataset registration.
    """
    def __init__(self, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get config parameters
        self.clip_pretrained = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        self.clip_pixel_mean = torch.Tensor(cfg.MODEL.CLIP_PIXEL_MEAN).view(-1, 1, 1).to(self.device)
        self.clip_pixel_std = torch.Tensor(cfg.MODEL.CLIP_PIXEL_STD).view(-1, 1, 1).to(self.device)
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        self.clip_resolution = (384, 384) if self.clip_pretrained == "ViT-B/16" else (336, 336)
        
        # Load CLIP model directly (matching CATSegPredictor logic)
        # Add catseg to path
        catseg_path = Path(__file__).parent / "catseg"
        if str(catseg_path) not in sys.path:
            sys.path.insert(0, str(catseg_path))
        
        # Import CLIP module using importlib with manual dependency injection
        # This avoids triggering cat_seg.__init__.py
        import importlib.util
        
        third_party_path = catseg_path / "cat_seg" / "third_party"
        
        # Load model_vpt first
        model_vpt_path = third_party_path / "model_vpt.py"
        spec_vpt = importlib.util.spec_from_file_location("model_vpt", model_vpt_path)
        model_vpt_module = importlib.util.module_from_spec(spec_vpt)
        spec_vpt.loader.exec_module(model_vpt_module)
        
        # Load simple_tokenizer
        tokenizer_path = third_party_path / "simple_tokenizer.py"
        spec_tokenizer = importlib.util.spec_from_file_location("simple_tokenizer", tokenizer_path)
        simple_tokenizer_module = importlib.util.module_from_spec(spec_tokenizer)
        spec_tokenizer.loader.exec_module(simple_tokenizer_module)
        
        # Now load clip.py and inject dependencies before execution
        clip_path = third_party_path / "clip.py"
        spec_clip = importlib.util.spec_from_file_location("clip", clip_path)
        clip_module = importlib.util.module_from_spec(spec_clip)
        
        # Create a mock module for the relative import
        # This prevents the relative import from triggering cat_seg.__init__
        class MockModule:
            def __init__(self, module):
                self._module = module
                # Make build_model available as if it was imported
                self.build_model = module.build_model
        
        # Inject dependencies into clip_module's namespace before execution
        # This way when clip.py does "from .model_vpt import build_model", 
        # we can intercept it by patching sys.modules
        original_modules = sys.modules.copy()
        
        # Create fake module entries to intercept relative imports
        fake_model_vpt = MockModule(model_vpt_module)
        fake_simple_tokenizer = simple_tokenizer_module
        
        # Temporarily add to sys.modules to intercept relative imports
        sys.modules['cat_seg.third_party.model_vpt'] = fake_model_vpt
        sys.modules['cat_seg.third_party.simple_tokenizer'] = fake_simple_tokenizer
        
        try:
            # Set package so relative imports work
            clip_module.__package__ = "cat_seg.third_party"
            spec_clip.loader.exec_module(clip_module)
        finally:
            # Restore original modules
            for key in list(sys.modules.keys()):
                if key.startswith('cat_seg.third_party') and key not in original_modules:
                    del sys.modules[key]
        
        clip = clip_module
        import open_clip
        
        prompt_depth = getattr(cfg.MODEL.SEM_SEG_HEAD, 'PROMPT_DEPTH', 0)
        prompt_length = getattr(cfg.MODEL.SEM_SEG_HEAD, 'PROMPT_LENGTH', 0)
        
        if self.clip_pretrained == "ViT-G" or self.clip_pretrained == "ViT-H":
            # For OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if self.clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, _ = open_clip.create_model_and_transforms(
                name, 
                pretrained=pretrain, 
                device=self.device, 
                force_image_size=336,
            )
        else:
            # For OpenAI models (matching CATSegPredictor line 76)
            clip_model, _ = clip.load(
                self.clip_pretrained, 
                device=self.device, 
                jit=False, 
                prompt_depth=prompt_depth, 
                prompt_length=prompt_length
            )
        
        self.clip_model = clip_model.float().eval()
        
        # Create a mock sem_seg_head.predictor structure for compatibility
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
    print("Loading config and CLIP model (avoiding dataset registration)...")
    
    # Setup basic config
    cfg = get_cfg()
    
    # Add minimal CATSeg config structure first
    add_minimal_catseg_config(cfg)
    
    # Get catseg directory path
    catseg_path = Path(__file__).parent / "catseg"
    
    # Read config from YAML file if available
    config_file = None
    if hasattr(args, 'config_file') and args.config_file:
        config_file = args.config_file
    
    # Resolve config file path relative to catseg directory
    if config_file:
        # If it's already an absolute path, use it as is
        if Path(config_file).is_absolute():
            config_path = Path(config_file)
        else:
            # Otherwise, resolve relative to catseg directory
            config_path = catseg_path / config_file
        
        if config_path.exists():
            print(f"Loading config from: {config_path}")
            cfg.merge_from_file(str(config_path))
        else:
            print(f"Warning: Config file not found at {config_path}, using defaults")
    
    # Merge any additional opts
    if hasattr(args, 'opts'):
        cfg.merge_from_list(args.opts)
    
    # Create minimal wrapper
    model_wrapper = MinimalCATSegWrapper(cfg)
    
    print("CLIP model loaded successfully.")
    return model_wrapper


if __name__ == "__main__":
    # Argparse
    parser = argparse.ArgumentParser()
    # Path to the yaml file that contains the paths to the domains training data
    parser.add_argument("--source_domains_file", type=str, required=True)
    # Path to the lora library where the statistics will be stored
    parser.add_argument("--lora_library_path", type=str, required=True)

    # Parse arguments
    args = parser.parse_args()
    source_domains_file = Path(args.source_domains_file)
    lora_library_path = Path(args.lora_library_path)

    with open(source_domains_file, "r") as f:
        source_domains = yaml.safe_load(f)

    embedding_manager = None
    catseg_model = None

    print("Generating embeddings for all source domains using CATSeg CLIP model...")
    
    for domain_name in source_domains:

        domain_args = get_domain_args(domain_name, "train", get_cofing_only=True)
        train_dataset_path = Path(domain_args.train_dataset_path)
        print(f"Domain: {domain_name}")
        print(f"Train dataset path: {train_dataset_path}")

        assert train_dataset_path.exists(), f"Path to training dataset {train_dataset_path} does not exist!"

        if embedding_manager is None:
            from domain_orchestrator import embedding_catseg
            
            # Load CLIP model directly from config (avoids dataset registration)
            print("Loading CLIP model from config...")
            catseg_model = load_catseg_clip_from_config(domain_args)
            
            # Create EmbeddingManager with the wrapper
            embedding_manager = embedding_catseg.EmbeddingManager(catseg_model=catseg_model)
            print("CATSeg embedding manager initialized.")

        domain_path = lora_library_path / Path(domain_name)

        embedding_manager.calculate_statistics(
            domain_name=domain_name,
            domain_path=domain_path,
            train_path=train_dataset_path,
        )

    print("Finished generating embeddings for all domains")
