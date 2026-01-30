import time
from typing import Union, Dict, Callable
from typing import Any, Literal, Mapping
from pathlib import Path
from argparse import Namespace
from dataclasses import dataclass
import logging
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
import os

import torch
from torch import nn
import torch.nn.functional as F
from typing import List

import peft
from PIL import Image

import tqdm

from .embedding import EmbeddingManager

logging.disable()

torch.set_float32_matmul_precision("high")

from .utils import custom_domain_args, get_domain_args, benchmark_catseg, load_catseg_model
from typing import Tuple, Dict
from typing import Optional


# def softmax(x: list[float], softmax_temperature) -> np.ndarray:
def softmax(x: List[float], softmax_temperature) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    
    # Add error handling for division by zero
    if softmax_temperature == 0:
        softmax_temperature = 1e-6
    exp_x = np.exp(np.divide(x, softmax_temperature))
    return exp_x / np.sum(exp_x, axis=0)


@dataclass
class Domain:
    """A simple Domain class with a name attribute."""
    name: str
    args: Namespace
    train_dataset_path: Path
    train_average_embedding: npt.NDArray
    data_loader: Any
    evaluator: Any
    lora_path: Path = None


class DomainObserver:
    """Observer class that holds and manages a collection of Domains."""

    def __init__(
        self,
    ) -> None:
        self.domain_prototypes = {}

    def add_domain_prototypes(
        self,
        domain: Domain,
        average_embedding: npt.NDArray,
    ) -> None:
        """
        Add the average embedding of a domain to the observer.
        """
        self.domain_prototypes.update({domain.name: average_embedding})

    def calculate_similarity_to_domains(
        self, 
        embedding: npt.NDArray, 
        # domains: list[Domain],
        domains: List[Domain],
        similarity_measure: Callable[[npt.NDArray, npt.NDArray], np.float64],
        sort_descending: bool = True
    ) -> Dict[str, float]:
        """
        Calculate the similarity between the target embedding and the domain prototypes.
        """
        similarities = []

        for domain in domains:
            prot = self.domain_prototypes[domain.name]
            similarity = similarity_measure(embedding, prot)
            similarities.append([domain.name, similarity])

        # sort similarities from lowest to highest
        similarities_dict = dict(sorted(similarities, key=lambda x: x[1], reverse=sort_descending))
        return similarities_dict


class DomainOrchestrator:
    def __init__(
        self,
        # domains: list[str],
        domains: List[str],
        lora_db_path: Union[str, Path] = "loradb/",
        embedding_manager: EmbeddingManager = EmbeddingManager(),
    ) -> None:
        
        # TODO: Currently, to use catseg for experiments, we need to change the directory to the catseg directory
        # This can be fixed by refactoring the catseg repo
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        catseg_path = os.path.join(parent_dir, "catseg") # TODO: This is a hardcoded path, it should be a parameter
        
        print(f"Changing directory to '{catseg_path}' ...")

        try:
            os.chdir(catseg_path)
        except FileNotFoundError:
            print(f"Error: The specified path '{catseg_path}' does not exist.")
            exit(1)
        except PermissionError:
            print(f"Error: Insufficient permissions to access '{catseg_path}'.")
            exit(1)
        except Exception as e:
            print(f"Unexpected error while changing directory: {e}")
            exit(1)

        self.lora_db_path: Path = Path(lora_db_path)

        self.embedding_manager = embedding_manager

        self.current_model = None

        self.observer: DomainObserver = DomainObserver()

        print("Adding source domains ...")
        
        self._source_domains: Mapping[str, Domain] = self._add_domains(
            domains, split="train"
        )
        print("Source domains added. \n\n")

        print("Adding target domains ...")
        self._target_domains: Mapping[str, Domain] = self._add_domains(
            domains, split="val"
        )
        print("Target domains added. \n\n")

        self._setup_observer()


    def _benchmark_on_current_target_domain(self, name: str, target_domain: Domain) -> Any:
        print(
            f"Benchmarking {name} on the domain {target_domain.name} ...\n"
        )
        res = benchmark_catseg(self.current_model, target_domain.args)
        return res

    def _set_current_target_domain(
        self,
        target_domain: Domain,
    ) -> None:
        """
        Set the current target domain to the specified domain.
        """
        # We need to load all adapters each time the target domain changes because the same
        # config can't be used across datasets and PEFT does not allow us to change the base
        # model and keep the loaded adapters

        print(f"Setting current target domain to {target_domain.name}.\n")

        self.current_model = None  # This will ensure that the current PEFT model will be initialized using base model with new config
        self._load_adapters(target_domain)

    def _load_adapters(self, target_domain: Domain) -> None:
        """
        Load all adapters for the source domains.
        """
        for source_domain in self._source_domains.values():
            lora_path = source_domain.lora_path
            assert lora_path.exists(), lora_path
            print(f"Loading LoRA: '{lora_path}' ...")
            self._load_lora(target_domain, source_domain, lora_path)
            print(f"LoRA: '{lora_path}' loaded\n\n")

    def _load_lora(self, target_domain: Domain, source_domain: Domain, lora_path: Path) -> None:
        """
        Load the LoRA adapter for the specified domain.
        """
        if self.current_model is None:
            # Wrap the model in PeftModel class the first time an adapter is loaded
            # The base model should be loaded with target domain config to avoid label space mismatch
            self.current_model = peft.PeftModel.from_pretrained(
                load_catseg_model(target_domain.args), lora_path, source_domain.name
            )
        else:
            self.current_model.load_adapter(lora_path, source_domain.name)

    def _add_domains(
        self,
        source_domain_names: List[str],
        split: Literal["train", "val"],
    ) -> Dict[str, Domain]:
        """
        Add the specified domains to the orchestrator.
        """

        source_domains = {}

        for source_domain_name in source_domain_names:
            args, evaluator, data_loader = get_domain_args(source_domain_name, split=split)
            source_domains.update(
                {
                    source_domain_name: self._add_domain(
                        domain_name=source_domain_name,
                        args=args,
                        evaluator=evaluator,
                        data_loader=data_loader,
                    )
                }
            )

        return source_domains


    def _add_domain(
        self,
        domain_name: str,
        args: Namespace,
        evaluator,
        data_loader,
        lora_path: Union[str, Path, None] = None,
    ) -> Domain:
        """Adds a Domain instance to the domains list."""

        train_dataset_path = Path(args.train_dataset_path)
        assert train_dataset_path.exists(), train_dataset_path

        if lora_path is None:
            lora_path = self.lora_db_path / domain_name

        statistics: Dict[str, npt.NDArray] = self.embedding_manager.calculate_statistics(
            domain_name, lora_path, train_dataset_path,
        )

        train_average_embedding: npt.NDArray = statistics[
            "train_average_embedding"
        ]

        domain = Domain(
            domain_name,
            args,
            train_dataset_path=train_dataset_path,
            lora_path=lora_path,
            train_average_embedding=train_average_embedding,
            evaluator=evaluator,
            data_loader=data_loader,
        )

        return domain
    
    def _batch_merge(
        self,
        target_domains: List[str],
        mode: Literal["uniform", "centroid"],
        remove_target_adapter: bool = False,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Merge the source domains and benchmark the merged adapter on the target domains.
        """

        results = {}
        weights = {}

        for current_target_domain_name in target_domains:

            current_target_domain = self._target_domains[current_target_domain_name]
            
            self._set_current_target_domain(
                current_target_domain,
            )

            weight_dict, merged_adpater_name = self._merge(
                current_target_domain,
                remove_target_adapter,
                mode,
                top_k=len(self._source_domains) - 1 if remove_target_adapter else len(self._source_domains)
            )

            weights.update({current_target_domain.name: weight_dict})

            result_dict = self._benchmark_on_current_target_domain(
                name=merged_adpater_name,
                target_domain=current_target_domain
            )

            print(result_dict)

            result = self._get_result_from_dict(result_dict)

            results.update(
                {
                    current_target_domain.name: result
                }  # Different datasets have different evaluation methods so this won't always work
            )

            # Delete the adapter so we can add another with the same name but different weights (remove unused adapters)
            print(f"Deleting adapter {merged_adpater_name}.")
            self.current_model.delete_adapter(merged_adpater_name)

            print("\n")

        return results, weights


    def _merge(
        self,
        target_domain: Domain,
        remove_target_adapter: bool,
        mode: Literal["uniform", "centroid"],
        target_embedding=None,
        softmax_temperature: Optional[float] = 0.05,
        top_k: int = 5,  # number of domains to merge
        combination_type: str = "cat",
        similarity_measure: Callable[
            [npt.NDArray, npt.NDArray], np.float64
        ] = lambda v1, v2: np.linalg.norm(v1 - v2),
        sort_descending: bool = True

    ) -> Tuple[Dict[str, float], str]:
        """
        Merge the source domains and benchmark the merged adapter on the target domain.
        """
    
        source_domains = None
        if remove_target_adapter:
            print(f"Removing {target_domain.name} from source domains!")
            source_domains = [
                domain
                for _, domain in self._source_domains.items() if domain.name != target_domain.name
            ]
        else:
            source_domains = [
                domain
                for _, domain in self._source_domains.items()
            ]

        if mode == "uniform":

            weights = [1 / len(source_domains) for _ in range(len(source_domains))]
            domain_weight_mapping = {domain.name: weight for domain, weight in zip(source_domains, weights)}

            merged_name = ""
            for n, w in domain_weight_mapping.items():
                merged_name += f"_{n}_{str(w).replace('.','_')}"
            merged_name += f"_{combination_type}_{target_domain.name}" # Create a unique name for merged adapter so that it does not override existing adapters

            self._merge_adapters(
                merge_domains=[domain.name for domain in source_domains],
                weights=weights,
                merged_name=merged_name,
                combination_type=combination_type
            )

        elif mode == "centroid":

            similarity_mapping = self.observer.calculate_similarity_to_domains(
                embedding=target_embedding,
                domains=source_domains,  
                similarity_measure=similarity_measure,
                sort_descending=sort_descending
            )

            k_closest_names = list(similarity_mapping.keys())[: top_k]
            k_closest_similarities = list(similarity_mapping.values())[: top_k]

            print('')

            print(f"Similarities to {top_k} closest domains: ")
            for n, d in zip(k_closest_names, k_closest_similarities):
                print(f"{n}: {d}", end=", ")
            print("")

            ####################################################################################################

            # # ===== SAVE CSV PART =====
            # log_path = os.path.join("/data/aniruddh/SemLA_base1/csv_2", "idd.csv")

            # # Ensure directory exists
            # os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

            # # Check append mode
            # append = os.path.exists(log_path)

            # df = pd.DataFrame({
            #     "name": k_closest_names[:top_k],
            #     "similarity": k_closest_similarities[:top_k],
            #     "occurrence": [1] * top_k,
            # })

            # df.to_csv(log_path, mode='a', header=not append, index=False)
            # print(f"CSV successfully written to: {os.path.abspath(log_path)}")
            # print(hey)
            # =========================



            weights = self._calculate_adapter_weights(k_closest_similarities, softmax_temperature)
            print('weights---',weights)
            print('k_closest_names---',k_closest_names)

            domain_weight_mapping = {
                k_closest_name: weight
                for k_closest_name, weight in zip(k_closest_names, weights)
            }

            merged_name = ""
            for n, w in domain_weight_mapping.items():
                merged_name += f"_{n}_{str(w).replace('.','_')}"
            merged_name += f"_{combination_type}_{target_domain.name}" # Create a unique name for merged adapter so that it does not override existing adapters

            self._merge_adapters(
                merge_domains=k_closest_names,
                weights=weights,
                merged_name=merged_name,
                combination_type=combination_type
            )
        # print(hey)
        print(f"Setting {merged_name} as the active adapter.\n")
        self.current_model.set_adapter(merged_name)

        return domain_weight_mapping, merged_name

    def _merge_adapters(
        self,
        merge_domains: List[str],
        weights: List[float],
        merged_name: str,
        combination_type: str,
    ) -> None:
        """
        Merge the specified adapters with the specified weights.
        """
        
        print(f"Merging domains with weights:")
        for n, w in zip(merge_domains, weights):
            print(f"{n}: {w}", end=", ")
        print("")

        self.current_model.add_weighted_adapter(
            merge_domains,
            weights,
            merged_name,
            combination_type=combination_type,
        )

    def _calculate_adapter_weights(self, similarities:List[float], temperature: float) -> List[float]:
        """
        Calculate the weights for the merged adapter based on the similarities to the source domains.
        """
        weights = softmax(similarities, temperature)
        return weights

    def _setup_observer(self):
        print("Adding domain prototypes to the observer.")
        for domain in self._source_domains.values():
            self.observer.add_domain_prototypes(
                domain=domain, average_embedding=domain.train_average_embedding
            )

    def _get_result_from_dict(self, result_dict: Mapping) -> float:
        res = result_dict["sem_seg"].get("IoU", None)
        if res is None:
            res = result_dict["sem_seg"].get("mIoU")
        return res



    def _get_metadata_colors(self, dataset_name: str) -> np.ndarray:
        """Get color palette from dataset metadata."""
        from detectron2.data import MetadataCatalog
        from detectron2.utils.colormap import colormap as d2_colormap
        try:
            meta = MetadataCatalog.get(dataset_name)
            if hasattr(meta, 'stuff_colors') and meta.stuff_colors is not None:
                colors = np.array(meta.stuff_colors, dtype=np.uint8)
            else:
                colors = (d2_colormap(rgb=True) * 255).astype(np.uint8)
        except Exception:
            colors = (d2_colormap(rgb=True) * 255).astype(np.uint8)
        return colors

    # def _colorize_mask(self, mask: np.ndarray, colors: np.ndarray, ignore_label: int = 255) -> np.ndarray:
    #     """Colorize a segmentation mask using the provided color palette."""
    #     h, w = mask.shape
    #     colored = np.zeros((h, w, 3), dtype=np.uint8)
    #     unique_labels = np.unique(mask)
    #     for label in unique_labels:
    #         if label == ignore_label:
    #             continue
    #         if label < len(colors):
    #             colored[mask == label] = colors[label]
    #         else:
    #             np.random.seed(label)
    #             colored[mask == label] = np.random.randint(0, 255, 3)
    #     return colored


    def _colorize_mask(
        self, 
        mask: np.ndarray, 
        colors: np.ndarray, 
        ignore_label: int = 255,
        class_names: list = None,
        add_labels: bool = True,
        large_mask_area_thresh: int = 120000,  # Same as detectron2
    ) -> np.ndarray:
        """Colorize a segmentation mask using the provided color palette and optionally add class labels.
        
        Uses connected component analysis to place labels on each separate instance,
        similar to detectron2's Visualizer.
        """
        import cv2
        from PIL import ImageDraw, ImageFont
        
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        unique_labels = np.unique(mask)
        
        for label in unique_labels:
            if label == ignore_label:
                continue
            if label < len(colors):
                colored[mask == label] = colors[label]
            else:
                np.random.seed(label)
                colored[mask == label] = np.random.randint(0, 255, 3)
        
        # Add class name labels if requested
        if add_labels and class_names is not None:
            pil_img = Image.fromarray(colored)
            draw = ImageDraw.Draw(pil_img)
            
            # Try to get a font, fall back to default if not available
            try:
                font_size = max(12, min(h, w) // 40)  # Scale font with image size
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
            
            for label in unique_labels:
                if label == ignore_label:
                    continue
                if label >= len(class_names):
                    continue
                
                # Get class name
                class_name = class_names[label]
                
                # Create binary mask for this class
                binary_mask = (mask == label).astype(np.uint8)
                
                # Use connected components to find separate instances (like detectron2)
                num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
                
                if num_cc <= 1:  # No components found (only background)
                    continue
                
                # Get areas of each component (stats[:, -1] is the area)
                # Component 0 is background, so we skip it
                component_areas = stats[1:, cv2.CC_STAT_AREA]
                if len(component_areas) == 0:
                    continue
                
                # Find the largest component
                largest_component_id = np.argmax(component_areas) + 1  # +1 because we skipped background
                
                # Draw text on largest component and other very large components
                for cid in range(1, num_cc):
                    component_area = stats[cid, cv2.CC_STAT_AREA]
                    
                    # Label if it's the largest OR if it's larger than threshold
                    if cid == largest_component_id or component_area > large_mask_area_thresh:
                        # Use median position (more stable than centroid, like detectron2)
                        component_mask = (cc_labels == cid)
                        ys, xs = np.where(component_mask)
                        if len(xs) == 0:
                            continue
                        
                        cx = int(np.median(xs))
                        cy = int(np.median(ys))
                        
                        # Get text bounding box for background
                        bbox = draw.textbbox((cx, cy), class_name, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                        
                        # Adjust position to center text
                        tx = cx - text_w // 2
                        ty = cy - text_h // 2
                        
                        # Ensure text stays within image bounds
                        tx = max(2, min(tx, w - text_w - 2))
                        ty = max(2, min(ty, h - text_h - 2))
                        
                        # Draw background rectangle for readability
                        padding = 2
                        draw.rectangle(
                            [tx - padding, ty - padding, tx + text_w + padding, ty + text_h + padding],
                            fill=(0, 0, 0, 180)
                        )
                        
                        # Draw text in white
                        draw.text((tx, ty), class_name, fill=(255, 255, 255), font=font)
            
            colored = np.array(pil_img)
        
        return colored
        

    def _save_visualization(
        self,
        input_path: str,
        inputs: list,
        outputs: list,
        viz_output_dir: Path,
        colors: np.ndarray,
        ignore_label: int = 255,
        viz_prefix: str = "",
        class_names: list = None,  # List of class names for labeling
    ) -> None:
        """Save original image, ground truth, and prediction visualizations."""
        from pathlib import Path as PathLib
        
        base_name = PathLib(input_path).stem
        
        # Create subdirectories
        original_dir = viz_output_dir / "original"
        gt_dir = viz_output_dir / "ground_truth"
        # import ipdb; ipdb.set_trace()
        pred_dir = viz_output_dir / f"prediction{'_' + viz_prefix if viz_prefix else ''}"
        
        original_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original image
        try:
            original_img = np.array(Image.open(input_path).convert("RGB"))
            Image.fromarray(original_img).save(str(original_dir / f"{base_name}.png"))
        except Exception as e:
            print(f"Warning: Could not save original image: {e}")
        
        # Save ground truth if available
        input_dict = inputs[0]
        if "sem_seg" in input_dict:
            gt_mask = input_dict["sem_seg"].cpu().numpy()
            # gt_colored = self._colorize_mask(gt_mask, colors, ignore_label)
            gt_colored = self._colorize_mask(gt_mask, colors, ignore_label, class_names=class_names)
            Image.fromarray(gt_colored).save(str(gt_dir / f"{base_name}.png"))
        elif "sem_seg_file_name" in input_dict:
            try:
                gt_mask = np.array(Image.open(input_dict["sem_seg_file_name"]))
                # gt_colored = self._colorize_mask(gt_mask, colors, ignore_label)
                gt_colored = self._colorize_mask(gt_mask, colors, ignore_label, class_names=class_names)
                Image.fromarray(gt_colored).save(str(gt_dir / f"{base_name}.png"))
            except Exception as e:
                print(f"Warning: Could not load ground truth: {e}")
        
        # Save prediction
        try:
            pred = outputs[0]["sem_seg"].argmax(dim=0).cpu().numpy()
            # pred_colored = self._colorize_mask(pred, colors, ignore_label)
            pred_colored = self._colorize_mask(pred, colors, ignore_label, class_names=class_names)
            Image.fromarray(pred_colored).save(str(pred_dir / f"{base_name}.png"))
        except Exception as e:
            print(f"Warning: Could not save prediction: {e}")


    def benchmark_zeroshot(self, target_domains: List[str]) -> Dict[str, float]:
        results = {}

        for current_target_domain_name in target_domains:
            current_target_domain = self._target_domains[current_target_domain_name]

            args: Namespace = custom_domain_args(
                config_file=current_target_domain.args.config_file,
                output_path="output/benchmark_zeroshot/",
                num_gpus=1,
                model_path="models/model_final.pth",
            )

            self.current_model = load_catseg_model(
                args, model_path=args.model_path
            )

            result_dict = self._benchmark_on_current_target_domain(name="zeroshot", target_domain=current_target_domain)

            print(f"Zeroshot results for {current_target_domain.name}:")
            print(result_dict)

            result = self._get_result_from_dict(result_dict)

            results.update(
                {
                    current_target_domain.name: result
                }  # res can look different from dataset to dataset
            )

        return results
    
    def benchmark_oracle(self, target_domains: List[str]) -> Dict[str, float]:

        save_visualizations = True

        viz_output_dir = "visualizations/zeroshot"
        from detectron2.evaluation import inference_context, SemSegEvaluator
        from contextlib import ExitStack

        
        results = {}

        for current_target_domain_name in target_domains:
            
            current_target_domain = self._target_domains[current_target_domain_name]

            self._set_current_target_domain(
                current_target_domain,
            )

            self.current_model.set_adapter(current_target_domain.name)

            result_dict = self._benchmark_on_current_target_domain(
                name=current_target_domain.name,
                target_domain=current_target_domain
            )
            print(result_dict)

            result = self._get_result_from_dict(result_dict)

            results.update(
                {
                    current_target_domain.name: result
                }
            )

        return results

    def benchmark_uniform(
        self,
        target_domains: List[str],
        remove_target_adapter: bool,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        print(f"Starting uniform merge on domains {target_domains}")

        results, weights = self._batch_merge(
            target_domains=target_domains,
            mode="uniform",
            remove_target_adapter=remove_target_adapter,
        )

        print(f"Finished uniform merge on domains {target_domains}")

        return results, weights

    def benchmark_semla(
        self,
        target_domains: List[str],
        remove_target_adapter: bool = False,
        softmax_temperature: Optional[float] = 0.05,
        top_k: int = 5,  # number of domains to merge
        combination_type: str = "cat",
        similarity_measure: Callable[
            [npt.NDArray, npt.NDArray], np.float64
        ] = lambda v1, v2: 1 / np.linalg.norm(v1 - v2),
        sort_descending: bool = True
    ) -> Tuple[Dict[str, float], Dict[str, float]]:


        viz_output_dir = "visualizations/cs_van_lora"
        viz_prefix = "lora",  # e.g., "lora" or "conv_dora"
        
        
        from detectron2.evaluation import inference_context, SemSegEvaluator
        from contextlib import ExitStack

        results = {}
        weights = {}

        t0 = time.time()

        for current_target_domain_name in target_domains:
            
            current_target_domain = self._target_domains[current_target_domain_name]

            self._set_current_target_domain(
                current_target_domain,
            )

            data_loader = current_target_domain.data_loader
            evaluator = current_target_domain.evaluator

            model = self.current_model

            # These lines are adopted from
            # https://github.com/facebookresearch/detectron2/blob/2a420edb307c9bdf640f036d3b196bed474b8593/detectron2/evaluation/evaluator.py#L103

            evaluator.reset()

            # Setup visualization if enabled
            viz_colors = None
            viz_ignore_label = 255
            viz_dir = None
            save_visualizations = False
            viz_class_names = None
            from detectron2.data import MetadataCatalog
            viz_dir = Path(viz_output_dir) / current_target_domain_name
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Get dataset name for metadata
            dataset_name = f"{current_target_domain_name}_sem_seg_val"
            # Handle special cases
            # if current_target_domain_name in ["IE_Segmentation", "indraeye"]:
            #     dataset_name = "IE_Segmentation_sem_seg_val"
            # elif current_target_domain_name == "isprs":
            #     dataset_name = "isprs_potsdam_sem_seg_val"
            
            viz_colors = self._get_metadata_colors(dataset_name)
            try:
                meta = MetadataCatalog.get(dataset_name)
                viz_ignore_label = getattr(meta, 'ignore_label', 255)
                # Get class names for labeling
                viz_class_names = getattr(meta, 'stuff_classes', None)
            except Exception:
                viz_ignore_label = 255
            print(f"Visualization enabled. Saving to: {viz_dir}")

            # with ExitStack() as stack:
            # import ipdb; ipdb.set_trace()
            with ExitStack() as stack:
                if isinstance(model, nn.Module):
                    stack.enter_context(inference_context(model))
                stack.enter_context(torch.no_grad())

                for _, inputs in enumerate(data_loader):
                # for _, inputs in tqdm(enumerate(data_loader), total=len(data_loader)):

                    input_path = inputs[0]["file_name"]

                    print(f"Predicting image: {input_path}")

                    current_embedding = self.embedding_manager.embed_image(input_path)

                    weight_dict, merged_adpater_name = self._merge(
                        target_domain=current_target_domain,
                        remove_target_adapter=remove_target_adapter,
                        mode="centroid", 
                        target_embedding=current_embedding,
                        softmax_temperature=softmax_temperature,
                        top_k=top_k,
                        combination_type=combination_type,
                        similarity_measure=similarity_measure,
                        sort_descending=sort_descending,
                    )

                    for domain, weight in weight_dict.items():
                        weights.setdefault(domain, []).append(weight)

                    model = self.current_model

                    # outputs = model(inputs)

                    outputs = model(inputs)
                    
                    # Save visualizations if enabled
                    # import ipdb;  ipdb.set_trace()
                    if  viz_dir is not None:
                        self._save_visualization(
                            input_path=input_path,
                            inputs=inputs,
                            outputs=outputs,
                            viz_output_dir=viz_dir,
                            colors=viz_colors,
                            ignore_label=viz_ignore_label,
                            viz_prefix="lora",
                            class_names=viz_class_names,
                        )
                    

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    if isinstance(evaluator, SemSegEvaluator):
                        _ = evaluator.process(inputs, outputs)
                    else:
                        _ = evaluator.process_image(inputs, outputs)

                    self.current_model.delete_adapter(merged_adpater_name)

            print(f"Benchmarking on domain '{current_target_domain.name}' ...")
            result_dict = evaluator.evaluate()
            result = self._get_result_from_dict(result_dict)
            print(f"Result for domain '{current_target_domain.name}': {result}\n")

            results.update({current_target_domain.name: result})

            if not isinstance(evaluator, SemSegEvaluator):
                evaluator._working_dir.cleanup()

        total = time.time() - t0
        print(f"Experiment took {total} seconds to complete!")

        return results, weights
