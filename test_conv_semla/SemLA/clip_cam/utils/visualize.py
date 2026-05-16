from PIL import Image
from matplotlib import pyplot as plt 
import cv2
import torch
import numpy as np
import os
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

def visualize(image, text, logits, output_dir='.'):
        W, H = logits.shape[-2:]
        if isinstance(image, Image.Image):
            image = image.resize((W, H))
        elif isinstance(image, torch.Tensor):
            if image.ndim > 3:
                image = image.squeeze(0)
            image_unormed = (image.detach().cpu() * torch.Tensor(OPENAI_DATASET_STD)[:, None, None]) \
                             + torch.Tensor(OPENAI_DATASET_MEAN)[:, None, None]  # undo the normalization
            # image_unormed = (image.detach().cpu() * clip_pixel_std[:, None, None].detach().cpu()) \
            #                 + clip_pixel_mean[:, None, None].detach().cpu()  # undo the normalization
            image = Image.fromarray((image_unormed.permute(1, 2, 0).numpy() * 255).astype('uint8'))  # convert to PIL
        else:
            raise f'image should be either of type PIL.Image.Image or torch.Tensor but found {type(image)}'

        if logits.ndim > 3:
            logits = logits.squeeze(0)
        logits = logits.detach().cpu().numpy()

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        logits = (logits * 255).astype('uint8')
        heat_maps = [cv2.applyColorMap(logit, cv2.COLORMAP_JET) for logit in logits]

        alpha=0.6
        vizs = [(1 - alpha) * img_cv + alpha * heat_map for heat_map in heat_maps]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for viz, cls_name in zip(vizs, text):
            viz = cv2.cvtColor(viz.astype('uint8'), cv2.COLOR_BGR2RGB)
            plt.imshow(viz)
            plt.title(cls_name)
            plt.axis('off')
            plt.tight_layout()
            # keep changing this since its hardcoded for now
            output_path = os.path.join(output_dir, f'sp-dora-nav_{cls_name}_.png')
            plt.savefig(output_path)
            plt.close()  # Close figure to avoid memory issues
            print(f"Saved: {output_path}")
