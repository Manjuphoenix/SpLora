import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F

def min_max(logits):
        B, num_prompt = logits.shape[:2]
        logits_min = logits.reshape(B, num_prompt, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        logits_max = logits.reshape(B, num_prompt, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        logits = (logits - logits_min) / (logits_max - logits_min)
        return logits

def process_input(image, device):
    clip_pixel_mean = torch.tensor([[[122.7709]],
                        [[116.7460]],
                        [[104.0937]]], device=device)

    clip_pixel_std = torch.tensor([[[68.5005]],
                        [[66.6322]],
                        [[70.3232]]], device=device)
        
    image = Image.open(image).convert("RGB")
    image = torch.tensor(np.array(image)).to(device, dtype=torch.float32)
    image = torch.permute(image, (2,0,1))
    image = (image - clip_pixel_mean) / clip_pixel_std
    
    size_divisibility = 32
    _, h, w = image.shape
    pad_h = (size_divisibility - h % size_divisibility) % size_divisibility
    pad_w = (size_divisibility - w % size_divisibility) % size_divisibility

    image_padded = F.pad(image, (0, pad_w, 0, pad_h), "constant", 0)
    
    clip_resolution = (384, 384)
    image_resized = F.interpolate(image_padded.unsqueeze(0), size=clip_resolution, mode='bilinear', align_corners=False)
    return image_resized
