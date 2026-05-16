import torch
from clip import clip
from torch.nn import functional as F
from einops import rearrange
import utils
import argparse
import os
import shutil

def parse_args():    
        parser = argparse.ArgumentParser(description="CLIP Image-Text Matching")
        parser.add_argument('--model_name', type=str, default="ViT-B/16", help='Name of the CLIP model')
        parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
        parser.add_argument('--text', type=str, required=True, help='Text input for matching (e.g., "Donald Trump")')
        parser.add_argument('--checkpoint', type=str, default=None, help='Path to CLIP checkpoint')
        parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output images')
        
        return parser.parse_args()

def main(args):        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #model_name = "ViT-B/16"
        # import ipdb; ipdb.set_trace()
        model = clip.load_clip(args.model_name, device, checkpoint=args.checkpoint)

        print(model, "0-___---____---____)00")
        print(HOIHOI)

        image = utils.process_input(args.image_path, device=device)

        vis_f = model.encode_image(image, dense=True)
        vis_f = F.normalize(vis_f, dim=-1)

        text = [str(args.text)]
        tokens = clip.tokenize(text).to(device)
        text_embeddings = model.encode_text(tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings.unsqueeze(1)

        W, H = (384, 384)
        img_txt_matching = vis_f[:,1:] @ torch.permute(text_embeddings, (1,2,0))
        
        # Dynamically calculate grid size from number of patches
        num_patches = img_txt_matching.shape[1]
        grid_size = int(num_patches ** 0.5)
        
        img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h', w=grid_size)
        img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')

        img_txt_matching = utils.min_max(img_txt_matching)

        # Create output subdirectory based on input image filename
        image_name = args.image_path.split("/")[-1]
        final_output_dir = os.path.join(args.output_dir, image_name)
        os.makedirs(final_output_dir, exist_ok=True)
        # Copy original image to output directory for easy comparison
        shutil.copy(args.image_path, os.path.join(final_output_dir, f"{image_name}"))
        utils.visualize(image, text, img_txt_matching, output_dir=final_output_dir)

if __name__ == "__main__":
        args = parse_args()
        main(args)