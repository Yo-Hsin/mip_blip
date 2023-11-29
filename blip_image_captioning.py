import os

import torch
from argparse import ArgumentParser
from PIL import Image
from lavis.models import load_model_and_preprocess


W = 596
H = 437


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        default='/nas/queue/group_affect/datasets/GAF_3.0/val/Negative/neg_998.jpg')
    parser.add_argument('--model_name', type=str, default='blip_caption')
    parser.add_argument('--model_type', type=str, default='large_coco')
    args = parser.parse_args()
    return args


def image_captioning(model, vis_processors, image_path, device='cpu'):
    raw_image = Image.open(image_path).convert('RGB')
    raw_image = raw_image.resize((W, H))

    image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
    caption = model.generate({"image": image}, use_nucleus_sampling=True)[0]
    
    return caption


if __name__ == '__main__':
    args = parse_arguments()

    device = 'cpu'

    model, vis_processors, _ = load_model_and_preprocess(
        name=args.model_name, model_type=args.model_type, is_eval=True, device=device
    )

    caption = image_captioning(model, vis_processors, args.image_path)
    print(caption)
