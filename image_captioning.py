import os

import torch
from argparse import ArgumentParser
from PIL import Image
from lavis.models import load_model_and_preprocess


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        default='../dataset/GAF_3.0/train/Negative/neg_294.jpg')
    parser.add_argument('--model_name', type=str, default='blip_caption')
    parser.add_argument('--model_type', type=str, default='large_coco')
    args = parser.parse_args()
    return args


def main(args):
    raw_image = Image.open(args.image_path).convert('RGB')
    raw_image = raw_image.resize((596, 437))

    device = 'cpu'

    model, vis_processors, _ = load_model_and_preprocess(
        name=args.model_name, model_type=args.model_type, is_eval=True, device=device
    )

    image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
    caption = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)
    print(os.path.basename(args.image_path), caption)

if __name__ == '__main__':
    main(parse_arguments())