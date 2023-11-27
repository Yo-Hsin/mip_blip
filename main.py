import os

from argparse import ArgumentParser

from blip2_image_captioning import image_captioning
from caption_intensity_computation import find_most_intense_phrase
from ..GroundingDINO.demo.inference_on_a_image import *


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--blip2_model_name', type=str,
                        default='Salesforce/blip2-flan-t5-xl')
    parser.add_argument('--image_path', type=str,
                        default='../dataset/GAF_3.0/train/Neutral/neu_88.jpg')
    parser.add_argument('--save_dir', type=str,
                        default='./mip_result')
    return parser.parse_args()


def main(args):
    caption = image_captioning(args.blip2_model_name, args.image_path)

    most_intense_phrase = find_most_intense_phrase(caption)
    if most_intense_phrase == 'No strong emotional expression found.':
        most_intense_phrase = caption
    
    os.makedirs(args.save_dir, exist_ok=True)



if __name__ == '__main__':
    main(parse_arguments())