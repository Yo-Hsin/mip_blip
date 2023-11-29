import glob
import os

from argparse import ArgumentParser
from lavis.models import load_model_and_preprocess
from tqdm.auto import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from blip_image_captioning import image_captioning
from caption_intensity_computation import find_most_intense_phrase


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--blip_model_name', type=str, default='blip_caption')
    parser.add_argument('--blip_model_type', type=str, default='large_coco')
    parser.add_argument('--image_root', type=str,
                        default='/nas/queue/group_affect/datasets/GAF_3.0/val/')
    return parser.parse_args()


def main(args):
    device = 'cpu'

    model, vis_processors, _ = load_model_and_preprocess(
        name=args.blip_model_name, model_type=args.blip_model_type, is_eval=True, device=device
    )

    image_paths = glob.glob(f'{args.image_root}/**/*.jpg')

    for image_path in tqdm(image_paths):
        caption = image_captioning(model, vis_processors, image_path, device)
        most_intense_phrase = find_most_intense_phrase(caption)
        print(f'{os.path.basename(image_path)}#{caption}#{most_intense_phrase}', flush=True)


if __name__ == '__main__':
    main(parse_arguments())
