import glob
import os

from argparse import ArgumentParser
from tqdm.auto import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from blip2_image_captioning import image_captioning
from caption_intensity_computation import find_most_intense_phrase


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--blip2_model_name', type=str,
                        default='Salesforce/blip2-flan-t5-xl')
    parser.add_argument('--image_root', type=str,
                        default='/nas/queue/group_affect/datasets/GAF_3.0/val/')
    # parser.add_argument('--finish_files', type=str,
    #                     default='./blip2-flan-t5-xl.txt')
    return parser.parse_args()


def main(args):
    processor = Blip2Processor.from_pretrained(args.blip2_model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(args.blip2_model_name)

    image_paths = glob.glob(f'{args.image_root}/**/*.jpg')

    # with open(args.finish_files, 'r') as f:
    #     comp_imgs = set([img.split('#')[0] for img in f.readlines()])

    for image_path in tqdm(image_paths):
        # if os.path.basename(image_path) in comp_imgs:
        #     continue

        caption = image_captioning(model, processor, image_path)
        most_intense_phrase = find_most_intense_phrase(caption)
        print(f'{os.path.basename(image_path)}#{caption}#{most_intense_phrase}', flush=True)


if __name__ == '__main__':
    main(parse_arguments())
