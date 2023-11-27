import os

from argparse import ArgumentParser
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Salesforce/blip2-flan-t5-xl')
    parser.add_argument('--image_path', type=str,
                        default='../dataset/GAF_3.0/train/Neutral/neu_88.jpg')
    return parser.parse_args()


def image_captioning(model_name, image_path):
    device = 'cpu'

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name)

    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print(generated_text)


if __name__ == '__main__':
    args = parse_arguments()
    image_captioning(args.model_name, args.image_path)
