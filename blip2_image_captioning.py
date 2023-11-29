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


def image_captioning(model, processor, image_path):
    device = 'cpu'

    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(
        **inputs,
        do_sample=True,
        max_length=256,
        min_length=1,
        top_p=0.9,
        temperature=1,
        repetition_penalty=1.5,
        length_penalty=1.0,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text


if __name__ == '__main__':
    args = parse_arguments()

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name)

    generated_text = image_captioning(model, processor, args.image_path)
    print(generated_text)
