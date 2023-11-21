import os

from argparse import ArgumentParser
from PIL import Image
import requests
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        default='../dataset/GAF_3.0/train/Neutral/neu_88.jpg')
    parser.add_argument('--model_name', type=str, default='Salesforce/instructblip-vicuna-13b')
    parser.add_argument('--prompt', type=str, default='What is the most important object or person in this image?')
    args = parser.parse_args()
    return args


def main(args):
    raw_image = Image.open(args.image_path).convert('RGB')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = InstructBlipForConditionalGeneration.from_pretrained(args.model_name).to(device)
    processor = InstructBlipProcessor.from_pretrained(args.model_name)

    inputs = processor(images=raw_image, text=args.prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(f'Generated text: {generated_text}')


if __name__ == '__main__':
    main(parse_arguments())