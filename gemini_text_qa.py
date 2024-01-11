import os
import sys
import time

import google.generativeai as genai
from argparse import ArgumentParser
from tqdm.auto import tqdm


START = 0


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--caption_file', type=str,default='./captions/GroupEmoW/blip2_flan_t5_xl_groupemow_val_neg.txt')
    parser.add_argument('--model', type=str, default='gemini-pro')
    args = parser.parse_args()
    return args


def generate_content_from_text(caption, model):
    query = f'Describe the most important object or person in this sentence in short answer: {caption}'
    return model.generate_content(
        query,
        generation_config=genai.types.GenerationConfig(candidate_count=1),
        safety_settings=[
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ])


if __name__ == '__main__':
    args = parse_arguments()
    with open(args.caption_file, 'r') as f:
        contents = f.readlines()
        image_paths = [content.split('#')[0] for content in contents]
        captions = [content.split('#')[1] for content in contents]
    
    model = genai.GenerativeModel(args.model)

    for image_path, caption in tqdm(zip(image_paths[START:], captions[START:]), total=len(image_paths[START:])):
        response = generate_content_from_text(caption, model)
        if len(response.candidates) > 0:
            print(f'{image_path}#{caption}#{response.text}', flush=True)
        else:
            print(f'{image_path}#{caption}#{caption}', flush=True)
        time.sleep(2)
