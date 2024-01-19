import glob
import os
import sys
import time

import google.generativeai as genai
from argparse import ArgumentParser
from PIL import Image
from tqdm.auto import tqdm


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str,default='/nas/queue/group_affect/datasets/GAF_3.0/train/')
    parser.add_argument('--model', type=str, default='gemini-pro-vision')
    args = parser.parse_args()
    return args


def generate_content_from_image_text(image, model):
    # query = 'Which object or person plays the most important role in the image? Do not answer the reasons and do not include "image" nor "important". Answer in less than 10 words, but the response should be descriptive enough for identification.'
    # query = 'Describe the most important object or person in the image. Format your answer as: "The most important object/person is: []."'
    query = 'Which emotion that people in this image convey? The answer should be either positive, negative, or neutral? Format the answer as "The main emotion in this image is [positive, negative, neutral], because [explanation]."'
    response = model.generate_content(
        [query, image],
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
    response.resolve()
    
    return response.text


def main(args):
    model = genai.GenerativeModel(args.model)

    image_paths = glob.glob(f'{args.root}/**/*.jpg')
    
    for image_path in tqdm(image_paths):
        image = Image.open(image_path)
        response = generate_content_from_image_text(image, model)
        print(os.path.basename(image_path), response)
        time.sleep(2)


if __name__ == '__main__':
    main(parse_arguments())
