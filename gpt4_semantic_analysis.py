import os

import backoff
import openai
from argparse import ArgumentParser
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.auto import tqdm


client = openai.OpenAI(
    api_key = os.environ.get('OPENAI_API_KEY')
)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--caption_file', type=str,default='./captions/GAF/blip2_flan_t5_xl_train.txt')
    parser.add_argument('--model', type=str, default='gpt-4')
    parser.add_argument('--temperature', type=float, default=1)
    args = parser.parse_args()
    return args


# @backoff.on_exception(backoff.expo, openai.RateLimitError)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_chat_completion(messages, model='gpt-4', temperature=1, max_token=80):
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_token,
        temperature=temperature,
        messages=messages
    )

    return response.choices[0].message.content


def find_most_intense_phrase(caption, model='gpt-4', temperature=1, max_token=None):
    prompt = 'Answer the most important'
    messages = [
        {'role': 'system', 'content': 'Answer the most important person or object in the following sentence. Format your response in a noun with its corresponding description.'},
        {'role': 'user', 'content': caption}
    ]

    return generate_chat_completion(messages)


if __name__ == '__main__':
    args = parse_arguments()
    with open(args.caption_file, 'r') as f:
        contents = f.readlines()
        image_paths = [content.split('#')[0] for content in contents]
        captions = [content.split('#')[1] for content in contents]
    
    for image_path, caption in tqdm(zip(image_paths, captions), total=len(image_paths)):
        phrase = find_most_intense_phrase(caption)
        print(f'{image_path}#{caption}#{phrase}', flush=True)
