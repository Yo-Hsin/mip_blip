import copy
import glob
import os
from argparse import ArgumentParser

import numpy as np
from lavis.common.gradcam import getAttMap
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from lavis.models import load_model_and_preprocess
from matplotlib import pyplot as plt
from PIL import Image
from tqdm.auto import tqdm


THRES = 2e-5
ALGO = 'greedy'
MIN_SIZE = 100
TOKEN = 7  # -1: whole sentence; 1: person; 7: emotion


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--image_root', type=str,
                        default='/nas/queue/group_affect/datasets/GroupEmoW/train')
    parser.add_argument('--model_name', type=str, default='blip_image_text_matching')
    parser.add_argument('--model_type', type=str, default='large')
    parser.add_argument('--question', type=str,
                        default='The person who has the most apparent emotion.')
    parser.add_argument('--save_dir', type=str, default='./mip_images/GroupEmoW/blip_itm/train')
    args = parser.parse_args()
    return args


def load_models(
    device,
    model_name,
    model_type
):
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=model_name,
        model_type=model_type,
        is_eval=True,
        device=device
    )
    return model, vis_processors, txt_processors


def load_images(image_path):
    raw_img = Image.open(image_path).convert('RGB')

    # Plot utilities for GradCam
    dst_w = 720
    w, h = raw_img.size
    scaling_factor = dst_w / w
    resized_img = raw_img.resize(
        (int(w * scaling_factor), int(h * scaling_factor))
    )
    norm_img = np.float32(resized_img) / 255

    return raw_img, norm_img


def plot_image(image, save_path):
    plt.imshow(image)
    plt.savefig(save_path)


def compute_attn_map(
    raw_img,
    question,
    model,
    vis_processors,
    txt_processors,
    device
):
    # Preprocess image and text inputs
    img = vis_processors['eval'](raw_img).unsqueeze(0).to(device)
    txt = txt_processors["eval"](question)

    # Compute GradCam
    txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    gradcam, _ = compute_gradcam(
        model, img, txt, txt_tokens, block_num=7
    )

    return gradcam, txt_tokens


def compute_boxes(coords):
    min_h = min(coords, key=lambda x: x[0])[0]
    min_w = min(coords, key=lambda x: x[1])[1]
    max_h = max(coords, key=lambda x: x[0])[0] + 1
    max_w = max(coords, key=lambda x: x[1])[1] + 1

    return min_h, min_w, max_h, max_w


def dfs(mask, h, w, coords):
    height, width = mask.shape

    if h < 0 or h >= height or w < 0 or w >= width or mask[h][w] == 0:
        return 0

    if mask[h][w] == 1 and (h, w) not in coords:
        coords.append((h, w))
    mask[h][w] = 0

    size = 1
    size += dfs(mask, h - 1, w, coords)
    size += dfs(mask, h + 1, w, coords)
    size += dfs(mask, h, w - 1, coords)
    size += dfs(mask, h, w + 1, coords)

    return size


def largest_connected_component(mask):
    height, width = mask.shape

    max_size = 0
    max_coords = []

    for h in range(height):
        for w in range(width):
            if mask[h][w] == 1:
                coords = []
                cur_size = dfs(mask, h, w, coords)
                if cur_size > max_size:
                    max_size = cur_size
                    max_coords = copy.deepcopy(coords)

    return compute_boxes(max_coords)


def greedy(mask, h, w):
    coords = []

    _ = dfs(mask, h, w, coords)

    return compute_boxes(coords)


def convert_to_original_scale(x_min, x_max, scale, max_length):
    x_min = x_min * scale
    x_max = x_max * scale

    x_diff = x_max - x_min
    if x_diff < MIN_SIZE:
        x_min = max(x_min - (MIN_SIZE - x_diff) / 2, 0)
        x_max = min(x_max + (MIN_SIZE - x_diff) / 2, max_length)

    return x_min, x_max


def crop_image(gradcam, img, save_path):
    width, height = img.size
    h, w = gradcam.shape
    w_scale = width / w
    h_scale = height / h

    binary_mask = (gradcam > THRES)

    if ALGO == 'LCC':
        min_h, min_w, max_h, max_w = largest_connected_component(binary_mask)
    elif ALGO == 'greedy':
        argmax_h = np.argmax(gradcam) // 24
        argmax_w = np.argmax(gradcam) % 24
        min_h, min_w, max_h, max_w = greedy(binary_mask, argmax_h, argmax_w)
    else:
        print('Please select one of the algorithms: "LCC", "greedy".')

    min_w, max_w = convert_to_original_scale(min_w, max_w, w_scale, width)
    min_h, max_h = convert_to_original_scale(min_h, max_h, h_scale, height)

    crop_img = img.crop((min_w, min_h, max_w, max_h))
    crop_img.save(save_path)


def per_token_visulization(
    norm_img,
    model,
    gradcam,
    txt_tokens,
    save_path,
    basename
):
    gradcam_iter = iter(gradcam[0][2:-1])
    token_id_iter = iter(txt_tokens.input_ids[0][1:-1])

    for _, (gc, token_id) in enumerate(zip(gradcam_iter, token_id_iter)):
        word = model.tokenizer.decode([token_id])
        gradcam_img = getAttMap(norm_img, gc.numpy(), blur=True)
        plot_image(
            gradcam_img,
            os.path.join(save_path, f'{word}_{basename}')
        )


def main(args):
    device = 'cpu'

    model, vis_processors, txt_processors = load_models(
        device,
        args.model_name,
        args.model_type
    )

    image_paths = glob.glob(f'{args.image_root}/**/*.jpg')

    os.makedirs(args.save_dir, exist_ok=True)

    for image_path in tqdm(image_paths):
        # For disconnect
        if os.path.exists(os.path.join(args.save_dir, os.path.basename(image_path))):
            continue

        raw_img, norm_img = load_images(image_path)

        gradcam, _ = compute_attn_map(
            raw_img,
            args.question,
            model,
            vis_processors,
            txt_processors,
            device
        )

        crop_image(
            gradcam[0][TOKEN + 2].numpy(),
            raw_img,
            os.path.join(args.save_dir, os.path.basename(image_path))
        )


if __name__ == '__main__':
    main(parse_arguments())
