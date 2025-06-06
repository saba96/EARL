# -*- coding: utf-8 -*-

import argparse
import json
import os

import torch

from emu3.tokenizer import Emu3VisionVQModel, Emu3VisionVQImageProcessor
from tqdm import tqdm

from pathlib import Path
import re
import numpy as np

from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import login, snapshot_download

with open("./emu3/train_image_editing/hf_token.txt", "r") as file:
    huggingface_token = file.read().strip()
login(huggingface_token)

def smart_resize(image: Image.Image, image_area: int = 720 * 720) -> Image.Image:
    """
    Resize an image while maintaining aspect ratio to match target area.
    
    Args:
        image (Image.Image): Input image to resize
        image_area (int): Target area in pixels (default: 720*720)
        
    Returns:
        Image.Image: Resized image
    """
    w, h = image.size
    current_area = h * w
    target_ratio = (image_area / current_area) ** 0.5

    th = int(round(h * target_ratio))
    tw = int(round(w * target_ratio))

    return image.resize((tw, th))


def create_CoT_dict(example, CoT_keys, original_w, original_h, resized_w, resized_h):
    if not CoT_keys:
        return {}

    CoT_dict = {}
    #  Important: Make sure the bbox_pattern here matches your data, otherwise change it to your bbox pattern
    bbox_pattern = re.compile(
        r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]'
    )
    for key in CoT_keys:
        text = example.get(key, "")
        if isinstance(text, str) and text.strip():
            def scale_bbox(match):
                bbox = np.array([float(match.group(i)) for i in range(1, 5)])
                scale_factors = [resized_w / original_w, resized_h / original_h] * 2
                scaled = np.round(bbox * scale_factors).astype(int)
                return str(scaled.tolist())

            updated_text = bbox_pattern.sub(scale_bbox, text)
            CoT_dict[key] = updated_text
    return CoT_dict

def main(args: argparse.Namespace):
    image_processor = Emu3VisionVQImageProcessor.from_pretrained(args.model_path, do_resize=False)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(args.model_path, device_map="cuda:0")
    image_tokenizer.eval()

    tokenized_file_path = f"{args.output_path}/list/tokenized_datalist.json"

    os.makedirs(f"{args.output_path}/feature", exist_ok=True)
    os.makedirs(f"{args.output_path}/list", exist_ok=True)

    # Load existing tokenized paths if available
    if os.path.exists(tokenized_file_path):
        print("Tokenized data found, loading from cache.")
        with open(tokenized_file_path, 'r') as f:
            datalist = json.load(f)
    else:
        print("Tokenized data not found, initializing new cache.")
        datalist = {
            "prefix": f"{args.output_path}/feature",
            "path_list": []
        }
        with open(tokenized_file_path, 'w') as f:
            json.dump(datalist, f)

    cached_indices = set(datalist["path_list"])


    if args.source == 'HF':
        base_dir = Path(snapshot_download(repo_id=args.dataset_name, repo_type="dataset"))
    else:
        base_dir = Path(args.dataset_name)
    parquet_files = sorted([str(file) for file in base_dir.rglob(f"**/*{args.split}*.parquet")])
    print("Loading from parquet files:")
    for file in parquet_files:
        print(file)
    datasets = []
    for file in parquet_files:
        dataset = Dataset.from_parquet(file)
        datasets.append(dataset)
    dataset = concatenate_datasets(datasets)

    id_key =  None
    if args.id_key is None:
        dataset = dataset.map(lambda example, idx: {"id": idx}, with_indices=True)
        id_key = "id"
    else:
        id_key = args.id_key
        print('id_key:', id_key)
    dataset = dataset.shuffle(seed=args.random_seed)
    new_entries = []
    update_interval = 1000
    for i, example in tqdm(enumerate(dataset), desc="Tokenizing dataset", total=len(dataset)):
        id = example[id_key]
        name = f"{id}.pth"
        if name in cached_indices:
            continue
        entry = {
            "idx": id,
            "original_image": example[args.original_image_key],
            "edit_instruction": example[args.edit_instruction_key],
            "CoT": '',
            "edited_image": example[args.edited_image_key]
        }
        prompt = entry["edit_instruction"]
        original_image = entry["original_image"]
        edited_image = entry["edited_image"]
        original_w, original_h = original_image.size
        original_image = smart_resize(original_image, args.image_area)
        edited_image = smart_resize(edited_image, args.image_area)
        w, h = original_image.size
        entry['CoT'] = create_CoT_dict(example, args.CoT_keys, original_w, original_h, w, h)
        original_image = image_processor(original_image, return_tensors="pt")["pixel_values"]
        edited_image = image_processor(edited_image, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            original_image = original_image.cuda()
            original_image_token_ids = image_tokenizer.encode(original_image)
            edited_image = edited_image.cuda()
            edited_image_token_ids = image_tokenizer.encode(edited_image)

        original_image_token_ids = original_image_token_ids.squeeze(0).cpu().numpy()
        edited_image_token_ids = edited_image_token_ids.squeeze(0).cpu().numpy()
        data = {
            "edit_id": name,
            "original_image": original_image_token_ids,
            "edited_image": edited_image_token_ids,
            "CoT": entry["CoT"],
            "instruction": prompt
        }
        torch.save(data, f"{args.output_path}/feature/{name}")
        new_entries.append(name)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % update_interval == 0 and new_entries:
            try:
                with open(tokenized_file_path, 'r') as f:
                    datalist = json.load(f)
                datalist["path_list"].extend(new_entries)
                with open(tokenized_file_path, 'w') as f:
                    json.dump(datalist, f)
                cached_indices.update(new_entries)
                cached_indices.update(datalist["path_list"])
                print(f"Added {len(new_entries)} entries to cache.")
                new_entries = []
            except Exception as e:
                print(f"Error updating cache at interval: {e}")

    if new_entries:
        datalist["path_list"].extend(new_entries)
        with open(tokenized_file_path, 'w') as f:
            json.dump(datalist, f)

    datalist["path_list"] = list(set(datalist["path_list"]))
    with open(tokenized_file_path, 'w') as f:
        json.dump(datalist, f)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tokenize data for Emu3, please note that the image should be in PIL format.")
    parser.add_argument("--dataset-name", type=str, help="The name of the dataset on Hugging Face or the path to the dataset.")
    parser.add_argument("--split", type=str, default="train", help="The split of data to be tokenized.")
    parser.add_argument('--model-path', type=str, help='vision tokenizer path', default="BAAI/Emu3-VisionTokenizer")
    parser.add_argument('--output-path', type=str, help='tokenized data save path')
    parser.add_argument('--id-key', type=str, help="Key to be used as id in the dataset")
    parser.add_argument('--image-area', type=int, default=720 * 720)
    parser.add_argument('--original-image-key', type=str, help="Key for the original image in the dataset")
    parser.add_argument('--edit-instruction-key', type=str, help="Key for the edit instruction text in the dataset")
    parser.add_argument("--CoT-keys", type=str, nargs='+', required=False, help="List of text keys to use for 'text for CoT'. Provide multiple keys separated by space.")
    parser.add_argument('--edited-image-key', type=str, help="Key for the edited image in the dataset")
    parser.add_argument('--random-seed', type=int, default=42, help="random seed for shuffling")
    parser.add_argument('--source', type=str, default='HF', help="Dataset location")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args)