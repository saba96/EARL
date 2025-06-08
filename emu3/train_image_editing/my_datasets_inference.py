# -*- coding: utf-8 -*-

import json
import os.path as osp
import random
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from datasets import Dataset as DatasetHF, concatenate_datasets
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor


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
    bbox_pattern = re.compile(
        r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]'
    )
    kp_pattern = re.compile(
        r"'(\w+)'\s*:\s*[\(\[]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\)\]]"
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


class Emu3RawDataset(Dataset):
    def __init__(self, args: "DataArguments"):
        super().__init__()
        self.args = args
        random.seed(self.args.random_seed)
        self.EMU_HUB = "BAAI/Emu3-Stage1"
        self.VQ_HUB = "BAAI/Emu3-VisionTokenizer"

        special_datasets = ["OmniEdit"]
        self.datasets = []
        n_samples_per_dataset = math.ceil(self.args.sample_size / len(args.dataset_names))
        for dataset_name in args.dataset_names:
            dataset_chunks = []
            if any(x in dataset_name for x in special_datasets):
                matching_sources = [x for x in special_datasets if x in dataset_name]
                data_source = matching_sources[0]
                if args.source == 'HF':
                    # with open("./emu3/train_image_editing/hf_token.txt", "r") as file:
                    #     huggingface_token = file.read().strip()
                    # login(huggingface_token)
                    if dataset_name == "TIGER-Lab/OmniEdit-Filtered-1.2M":
                        base_dir = Path(snapshot_download(repo_id=dataset_name, repo_type="dataset", allow_patterns=[f"*{args.split}*.parquet"]))
                    else:
                        base_dir = Path(snapshot_download(repo_id=dataset_name, repo_type="dataset"))
                else:
                    base_dir = Path(dataset_name)

                parquet_files = sorted([str(file) for file in base_dir.rglob(f"**/*{args.split}*.parquet")])

                print(f"Loading parquet files for dataset '{dataset_name}':")
                dataset_list = []
                for file in parquet_files:
                    dataset = DatasetHF.from_parquet(file)
                    dataset = dataset.map(lambda example: example.update({"data_source": data_source}) or example)
                    dataset_list.append(dataset)

                if not dataset_list:
                    print(f"Warning: No valid parquet files found for dataset '{dataset_name}'. Skipping.")
                    continue

                full_dataset = concatenate_datasets(dataset_list)
                sample_size = min(n_samples_per_dataset, len(full_dataset))
                print('sample size from', dataset_name, sample_size)
                sampled_dataset = full_dataset.select(random.sample(range(len(full_dataset)), sample_size))
                dataset_chunks.append(sampled_dataset) 
            dataset_chunks = concatenate_datasets(dataset_chunks)
            self.datasets.append(dataset_chunks) 
            if not self.datasets:
                raise ValueError("No valid datasets were loaded. Check dataset names and paths.")

        self.datasets = concatenate_datasets(self.datasets)
        self.datasets = self.datasets.map(lambda example, idx: {"id": idx}, with_indices=True)
        
    def __getitem__(self, index):
        data = self.datasets[index]
        id = data["id"]
        if "OmniEdit" in data['data_source']:   
            edit_instruction= data[self.args.edit_instruction][0]
        else:
            edit_instruction = data[self.args.edit_instruction]
        entry = {
            "idx": id,
            "original_image": data[self.args.original_image_key],
            "edit_instruction": edit_instruction,
            "CoT": '',
            "edited_image": data[self.args.edited_image_key],
            "data_source": data['data_source'],
            "edit_type": data[self.args.edit_type_key] if self.args.edit_type_key else "unknown"
        }
        prompt = entry["edit_instruction"]

        original_image = entry['original_image']
        edited_image = entry['edited_image'] 
        original_w, original_h = original_image.size
        
        original_image = smart_resize(original_image, self.args.image_area)
        edited_image = smart_resize(edited_image, self.args.image_area)
        w, h = original_image.size

        # Convert to tensors
        with torch.no_grad():
            original_image = torch.tensor(np.array(original_image), dtype=torch.float32)
            edited_image = torch.tensor(np.array(edited_image), dtype=torch.float32)
        if len(self.args.cot_keys):
            entry['CoT'] = create_CoT_dict(data, self.args.cot_keys, original_w, original_h, w, h)
            CoT = []
            for key in self.args.cot_keys:
                if key in data:
                    CoT.append(f"{key}: {str(data[key]).strip()}")
                else:
                    print(f"Key '{key}' is not available in data['CoT'] at index {data['edit_id']}, {data['CoT']}")
            entry['CoT'] = ' '.join(CoT)
        sample = {
            "edit_id": id,
            "original_image": original_image,
            "gt_edited_image": edited_image,
            "CoT": entry["CoT"],
            "instruction": prompt,
            "data_source": entry["data_source"],
            "edit_type": entry["edit_type"]
        }

        for k, v in sample.items():
            if k not in ["h", "w", "edit_id", "CoT", "instruction", "data_source", "edit_type"]:
                sample[k] = v.squeeze(0)

        return sample
    
    def __len__(self):
        return len(self.datasets)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate function for handling variable-length sequences.
        
        Args:
            batch (List[Dict[str, Any]]): List of samples to collate
            
        Returns:
            Dict[str, Any]: Collated batch
        """
        original_image = [sample["original_image"] for sample in batch]
        gt_edited_image = [sample["gt_edited_image"] for sample in batch]
        instruction = [sample["instruction"] for sample in batch]
        CoT = [sample["CoT"] for sample in batch]
        edit_id = [sample["edit_id"] for sample in batch]
        data_source = [sample["data_source"] for sample in batch]
        edit_type = [sample["edit_type"] for sample in batch]
        return {
            "edit_id": edit_id,
            "original_image": original_image,
            "gt_edited_image": gt_edited_image,
            "CoT": CoT,
            "instruction": instruction,
            "data_source": data_source,
            "edit_type": edit_type
            
        }

    def format_image_prompt(self, image_tokens):
        """
        Formats image tokens for processing.
        """
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)

        image_prompt = (
            self.tokenizer.boi_token +
            f"{h}*{w}" +
            self.tokenizer.img_token +
            imgstr +
            self.tokenizer.eol_token +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )

        return image_prompt

    def to_imgstr(self, image_tokens):
        """
        Converts image tokens to a string format.
        """
        image_token_str = [
            [
                self.args.visual_token_pattern.format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr

