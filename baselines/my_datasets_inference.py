# -*- coding: utf-8 -*-

import json
import os.path as osp
import random

import torch
from torch.utils.data import Dataset

from typing import List, Dict
from PIL import Image

from pathlib import Path
from datasets import Dataset as DatasetHF
from datasets import concatenate_datasets
# from datasets import Dataset, concatenate_datasets
import random
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import numpy as np
import re
import math


def smart_resize(image, image_area: int = 720 * 720):
    w, h = image.size
    current_area = h * w
    target_ratio = (image_area / current_area) ** 0.5

    th = int(round(h * target_ratio))
    tw = int(round(w * target_ratio))

    image = image.resize((tw, th))
    return image

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
            def scale_keypoint(match):
                name, x, y = match.group(1), float(match.group(2)), float(match.group(3))
                scaled_x = int(x * resized_w / original_w)
                scaled_y = int(y * resized_h / original_h)
                return f"'{name}': [{scaled_x}, {scaled_y}]"

            updated_text = bbox_pattern.sub(scale_bbox, text)
            updated_text = kp_pattern.sub(scale_keypoint, updated_text)
            CoT_dict[key] = updated_text
    return CoT_dict

class Emu3FeatureDataset(Dataset):

    def __init__(self, args: "DataArguments", validation:False, tokenizer: "Emu3Tokenizer"):
        super().__init__()
        self.args = args
        random.seed(self.args.random_seed)
        self.filelist = []
        dataset_sizes = {}
        dataset_coefficients = {}
        if not validation:
            print('self.args.coefficients:', self.args.coefficients)
            if self.args.coefficients == []:
                self.args.coefficients = [None] * len(self.args.data_paths)
                print('****Important: Upsampling all to size of the largest dataset****')
            for path, coefficient in zip(self.args.data_paths, self.args.coefficients):
                with open(path) as f:
                    d = json.load(f)
                    prefix = d["prefix"]
                    files = d["path_list"]
                    dataset_sizes[prefix] = len(files)
                    dataset_coefficients[prefix] = coefficient
                    # self.filelist.extend([(prefix, f) for f in files])
            

            if all(coef is None for coef in dataset_coefficients.values()):
                max_size = max(dataset_sizes.values())
                dataset_coefficients = {prefix: float(max_size) / size for prefix, size in dataset_sizes.items()}
            else:
                dataset_coefficients = {prefix: coef if coef is not None else 1.0 for prefix, coef in dataset_coefficients.items()}
            
            print('dataset_coefficients', dataset_coefficients)

            for path in self.args.data_paths:
                with open(path) as f:
                    d = json.load(f)
                    prefix = d["prefix"]
                    files = d["path_list"]
                coef = dataset_coefficients.get(prefix, 1.0)
                sampled_files = []
                if coef >= 1:
                    sampled_files.extend([(prefix, f) for f in files] * int(coef))
                    remainder = coef - int(coef)
                    if remainder > 0:
                        sample_size = int(len(files) * remainder)
                        sampled_files.extend(random.sample([(prefix, f) for f in files], sample_size))
                elif coef < 1: 
                    sample_size = int(len(files) * coef)
                    sampled_files = random.sample([(prefix, f) for f in files], sample_size)
                else:
                    sampled_files = []
                self.filelist.extend(sampled_files)
                print('dataset name', prefix, 'original size:', dataset_sizes[prefix], 'new size:', len(sampled_files))
        else:
            num_datasets = len(self.args.validation_paths)
            validation_sample_per_dataset = math.ceil(self.args.validation_size // num_datasets)
            for path in self.args.validation_paths:
                with open(path) as f:
                    d = json.load(f)
                    prefix = d["prefix"]
                    files = d["path_list"]
                    dataset_sizes[prefix] = len(files)
                sample_size = min(validation_sample_per_dataset, len(files))
                sampled_files = random.sample([(prefix, f) for f in files], sample_size)
                self.filelist.extend(sampled_files)
                print(f"Validation Dataset: {prefix}, Original Size: {dataset_sizes[prefix]}, Sampled Size For Validation: {sample_size}")
        random.shuffle(self.filelist)
        self.tokenizer = tokenizer
        self.bov = tokenizer.encode(args.visual_token_pattern.format(token_id=0))[0]
        self.eov = tokenizer.encode(args.visual_token_pattern.format(token_id=args.codebook_size - 1))[0]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index: int):
        prefix, filename = self.filelist[index]
        path = osp.join(prefix, filename)
        data = torch.load(path)

        original_image= data["original_image"]
        original_image_prompt = self.format_image_prompt(original_image)

        edited_image = data["edited_image"]
        edited_image = self.format_image_prompt(edited_image)

        text_prompt = data["instruction"]
        h, w = data["edited_image"].shape

        prompt = self.tokenizer.bos_token + original_image_prompt + text_prompt  + '<start of reasoning> '
        sample = self.tokenizer(
            prompt,
            # self.tokenizer.boi_token +
            # f"{h}*{w}" +
            # self.tokenizer.img_token,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        for k, v in sample.items():
            if k not in ["h", "w"]:
                sample[k] = v.squeeze(0)
        return sample
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        # tested it and looks fine
        """
        Collate function for Emu3FeatureDataset to handle variable-length sequences.

        Args:
            batch (List[Dict]): List of samples from __getitem__.

        Returns:
            Dict[str, torch.Tensor]: Batch of padded tensors.
        """
        input_ids = [sample["input_ids"] for sample in batch]
        attention_masks = [sample["attention_mask"] for sample in batch]
        # labels = [sample["labels"] for sample in batch]
        h = [sample["h"] for sample in batch]
        w = [sample["w"] for sample in batch]

        max_length = max(x.shape[0] for x in input_ids)

        def pad_tensor_list(tensor_list, pad_value=0):
            # right
            # return torch.stack([torch.cat([t, torch.full((max_length - t.shape[0],), pad_value, dtype=t.dtype)]) for t in tensor_list])
            # left padding
            return torch.stack([
                torch.cat([torch.full((max_length - t.shape[0],), pad_value, dtype=t.dtype, device=t.device), t])  # Left-padding
                for t in tensor_list
            ])


        padded_input_ids = pad_tensor_list(input_ids, pad_value=0)
        padded_attention_masks = pad_tensor_list(attention_masks, pad_value=0)
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "h": h,
            "w": w
        }


    def format_image_prompt(self, image_tokens):
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


class Emu3RawDataset(Dataset):
    def __init__(self, args: "DataArguments"):
        super().__init__()
        self.args = args
        random.seed(self.args.random_seed)
        self.EMU_HUB = "BAAI/Emu3-Stage1"
        self.VQ_HUB = "BAAI/Emu3-VisionTokenizer"

        # self.tokenizer = AutoTokenizer.from_pretrained(self.EMU_HUB, trust_remote_code=True)
        # self.image_processor = AutoImageProcessor.from_pretrained(self.VQ_HUB, trust_remote_code=True, do_resize=False)
        # self.image_tokenizer = AutoModel.from_pretrained(self.VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval()
        # self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer, decode_mode=args.mode)
        special_datasets = ["vismin", "aurora", "omniedit", "magicbrush", 
                            "human", "jester", "motionfix", "object3dedit", "vision_synthesis"]
        self.datasets = []
        n_samples_per_dataset = math.ceil(self.args.sample_size / len(args.dataset_names))
        dataset_edit_types = {
            "magicbrush": "no_edit_type",
            "aurora": "action",
            "vismin": ["attribute", "counting", "object", "relation"],
            "omniedit": ["addition", "attribute_modification", "removal", "swap"],
            "human": ["action", "add", "counting", "relation", "remove", "replace"]
        }
        for dataset_name in args.dataset_names:
            dataset_chunks = []
            if any(x in dataset_name for x in special_datasets):
                matching_sources = [x for x in special_datasets if x in dataset_name]
                data_source = matching_sources[0]
                if args.source == 'HF':
                    with open("./emu3/train_image_editing/hf_token.txt", "r") as file:
                        huggingface_token = file.read().strip()
                    login(huggingface_token)
                    base_dir = Path(snapshot_download(repo_id=dataset_name, repo_type="dataset"))
                else:
                    base_dir = Path(dataset_name)

                parquet_files = sorted([str(file) for file in base_dir.rglob(f"**/*{args.split}*.parquet")])

                print(f"Loading parquet files for dataset '{dataset_name}':")
                # for file in parquet_files:
                #     print(file)
                dataset_list = []
                for file in parquet_files:
                    print(file)
                    edit_type = "unknown"
                    if data_source in dataset_edit_types:
                        dataset_types = dataset_edit_types[data_source]
                        if isinstance(dataset_types, list):
                            for edit in dataset_types:
                                if f"/{edit}/" in file or f"_{edit}_" in file:
                                    edit_type = edit
                                    break
                        else:
                            edit_type = dataset_types

                    dataset = DatasetHF.from_parquet(file)
                    dataset = dataset.map(lambda example: example.update({"data_source": data_source, "edit_type": edit_type}) or example)
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
            # dataset_chunks = dataset_chunks.map(lambda example: example.update({"data_source": data_source}) or example)
            self.datasets.append(dataset_chunks) 
            if not self.datasets:
                raise ValueError("No valid datasets were loaded. Check dataset names and paths.")

        self.datasets = concatenate_datasets(self.datasets)
        self.datasets = self.datasets.map(lambda example, idx: {"id": idx}, with_indices=True)
        # self.datasets = self.datasets.shuffle(seed=args.random_seed)
    
    def __getitem__(self, index):
        data = self.datasets[index]
        id = data["id"]
        entry = {
            "idx": id,
            "original_image": data[self.args.original_image_key],
            "edit_instruction": data[self.args.edit_instruction],
            "CoT": '',
            "edited_image": data[self.args.edited_image_key],
            "data_source": data['data_source'],
            "edit_type": data['edit_type']
        }
        prompt = entry["edit_instruction"]

        original_image = entry['original_image']
        edited_image = entry['edited_image'] 
        original_w, original_h = original_image.size
        # print(original_w, original_h)
        original_image = smart_resize(original_image, self.args.image_area)
        edited_image = smart_resize(edited_image, self.args.image_area)
        w, h = original_image.size
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
            # "w": w,
            # "h": h,
            "instruction": prompt,
            "data_source": entry["data_source"],
            "edit_type": entry["edit_type"]
        }
        # print('sample orginal', len(sample)) 

        for k, v in sample.items():
            if k not in ["h", "w", "edit_id", "CoT", "instruction", "data_source", "edit_type"]:
                sample[k] = v.squeeze(0)
        # print('sample orginal', len(sample))
        return sample
    
    def __len__(self):
        return len(self.datasets)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate function for handling variable-length sequences.
        """
        original_image = [sample["original_image"] for sample in batch]
        gt_edited_image = [sample["gt_edited_image"] for sample in batch]
        instruction = [sample["instruction"] for sample in batch]
        # h = [sample["h"] for sample in batch]
        # w = [sample["w"] for sample in batch]
        CoT = [sample["CoT"] for sample in batch]
        edit_id = [sample["edit_id"] for sample in batch]
        data_source = [sample["data_source"] for sample in batch]
        edit_type = [sample["edit_type"] for sample in batch]
        return {
            "edit_id": edit_id,
            "original_image": original_image,
            "gt_edited_image": gt_edited_image,
            # "h": h,
            # "w": w,
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

