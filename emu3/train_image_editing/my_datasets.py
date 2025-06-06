# -*- coding: utf-8 -*-

import json
import os.path as osp
import random
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset


class Emu3FeatureDataset(Dataset):
    """
    Dataset class for EMU3 image editing tasks.
    
    This dataset handles loading and processing of image editing data, including:
    - Original and edited images
    - Text instructions
    - Chain of Thought (CoT) reasoning steps
    
    Attributes:
        args (DataArguments): Configuration arguments for the dataset
        tokenizer (Emu3Tokenizer): Tokenizer for processing text and image tokens
        filelist (List[Tuple[str, str]]): List of (prefix, filename) pairs
        bov (int): Beginning of vision token ID
        eov (int): End of vision token ID
    """

    def __init__(self, args: "DataArguments", validation:False, tokenizer: "Emu3Tokenizer"):
        """
        Initialize the dataset.
        
        Args:
            args (DataArguments): Configuration arguments
            validation (bool): Whether this is a validation dataset, False for training, True for validation
            tokenizer (Emu3Tokenizer): Tokenizer for processing text and images
        """
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
            validation_sample_per_dataset = self.args.validation_size // num_datasets
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

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample.
        
        Args:
            index (int): Index of the sample to get
            
        Returns:
            Dict[str, torch.Tensor]: Processed sample with input_ids, attention_mask, and labels
            
        Raises:
            RuntimeError: If data loading fails
            ValueError: If required data fields are missing
        """
        prefix, filename = self.filelist[index]
        path = osp.join(prefix, filename)
        try:
            data = torch.load(path, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Error loading data from {path}: {str(e)}")

        # Validate required fields
        required_fields = ["original_image", "edited_image", "instruction"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields in data at {path}: {missing_fields}")

        # Process original image
        original_image = data["original_image"]
        original_image_prompt = self.format_image_prompt(original_image)

        # Process edited image
        edited_image = data["edited_image"]
        edited_image = self.format_image_prompt(edited_image)

        # Process text instruction
        text_prompt = data["instruction"]
        h, w = data["edited_image"].shape
        # Create input sequence
        prompt = self.tokenizer.bos_token + original_image_prompt + text_prompt
        sample = self.tokenizer(
            prompt,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        ignore_length = len(sample['input_ids'][0])

        # Process Chain of Thought if available
        if data.get("CoT") and self.args.cot_keys:
            try:
                CoT = []
                for key in self.args.cot_keys:
                    if key in data["CoT"]:
                        CoT.append(f"{key}: {str(data['CoT'][key]).strip()}")
                    else:
                        print(f"Warning: Key '{key}' not found in CoT data at {path}")
                
                concatenated_CoT = ' '.join(CoT)
                if len(concatenated_CoT) > 1750:
                    concatenated_CoT = concatenated_CoT[:1750]
                    print(f"Warning: CoT trimmed to 1750 characters at {path}")
                
                output = f" Let's think step by step. <|start thinking|> {concatenated_CoT}<|end thinking|>{edited_image}{self.tokenizer.eos_token}"
            except Exception as e:
                print(f"Error processing CoT at {path}: {str(e)}")
                output = edited_image + self.tokenizer.eos_token
        else:
            output = edited_image + self.tokenizer.eos_token

        # Process output sequence
        edited_image_tokens = self.tokenizer(
            output,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        # Combine input and output sequences
        concatenated_input_ids = torch.cat([sample['input_ids'], edited_image_tokens['input_ids']], dim=1)
        attention_mask = torch.ones(concatenated_input_ids.size(), dtype=torch.long).to(concatenated_input_ids.device)
        
        sample['input_ids'] = concatenated_input_ids
        sample['attention_mask'] = attention_mask
        sample["labels"] = sample['input_ids'].clone()
        sample["labels"][:,:ignore_length] = self.args.ignore_index

        # Squeeze tensors
        for k, v in sample.items():
            sample[k] = v.squeeze(0)
        
        return sample

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate function for Emu3FeatureDataset to handle variable-length sequences.

        Args:
            batch (List[Dict[str, torch.Tensor]]): List of samples from __getitem__.

        Returns:
            Dict[str, torch.Tensor]: Batch of padded tensors with input_ids, attention_mask, labels, and position_ids.
        """
        input_ids = [sample["input_ids"] for sample in batch]
        attention_masks = [sample["attention_mask"] for sample in batch]
        labels = [sample["labels"] for sample in batch]

        max_length = max(x.shape[0] for x in input_ids)
        
        def pad_tensor_list(tensor_list: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
            """
            Pad a list of tensors to the same length.
            
            Args:
                tensor_list (List[torch.Tensor]): List of tensors to pad
                pad_value (int): Value to use for padding
                
            Returns:
                torch.Tensor: Stacked and padded tensors
            """
            return torch.stack([
                torch.cat([
                    torch.full((max_length - t.shape[0],), pad_value, dtype=t.dtype, device=t.device),
                    t
                ])
                for t in tensor_list
            ])

        padded_input_ids = pad_tensor_list(input_ids, pad_value=0)
        padded_attention_masks = pad_tensor_list(attention_masks, pad_value=0)
        padded_labels = pad_tensor_list(labels, pad_value=-100)
        
        position_ids = padded_attention_masks.long().cumsum(-1) - 1
        position_ids.masked_fill_(padded_attention_masks == 0, 1)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "labels": padded_labels,
            "position_ids": position_ids
        }

    def format_image_prompt(self, image_tokens: torch.Tensor) -> str:
        """
        Format image tokens into a prompt string.
        
        Args:
            image_tokens (torch.Tensor): Image token tensor
            
        Returns:
            str: Formatted image prompt
        """
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)

        return (
            self.tokenizer.boi_token +
            f"{h}*{w}" +
            self.tokenizer.img_token +
            imgstr +
            self.tokenizer.eol_token +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )

    def to_imgstr(self, image_tokens):
        image_token_str = [
            [
                self.args.visual_token_pattern.format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        return self.tokenizer.eol_token.join(image_row_str)