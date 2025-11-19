import os

import re
import json
import numpy as np

from datasets import load_dataset
from datasets import Image as HFImage
from datasets import Dataset, Features, Value

from my_datasets_inference import Emu3RawDataset

from PIL import Image
from tqdm import tqdm
from argparse import Namespace
from huggingface_hub import snapshot_download
from datasets import concatenate_datasets
from huggingface_hub import hf_hub_download


def init_omniedit():
    data_files = {"dev": "data/dev-0000*-of-00001.parquet"}
    dataset = load_dataset("TIGER-Lab/OmniEdit-Filtered-1.2M", data_files=data_files, split="dev", verification_mode="no_checks")
    print("Initialized OmniEdit dataset")
    print("Size of dataset: ", len(dataset))
    return dataset


def init_magicbrush():
    magicbrush_data_path = "./benchmarks/magicbrush-test-00000.parquet"
    
    if not os.path.exists(magicbrush_data_path):
        images_root = "./MagicBrush_benchmark/images"
        test_split_json = "./MagicBrush_benchmark/edit_turns.json"
        with open(test_split_json, "r") as f:
            test_split = json.load(f)

        magicrbush_list= []
        for edit_sample in tqdm(test_split):
            img_id = edit_sample["input"].split("-")[0]
            img_path = os.path.join(images_root, img_id)

            input_img = Image.open(os.path.join(img_path, edit_sample["input"]))
            mask_img = Image.open(os.path.join(img_path, edit_sample["mask"]))
            output_img = Image.open(os.path.join(img_path, edit_sample["output"]))

            magicrbush_list.append({
                "input": HFImage().encode_example(np.array(input_img)),
                "mask": HFImage().encode_example(np.array(mask_img)),
                "output": HFImage().encode_example(np.array(output_img)),
                "instruction": edit_sample["instruction"]
            })

        features = Features(
                {
                    "input": HFImage(),
                    "mask": HFImage(),
                    "output": HFImage(),
                    "instruction": Value("string"),
                }
            )
        try:
            dataset = Dataset.from_list(magicrbush_list).cast(features)
            save_path = magicbrush_data_path
            dataset.to_parquet(save_path)
        except Exception as e:
            print("Error in saving dataset", e)
    else:
        dataset = load_dataset('parquet', data_files=[magicbrush_data_path])

    dataset = dataset["train"]
    print("Initialized MagicBrush dataset")
    print("Size of dataset: ", len(dataset))
    return dataset

def init_aurora():
    dataset = load_dataset("McGill-NLP/aurora-bench")
    dataset = dataset["test"]
    print("Initialized Aurora dataset")
    print("Size of dataset: ", len(dataset))
    return dataset

def init_ourValidation():
    args = Namespace(
        random_seed=75, 
        sample_size=10000000, #64, 
        dataset_names=[
            "./editing_annotations/omniedit", 
            "./editing_annotations/vismin", 
            "./editing_annotations/magicbrush/magicbrush_concise_v3", 
            "./editing_annotations/aurora_ag/aurora_ag_concise_v4", 
            "./editing_annotations/human_edit"
            ],
        source="local", 
        split="validation", 
        original_image_key="source_image", 
        edit_instruction="edit_instruction", 
        edited_image_key="edited_image", 
        image_area=65536, 
        cot_keys=["source_image_grounding_information", "conditioning_information"]
    )

    dataset = Emu3RawDataset(args)
    print("Initialized OurValidation dataset")
    print("Size of dataset: ", len(dataset))
    return dataset

def init_i2ebench():
    """Load I2EBench dataset from the specified folder structure."""
    import os
    import json
    from PIL import Image
    from tqdm import tqdm
    from datasets import Dataset, Features, Value, Image as HFImage, load_dataset
    
    # Path for the saved parquet file
    i2ebench_data_path = "./benchmarks/i2ebench-dataset.parquet"
    
    # Check if the parquet file already exists
    if os.path.exists(i2ebench_data_path):
        # Load from the parquet file
        dataset = load_dataset('parquet', data_files=[i2ebench_data_path])
        dataset = dataset["train"]
        print("Loaded I2EBench dataset from parquet file")
        print("Size of dataset: ", len(dataset))
        return dataset
    
    # Base path to the I2EBench dataset
    base_path = "./I2EBench"
    edit_data_path = os.path.join(base_path, "EditData")
    
    # Check if the path exists
    if not os.path.exists(edit_data_path):
        raise FileNotFoundError(f"I2EBench dataset path not found: {edit_data_path}")
    
    # Get all task folders in EditData
    task_folders = [f for f in os.listdir(edit_data_path) if os.path.isdir(os.path.join(edit_data_path, f))]
    print(f"Found {len(task_folders)} tasks in I2EBench dataset")
    
    # List to collect all samples
    all_samples = []
    
    # Collect all possible field names across all task data
    all_fields = set(["task", "ori_exp", "input", "sample_id"])
    
    # First pass: collect all field names
    print("First pass: collecting all field names...")
    for task_folder in tqdm(task_folders, desc="Scanning task fields"):
        task_dir = os.path.join(edit_data_path, task_folder)
        
        # Find JSON file in the task folder without numbers in filename
        json_files = [f for f in os.listdir(task_dir) if f.endswith('.json') and not any(c.isdigit() for c in f)]
        if not json_files:
            continue
        
        # Load the JSON file
        json_path = os.path.join(task_dir, json_files[0])
        try:
            with open(json_path, 'r') as f:
                task_data = json.load(f)
                
            # Collect field names from the first sample (assuming all samples in a task have same fields)
            if task_data and isinstance(task_data, dict) and len(task_data) > 0:
                first_sample = next(iter(task_data.values()))
                for key in first_sample.keys():
                    if key != "image":  # Skip image as it's handled separately
                        all_fields.add(key)
        except Exception:
            continue
    
    print(f"Collected {len(all_fields)} unique fields: {all_fields}")
    
    # Process each task folder
    for task_folder in tqdm(task_folders, desc="Processing I2EBench tasks"):
        task_dir = os.path.join(edit_data_path, task_folder)
        
        # Find JSON file in the task folder without numbers in filename
        json_files = [f for f in os.listdir(task_dir) if f.endswith('.json') and not any(c.isdigit() for c in f)]
        if not json_files:
            print(f"Warning: No JSON file without numbers found in {task_dir}, skipping")
            continue
        
        # Load the JSON file (assuming there's only one per task folder)
        json_path = os.path.join(task_dir, json_files[0])
        try:
            with open(json_path, 'r') as f:
                task_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {json_path}: {e}")
            continue
        
        # Process each sample in the task data
        # The JSON structure has numbered keys (1, 2, 3...) instead of being a list
        for sample_id, sample in task_data.items():
            try:
                # Construct absolute paths for images
                if "image" in sample:
                    input_path = os.path.join(task_dir, "input", sample["image"])
                    if os.path.exists(input_path):
                        input_img = Image.open(input_path).convert("RGB")
                    else:
                        print(f"Warning: Input image not found: {input_path}")
                        continue
                else:
                    print(f"Warning: Sample missing image field, skipping")
                    continue
                
                # Create a new sample with the required fields
                new_sample = {
                    "task": task_folder,
                    "instruction": sample.get("ori_exp", ""),
                    "image": HFImage().encode_example(np.array(input_img)),
                    "sample_id": sample_id
                }
                
                # Add all other possible fields with empty values if not present
                for field in all_fields:
                    if field not in new_sample:  # Skip already added fields
                        new_sample[field] = sample.get(field, "")  # Default to empty string
                
                all_samples.append(new_sample)
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
    
    # Create dataset features
    features = {
        "task": Value("string"),
        "instruction": Value("string"),
        "image": HFImage(),
        "sample_id": Value("string"),
    }
    
    # Add features for all other fields
    for field in all_fields:
        if field not in features:
            features[field] = Value("string")  # Default to string for all fields
    
    # Create the dataset
    try:
        dataset = Dataset.from_list(all_samples)
        print(f"Dataset columns: {dataset.column_names}")
        
        # Explicitly select columns to match features
        columns_to_keep = list(features.keys())
        dataset = dataset.select_columns(columns_to_keep)
        
        # Now cast the dataset with features
        dataset = dataset.cast(Features(features))
        
        # Save the dataset as a parquet file
        dataset.to_parquet(i2ebench_data_path)
        print(f"Successfully created I2EBench dataset with {len(dataset)} samples")
        print(f"Saved I2EBench dataset to {i2ebench_data_path}")
    except Exception as e:
        print(f"Error creating or saving dataset: {e}")
        raise
    
    return dataset

def init_omniedit_got(overlap_w_omniedit=True):
    dataset = load_dataset("LucasFang/OmniEdit-GoT")
    dataset = dataset["train"]
    print("Initialized OmniEdit-GoT dataset")
    print("Size of dataset: ", len(dataset))

    if overlap_w_omniedit:
        # Path for the cached overlapping samples
        overlap_cache_path = "./omniedit_got_overlap.parquet"
        
        if os.path.exists(overlap_cache_path):
            # Load from cache if exists
            dataset = load_dataset('parquet', data_files=[overlap_cache_path])
            dataset = dataset["train"]
            print("Loaded overlapping samples from cache")
        else:
            # Load OmniEdit validation set
            data_files = {"dev": "data/dev-0000*-of-00001.parquet"}
            omniedit_val = load_dataset("TIGER-Lab/OmniEdit-Filtered-1.2M", data_files=data_files, split="dev", verification_mode="no_checks")
            
            # Get sets of IDs
            got_ids = set(dataset["omni_edit_id"])
            omniedit_val_ids = set(omniedit_val["omni_edit_id"])
            
            # Find overlapping IDs
            overlapping_ids = got_ids.intersection(omniedit_val_ids)
            print(f"Found {len(overlapping_ids)} overlapping samples")
            
            # Create a mapping of omni_edit_id to got and coord
            got_mapping = {
                item['omni_edit_id']: {'got': item['got'], 'coord': item['coord']}
                for item in dataset
                if item['omni_edit_id'] in overlapping_ids
            }
            
            # Filter OmniEdit dataset to only include overlapping samples and add got/coord fields
            def add_got_fields(example):
                if example['omni_edit_id'] in got_mapping:
                    mapping = got_mapping[example['omni_edit_id']]

                    got = mapping['got']
                    coords = json.loads(mapping['coord'])
                    for key, value in coords.items():
                        got = re.sub(re.escape(key), json.dumps(value), got)
                    example['got'] = got
                    example['coord'] = mapping['coord']

                return example
            
            dataset = omniedit_val.filter(lambda x: x["omni_edit_id"] in overlapping_ids)
            dataset = dataset.map(add_got_fields)
            
            # Save to parquet
            dataset.to_parquet(overlap_cache_path)
            print(f"Saved overlapping samples to {overlap_cache_path}")

    return dataset

def init_omniedit_got_ourval():
    from huggingface_hub import hf_hub_download
    from datasets import Dataset, concatenate_datasets
    import os
    
    # Define the repository and files to download
    repo_id = "ServiceNow/escher-omniedit"
    parquet_files = [
        "data/addition-dev-00000.parquet",
        "data/addition-dev-00001.parquet",
        "data/attribute_modification-dev-00000.parquet",
        "data/attribute_modification-dev-00001.parquet",
        "data/env-dev-00000.parquet",
        "data/removal-dev-00000.parquet",
        "data/removal-dev-00001.parquet",
        "data/style-dev-00000.parquet",
        "data/style-dev-00001.parquet",
        "data/swap-dev-00000.parquet",
        "data/swap-dev-00001.parquet"
    ]
    
    # Download each parquet file
    datasets = []
    for file in parquet_files:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file,
            repo_type="dataset"
        )
        dataset = Dataset.from_parquet(file_path)
        datasets.append(dataset)
    
    # Concatenate all datasets
    dataset = concatenate_datasets(datasets)
    print("Initialized OmniEdit-GoT dataset")
    print("Size of dataset: ", len(dataset))
    return dataset

def init_tokenized_validation():
    import json
    import os
    import torch
    from datasets import Dataset, concatenate_datasets
    from huggingface_hub import hf_hub_download
    
    # Load the tokenized datalist
    tokenized_path = "./validation_22march_256/list/tokenized_datalist.json"
    with open(tokenized_path, 'r') as f:
        tokenized_data = json.load(f)
    
    # Extract task types and IDs from filenames
    tokenized_samples = []
    for path in tokenized_data["path_list"]:
        # Extract task type and ID from filename
        # Format: task_type_ID.pth
        parts = path.replace(".pth", "").split("_")
        task_id = parts[-1]
        task_type = "_".join(parts[0:-1])  # Join all parts except the last ID
        
        feature_path = os.path.join(tokenized_data["prefix"], path)
        # Load the PyTorch feature
        try:
            feature = torch.load(feature_path, weights_only=False)
            tokenized_samples.append({
                "id": task_type + '_' + task_id,
                "feature_path": feature_path,
                "feature": feature
            })
        except Exception as e:
            print(f"Error loading feature from {feature_path}: {e}")
            continue
    
    # Create dataset from samples
    dataset = Dataset.from_list(tokenized_samples)
    
    # Load ServiceNow/escher-omniedit validation set
    repo_id = "ServiceNow/escher-omniedit"
    parquet_files = [
        "data/addition-dev-00000.parquet",
        "data/addition-dev-00001.parquet",
        "data/attribute_modification-dev-00000.parquet",
        "data/attribute_modification-dev-00001.parquet",
        "data/env-dev-00000.parquet",
        "data/removal-dev-00000.parquet",
        "data/removal-dev-00001.parquet",
        "data/style-dev-00000.parquet",
        "data/style-dev-00001.parquet",
        "data/swap-dev-00000.parquet",
        "data/swap-dev-00001.parquet"
    ]
    
    # Download and load each parquet file
    escher_datasets = []
    for file in parquet_files:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file,
            repo_type="dataset"
        )
        escher_dataset = Dataset.from_parquet(file_path)
        escher_datasets.append(escher_dataset)
    
    # Concatenate all escher datasets
    escher_val = concatenate_datasets(escher_datasets)
    
    # Filter to keep only overlapping samples
    # escher_ids = set(escher_val["id"])
    # dataset = dataset.filter(lambda x: x["id"] in escher_ids)
    
    print("Initialized Tokenized Validation dataset")
    print("Size of dataset: ", len(dataset))
    return dataset


def init_Vismin():
    args = Namespace(
        random_seed=75, 
        sample_size=10000000, #64, 
        dataset_names=[
            "./editing_annotations/vismin", 
            ],
        source="local", 
        split="validation", 
        original_image_key="source_image", 
        edit_instruction="edit_instruction", 
        edited_image_key="edited_image", 
        image_area=65536, 
        cot_keys=["source_image_grounding_information", "conditioning_information"]
    )

    dataset = Emu3RawDataset(args)
    print("Initialized Vismin dataset")
    print("Size of dataset: ", len(dataset))
    return dataset

def init_emuedit():
    dataset = load_dataset("facebook/emu_edit_test_set")
    dataset = dataset["test"]
    print("Initialized EmuEdit dataset")
    print("Size of dataset: ", len(dataset))
    return dataset

def init_vismin_gold():
    repo_id = "mair-lab/escher-vismin-dev"
    parquet_files = [
        "data/train/counting-00001.parquet",
        "data/train/relation-00001.parquet"
    ]

    vismin_datasets = []
    for file in parquet_files:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file,
            repo_type="dataset"
        )
        # Extract task name from file name
        if "counting" in file:
            task_name = "counting"
        elif "relation" in file:
            task_name = "relation"
        else:
            task_name = "unknown"
        vismin_dat = Dataset.from_parquet(file_path)
        vismin_dat = vismin_dat.map(lambda x: {**x, "task": task_name})
        vismin_datasets.append(vismin_dat)
    
    # Concatenate all escher datasets
    vismin_gold = concatenate_datasets(vismin_datasets)
    print("Initialized Vismin Gold dataset")
    print("Size of dataset: ", len(vismin_gold))
    return vismin_gold
