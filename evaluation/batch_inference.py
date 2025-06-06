# -*- coding: utf-8 -*-
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

import torch

from emu3.train_image_editing.my_datasets_inference import Emu3RawDataset
from torch.utils.data import DataLoader

from vllm_processing_emu3 import (
    CachedPrefixConstrainedLogitsProcessor,
    Emu3Processor,
)

from vllm import LLM, SamplingParams

from typing import List, Dict
from PIL import Image

import datetime
import json
from emu3.mllm.processing_emu3 import Emu3Processor
import argparse
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from typing import Dict, List, Tuple
import os
from huggingface_hub import login, hf_hub_download
os.environ["VLLM_USE_V1"] = "0"

EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"
EMU_PATH = ""

tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True, do_resize=False)
image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval()
processor = None

def process_model_output(
    response_text: str,
    response_token_ids: List[int],
    mode: str,
    tokenizer: AutoTokenizer,
) -> tuple[List[int], str]:
    """Process the model's response based on the inference mode."""
    if mode == "E":
        response_token_ids = response_token_ids
        reasoning = ""
    elif mode == "CE":
        reasoning = response_text.split(tokenizer.img_token)[0]
    elif mode == "mixed":
        if tokenizer.img_token in response_text:
            reasoning = response_text.split(tokenizer.img_token)[0]
        else:
            response_token_ids = response_token_ids
            reasoning = ""
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return response_token_ids, reasoning

def plot_edit_comparison(original, edited, edit_instruction, id, save_dir, prefix):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original)
    ax[0].set_title("Original Image", fontsize=10, fontweight='bold', color='blue')
    ax[0].axis("off")
    ax[1].imshow(edited)
    ax[1].set_title("Edited Image", fontsize=10, fontweight='bold', color='green')
    ax[1].axis("off")
    fig.suptitle(edit_instruction, fontsize=14, fontweight='bold', ha='center', y=0.92, color='black')
    save_path = f'{save_dir}/{id}_{prefix}.png'
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    return save_path

def visualize_bboxes_and_keypoints(bboxes, keypoints_dict, image, edit_id, save_dir, prefix=''):
    os.makedirs(save_dir, exist_ok=True)
    w, h = image.size
    
    fig, ax = plt.subplots()
    ax.imshow(image)

    for label, bbox in bboxes.items():
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.text(x_min, y_min - 5, label, color='blue', fontsize=5, weight='bold', va='bottom')
        ax.add_patch(rect)

    for (x, y) in keypoints_dict.values():
        ax.add_patch(plt.Circle((x, y), radius=3, color="red"))

    visualized_output = f'{save_dir}/{edit_id}_{prefix}.png'
    plt.savefig(visualized_output)
    plt.close()
    return visualized_output

def extract_bboxes_and_keypoints(text: str) -> Tuple[Dict[str, List[int]], Dict[str, Tuple[int, int]]]:
    # Adjust this to your CoT format
    bbox_pattern = re.compile(
        r'[\[\(]\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]]'
    )
    
    bboxes = {}
    for i, match in enumerate(bbox_pattern.findall(text), start=1):
        bboxes[f"bbox {i}"] = [int(n) for n in match]

    keypoint_pattern = re.compile(r"[\[\(]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]]")
    
    keypoints = {}
    for i, match in enumerate(keypoint_pattern.findall(text), start=1):
        if len(match) == 2:
            keypoints[f"keypoint {i}"] = (int(match[0]), int(match[1]))

    return bboxes, keypoints

def main(args: argparse.Namespace):
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer, decode_mode=args.mode, vllm=True)
    args.cot_keys = args.cot_keys if args.cot_keys is not None else []
    
    # if args.model_location != 'local':
    #     with open("./emu3/train_image_editing/dummy_hf_token.txt", "r") as file:
    #         dummy_huggingface_token = file.read().strip()
    #     login(dummy_huggingface_token)
    
    llm = LLM(
        model=args.model_path,
        tokenizer=EMU_HUB,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.6,
        swap_space=4,
    )
    
    experiment_name = "-".join(args.cot_keys) if args.cot_keys else "no-reasoning"
    formatted_model_name = (args.model_path).replace("/", "_")

    trainer_state_path = os.path.join(args.model_path, "trainer_state.json")

    if os.path.exists(trainer_state_path):
        print(f"Found local trainer_state.json at {trainer_state_path}")
    else:
        print(f"Local trainer_state.json not found, trying to download from Hugging Face Hub...")
        try:
            trainer_state_path = hf_hub_download(repo_id=args.model_path, filename="trainer_state.json")
        except Exception as e:
            print(f"Could not retrieve trainer_state.json from Hugging Face Hub: {e}")
            trainer_state_path = None
            
    if trainer_state_path and os.path.exists(trainer_state_path):
        try:
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
                global_step = trainer_state.get("global_step", "unknown")
        except Exception as e:
            print(f"Error reading trainer_state.json: {e}")
            global_step = "unknown"
    else:
        global_step = "unknown"
        
    print("Global Step:", global_step)
    experiment_identifier = f"{formatted_model_name}_step-{global_step}_{experiment_name}"
    
    wandb.login()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_NAME = f'{args.split}_eval'
    TEST_TABLE_NAME = f'{args.mode}'
    columns = ["overview", "original_image", "edit_instruction", "gt_CoT", "reasoning", "reasoning_visualiztion", "valid_edited_image", "edited_image", "gt_edited", "data_source", "edit_type"]
    
    unique_name = f"{experiment_identifier}-{TEST_TABLE_NAME}-{RUN_NAME}-{timestamp}"
    args.save_dir = os.path.join(args.save_dir, unique_name)
    os.makedirs(args.save_dir, exist_ok=True)
    
    run = wandb.init(
        dir='./',
        project="Emu3",
        job_type="visualization",
        name=unique_name
    )
    
    table_data = wandb.Table(columns=columns)
    table_rows = []

    print("Experiment Name:", experiment_identifier)
    wandb_report = pd.DataFrame(columns=[])
    my_dataset = Emu3RawDataset(args)
    dataloader = DataLoader(my_dataset, batch_size=args.batch_size, collate_fn=Emu3RawDataset.collate_fn, num_workers=4, pin_memory=True)

   
    wandb_report = pd.DataFrame(columns=[])
    table_rows = []
    
    with torch.inference_mode():
        for batch in dataloader:
            original_image = batch["original_image"]
            gt_edited_image = batch["gt_edited_image"]
            instruction = batch["instruction"]
            data_source = batch["data_source"]
            edit_type = batch["edit_type"]
            CoT = batch["CoT"]
            edit_id = batch["edit_id"]
            
            original_images = [
                Image.fromarray(img.cpu().numpy().astype("uint8")) for img in original_image
            ]
            gt_edited_images = [
                Image.fromarray(img.cpu().numpy().astype("uint8")) for img in gt_edited_image
            ]
            
            list_prompt_token_ids = []
            list_sampling_params = []
            
            for img, instr in zip(original_images, instruction):
                inputs = processor(
                    text=instr,
                    image=img,
                    mode=args.mode,
                    image_area=args.image_area,
                    return_tensors="pt",
                    padding=False
                )
                h = inputs["image_size"][0][0]
                w = inputs["image_size"][0][1]
                constrained_fn = processor.build_prefix_constrained_fn(
                    np.array([h]), np.array([w])
                )
                logits_processor = [
                    CachedPrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1)
                ]
                sampling_params = SamplingParams(
                    n=args.k_sample,
                    temperature=1.0,
                    top_k=2048,
                    top_p=1.0,
                    max_tokens=args.max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id],
                    detokenize=False,
                    logits_processors=logits_processor if args.use_logit_processor else None,
                )
                list_prompt_token_ids.append(inputs["input_ids"])
                list_sampling_params.append(sampling_params)
                
            inputs = [
                {"prompt_token_ids": prompt_token_ids}
                for prompt_token_ids in list_prompt_token_ids
            ]
            

            responses = llm.generate(inputs, sampling_params=list_sampling_params)
            flattened_responses = [sample.token_ids for resp in responses for sample in resp.outputs]
            
            
            decoded_outputs = processor.batch_decode(
                flattened_responses,
                skip_special_tokens=False
            )
            
            for idx_, mm_list in enumerate(decoded_outputs):
                idx_i = idx_//args.k_sample
                id = edit_id[idx_i]
                image = original_images[idx_i]
                edited_image_path = None
                generated_text = ''
                valid_image = False

                edited_image = None
                generated_text = ''
                
                for idx, im in enumerate(mm_list):
                    if not isinstance(im, Image.Image):
                        generated_text += im
                        print('generated_text', generated_text)
                        continue
                    else:
                        im.save(f"{args.save_dir}/result_{id}_{idx}.png")
                        edited_image = f'{args.save_dir}/result_{id}_{idx}.png'
                        break
                        
                image.save(f"{args.save_dir}/org_{id}.png")
                gt_edited_images[idx_i].save(f"{args.save_dir}/gt_{id}.png")
                org_image_path = f"{args.save_dir}/org_{id}.png"
                gt_edited_path = f'{args.save_dir}/gt_{id}.png'
                
                parsed_bboxes, parsed_kps = extract_bboxes_and_keypoints(generated_text)
                grounding_path = visualize_bboxes_and_keypoints(parsed_bboxes, parsed_kps, image, id, args.save_dir, prefix='grounding')
                
                if edited_image == None:
                    im = Image.new("RGB", (256, 256), (255, 255, 255))
                    
                overview = plot_edit_comparison(image, im, instruction[idx_i], id, args.save_dir, prefix='overview')
                id += 1
                
                wandb_report_row = {
                    "original_image": org_image_path,
                    "edited_image": edited_image,
                    "gt_edited": gt_edited_path,
                    "edit_instruction": instruction[idx_i],
                    'gt_CoT': CoT[idx_i],
                    'reasoning': generated_text,
                    'grounding': grounding_path,
                    "data_source": data_source[idx_i], 
                    "overview": overview, 
                    "edit_type": edit_type[idx_i]
                }
                
                wandb_report = wandb_report._append(wandb_report_row, ignore_index=True)
                original_image = image.convert("RGB")
                
                if wandb_report_row['edited_image'] != None:
                    edited_image = im.convert("RGB")
                    valid_image = True
                else:
                    edited_image = Image.new("RGB", (256, 256), (255, 255, 255))
                    valid_image = False
                    
                gt_edited = Image.open(wandb_report_row['gt_edited']).convert("RGB")
                grounding = Image.open(wandb_report_row['grounding']).convert("RGB")
                overview = Image.open(overview).convert("RGB")
                
                row = [
                    wandb.Image(overview), wandb.Image(original_image), wandb_report_row['edit_instruction'], 
                    wandb_report_row['gt_CoT'], wandb_report_row['reasoning'], wandb.Image(grounding), 
                    valid_image, wandb.Image(edited_image), wandb.Image(gt_edited), 
                    wandb_report_row['data_source'], wandb_report_row['edit_type']
                ]
                
                table_rows.append(row)
                table_data = wandb.Table(data=table_rows, columns=columns)
                run.log({TEST_TABLE_NAME: table_data}, commit=True)
                
    json_path = f'{args.save_dir}/eval_report.jsonl'
    wandb_report.to_json(json_path, orient='records', lines=True)
    run.finish()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate edited image based on an image and a text instruction.")
    parser.add_argument("-m", "--model_path", type=str, help="The model path to be evaluated.")
    parser.add_argument("-s", "--save_dir", type=str, default="./", help="The directory to save the generated images.")
    parser.add_argument("-r", "--random_seed", type=int, default=42, help="The seed for sampling from data.")
    parser.add_argument("-n", "--sample_size", type=int, default=10, help="The random sample size from data.")
    parser.add_argument("-i", "--image_area", type=int, help="Image area used for resizing.")
    parser.add_argument("--split", type=str, default='train', help="Split to use for evaluation.")
    parser.add_argument('--source', type=str, default='HF', help="Dataset location")
    parser.add_argument('--cot_keys', nargs='*', default=[], help="List of CoT (Chain of Thought) keys")
    parser.add_argument('--dataset_names', nargs='*', default=[], help="List of datasets")
    parser.add_argument('--mode', type=str, default='CE', help="CE, E, or mixed")
    parser.add_argument('--max_new_tokens', type=int, default=2000, help="Max number of new tokens to be generated.")
    parser.add_argument('--model_location', type=str, default='local', help="Model location can be local or HF.")
    parser.add_argument('--image-area', type=int, default=720 * 720)
    parser.add_argument('--original-image-key', type=str, help="Key for the original image in the dataset")
    parser.add_argument('--edit-instruction', type=str, help="Key for the edit instruction text in the dataset")
    parser.add_argument('--edited-image-key', type=str, help="Key for the edited image in the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Index to ignore during processing.")
    parser.add_argument("--report_to_wandb", type=bool, default=False, help="Report to wandb, should be False for large sample size.")
    parser.add_argument("--visual_token_pattern", type=str, default="<|visual token {token_id:0>6d}|>", help="visual_token_pattern")
    parser.add_argument("--codebook_size", type=int, default=32768, help="codebook_size")
    parser.add_argument("--use_logit_processor", action="store_true", help="Whether to use logit processor for constrained generation")
    parser.add_argument("--k_sample", type=int, default=1, help="Generate k samples per prompt.")
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args)
