import os
os.environ["VLLM_USE_V1"] = "0"

import sys
sys.path.append('../')
sys.path.append('../evaluation/')

import gc
import time
import wandb
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from argparse import Namespace

from utils import plot_overview
from utils import pad_tensor_list, smart_resize
from utils import extract_bboxes_and_keypoints, visualize_bboxes_and_keypoints
from utils import update_bbox_in_text

from benchmarks import init_emuedit


from huggingface_hub import login, hf_hub_download

import contextlib
import gc

import torch

def cleanup():
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    # if not is_cpu():
    #     torch.cuda.empty_cache()


def eval_automatic(baseline_name):

    BATCH_SIZE = BATCH_SIZE_GLOBAL
    baseline = baseline_name
    BENCHMARK = "EmuEdit"
    EVAL_TABLE = "evaluation_table"

    data = init_emuedit()
    data = data.shuffle(seed=42).select(range(1000))


    idx_dict = {}
    instruction_dict = {}
    image_dict = {}
    task_type_dict = {}

    for sample in data:
        image = sample["image"]
        instruction = sample["instruction"]
        task_type = sample["task"]
        idxx = sample["idx"]


        img_size = str(image.size)
        
        # if RESIZE:
        #     image = smart_resize(image, RESOLUTION*RESOLUTION)
        #     gt_edited_image = smart_resize(gt_edited_image, RESOLUTION*RESOLUTION)

        if img_size not in list(image_dict.keys()):
            image_dict[img_size] = [image.convert("RGB")]
            instruction_dict[img_size] = [instruction]
            task_type_dict[img_size] = [task_type]
            idx_dict[img_size] = [idxx]
        else:
            image_dict[img_size].append(image.convert("RGB"))
            instruction_dict[img_size].append(instruction)
            task_type_dict[img_size].append(task_type)
            idx_dict[img_size].append(idxx)

    
    SAVE_DIR = f'{SAVE_DIR_GLOBAL}/results/{baseline}_final/{BENCHMARK}'


    if baseline == "InstructPix2Pix":
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        model_id = "timbrooks/instruct-pix2pix"

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16, 
                    safety_checker=None
                    ).to("cuda")
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )

    elif baseline == "MagicBrush":
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        model_id = "vinesmsuic/magicbrush-paper"

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16, 
                    safety_checker=None
                    ).to("cuda")
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )

    elif baseline == "InstructPix2Pix_XL":
        from diffusers import StableDiffusionXLInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        model_id = "diffusers/sdxl-instructpix2pix-768"

        pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16, 
                    safety_checker=None
                    ).to("cuda")
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )

    elif baseline == "MagicBrush_XL":
        from diffusers import StableDiffusionXLInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        model_id = "vinesmsuic/magicbrush-paper"

        pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16, 
                    safety_checker=None
                    ).to("cuda")
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )

    elif baseline == "Omnigen":
        from diffusers import OmniGenPipeline
        model_id = "Shitao/Omnigen-v1-diffusers"

        BATCH_SIZE = 10

        pipeline = OmniGenPipeline.from_pretrained(
                    model_id, 
                    torch_dtype=torch.bfloat16).to("cuda")
        pipeline.enable_model_cpu_offload()
    elif baseline == "Aurora":
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        model_id = "McGill-NLP/AURORA"

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16, 
                    safety_checker=None).to("cuda")
    elif baseline == "Aurora_XL":
        from diffusers import StableDiffusionXLInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        model_id = "McGill-NLP/AURORA"

        pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16, 
                    safety_checker=None).to("cuda")

    if baseline in ["InstructPix2Pix_XL", "MagicBrush_XL", "Aurora_XL", "InstructPix2Pix", "MagicBrush", "Aurora"]:
        images_dict = {"All_Sizes": [img for img_list in image_dict.values() for img in img_list]}
        instructions_dict = {"All_Sizes": [instr for instr_list in instruction_dict.values() for instr in instr_list]}
        task_types_dict = {"All_Sizes": [task for task_list in task_type_dict.values() for task in task_list]}
        idxs_dict = {"All_Sizes": [id for id_list in idx_dict.values() for id in id_list]}
    else:
        images_dict = image_dict
        instructions_dict = instruction_dict
        task_types_dict = task_type_dict
        idxs_dict = idx_dict



    os.makedirs(f'{SAVE_DIR}/results/{baseline}_final/{BENCHMARK}', exist_ok=True)

    project_name = f"Baseline_{baseline}_on_{BENCHMARK}_Auto"
    run_name = f"Baseline_{baseline}_on_{BENCHMARK}"

    run = wandb.init(
            dir=f'{SAVE_DIR}',
            project=project_name,
            job_type="visualization",
            name=run_name,
        )
    wandb_batch_table_rows = []
    wandb_report = pd.DataFrame(columns=[])
    
    uniq_id = 0

    for img_size_index, img_size in tqdm(enumerate(images_dict), desc="Processing batches with different image sizes"):
        print(f"Processing size {img_size_index+1} of {len(images_dict)} sizes")
        print(f"Processing images of size {img_size}")

        input_images = images_dict[img_size]
        edit_instructions = instructions_dict[img_size]
        task_type = task_types_dict[img_size]
        idx = idxs_dict[img_size]
        for batch_idx, i in tqdm(enumerate(range(0, len(input_images), BATCH_SIZE)), desc="Processing batches"):
            print(f"Processing batch {batch_idx+1} of {(len(input_images) // BATCH_SIZE)+1} batches")

            if i + BATCH_SIZE > len(input_images):
                batch_input_images = input_images[i:]
                batch_edit_instructions = edit_instructions[i:]
                batch_task_type = task_type[i:]
                batch_idx = idx[i:]
            else:
                batch_input_images = input_images[i:i + BATCH_SIZE]
                batch_edit_instructions = edit_instructions[i:i + BATCH_SIZE]
                batch_task_type = task_type[i:i + BATCH_SIZE]
                batch_idx = idx[i:i + BATCH_SIZE]

            
            if baseline in ["InstructPix2Pix", "MagicBrush"]:
                batch_edited_images = pipeline(
                    prompt=batch_edit_instructions, 
                    image=batch_input_images
                ).images

            elif baseline in ["InstructPix2Pix_XL", "MagicBrush_XL"]:
                batch_edited_images = pipeline(
                    prompt=batch_edit_instructions, 
                    image=batch_input_images
                ).images

            elif baseline == "Aurora":
                batch_edited_images = pipeline(
                    prompt=batch_edit_instructions, 
                    image=batch_input_images
                ).images

            elif baseline == "Aurora_XL":
                batch_edited_images = pipeline(
                    prompt=batch_edit_instructions, 
                    image=batch_input_images
                ).images

            elif baseline == "Omnigen":
                omnigen_batch_edit_instructions = []
                omnigen_batch_input_images = []
                for instruction in batch_edit_instructions:
                    omnigen_batch_edit_instructions.append(f'<img><|image_1|></img> {instruction}')
                for img in batch_input_images:
                    # omnigen_batch_input_images.append([img])
                    omnigen_batch_input_images.append([img.resize((512, 512))])

                batch_edited_images = pipeline(
                    prompt=omnigen_batch_edit_instructions, 
                    input_images=omnigen_batch_input_images, 
                    guidance_scale=2, 
                    img_guidance_scale=1.6, 
                    use_input_image_size_as_output=True, 
                    generator=torch.Generator(device="cpu").manual_seed(222)
                    ).images

            batch_overview_paths = []
            batch_input_image_paths = []
            batch_edited_image_paths = []
            # batch_gt_edited_image_paths = []
            # batch_grounding_image_paths = [] # Specific to OurModel

            for batch_sample_idx in tqdm(range(len(batch_input_images)), desc="Processing batches for saving images"):
                current_uniq_id = uniq_id + batch_sample_idx # Use a consistent ID for paths within this sample

                overview_path = f'{SAVE_DIR}/results/{baseline}_final/{BENCHMARK}/overview_{current_uniq_id}.png'
                input_image_path = f'{SAVE_DIR}/results/{baseline}_final/{BENCHMARK}/original_{current_uniq_id}.png'
                edited_image_path = f'{SAVE_DIR}/results/{baseline}_final/{BENCHMARK}/edited_{current_uniq_id}.png'
                # gt_edited_image_path = f'{SAVE_DIR}/results/{baseline}_final/{BENCHMARK}/gt_edited_{current_uniq_id}.png'
                # grounding_image_path = f'{SAVE_DIR}/results/{baseline}/{OUR_MODEL_ARGS.model_path.split("/")[-1]}_{OUR_MODEL_ARGS.revision}/{BENCHMARK}/grounding_{current_uniq_id}.png'
                
                
                overview = plot_overview(
                    batch_input_images[batch_sample_idx], 
                    batch_edited_images[batch_sample_idx], 
                    batch_edit_instructions[batch_sample_idx]
                )
                overview.save(overview_path)
                smart_resize(batch_input_images[batch_sample_idx], 256*256).save(input_image_path)
                smart_resize(batch_edited_images[batch_sample_idx], 256*256).save(edited_image_path)
                # smart_resize(batch_gt_edited_images[batch_sample_idx], 256*256).save(gt_edited_image_path)
                # batch_grounding_images[batch_sample_idx].save(grounding_image_path)
                
                batch_overview_paths.append(overview_path)
                batch_input_image_paths.append(input_image_path)
                batch_edited_image_paths.append(edited_image_path)
                # batch_gt_edited_image_paths.append(gt_edited_image_path)
                # if baseline == "OurModel":
                # batch_grounding_image_paths.append(grounding_image_path)


            for batch_sample_idx in tqdm(range(len(batch_input_images)), desc="Processing batches for wandb"):
                # Retrieve saved paths and other data for this sample
                overview_path = batch_overview_paths[batch_sample_idx]
                input_image_path = batch_input_image_paths[batch_sample_idx]
                edited_image_path = batch_edited_image_paths[batch_sample_idx]
                # gt_edited_image_path = batch_gt_edited_image_paths[batch_sample_idx]
                # if baseline == "OurModel":
                # grounding_image_path = batch_grounding_image_paths[batch_sample_idx]

                wandb_report_row = {
                    "overview": overview_path, 
                    "original_image": input_image_path,
                    "edit_instruction": batch_edit_instructions[batch_sample_idx],
                    # "gt_reasoning": batch_reasoning_super_concise[batch_sample_idx],
                    # "coord": batch_coords[batch_sample_idx] if eval_split == "train" else '',
                    # "reasoning": batch_generated_cot[batch_sample_idx],
                    "edited_image": edited_image_path,
                    # "grounding_image": grounding_image_path,
                    # 'gt_edited_image': gt_edited_image_path,
                    'id': batch_idx[batch_sample_idx],
                    'task': batch_task_type[batch_sample_idx],
                }
                wandb_batch_table_rows.append(
                    [
                        wandb.Image(overview_path), # Log path instead of image object
                        wandb.Image(input_image_path), 
                        batch_edit_instructions[batch_sample_idx], 
                        # batch_reasoning_super_concise[batch_sample_idx],
                        # batch_coords[batch_sample_idx] if eval_split == "train" else '',
                        # batch_generated_cot[batch_sample_idx],
                        wandb.Image(edited_image_path), 
                        # wandb.Image(grounding_image_path) if grounding_image_path else None, 
                        # wandb.Image(gt_edited_image_path),
                        wandb_report_row['id'],
                        wandb_report_row['task'],
                    ]
                )

                wandb_report = wandb_report._append(wandb_report_row, ignore_index=True)
                    
                # Make sure uniq_id is incremented correctly after processing all samples in the batch
                if batch_sample_idx == len(batch_input_images) - 1:
                    uniq_id += len(batch_input_images)

            columns = ["overview", "original_image", "edit_instruction", "edited_image",
                        "id", "task"]
            # columns = ["overview", "original_image", "edit_instruction", "gt_cot", "coord", "reasoning", "edited_image",
            #             "id", "task"]

            wandb_batch_table_data = wandb.Table(data=wandb_batch_table_rows, columns=columns)
            run.log({EVAL_TABLE: wandb_batch_table_data}, commit=True)
            wandb_batch_table_rows = [] # Reset for the next batch

            # Clear memory after each batch
            torch.cuda.empty_cache()     # Clear CUDA cache
            gc.collect()
            # del llm
            cleanup()


    json_path = f'{SAVE_DIR}/eval_report_{baseline}_final_{BENCHMARK}.json'
    wandb_report.to_json(json_path, orient='records', lines=True)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size to use for evaluation")
    parser.add_argument("--baseline", nargs="+", help="Model paths to use for evaluation")
    args = parser.parse_args()

    BATCH_SIZE_GLOBAL = args.batch_size
    SAVE_DIR_GLOBAL = args.save_dir


    for baseline in tqdm(args.baseline, desc="Evaluating models"):
        print(f"Evaluating {baseline}")
        eval_automatic(baseline)
