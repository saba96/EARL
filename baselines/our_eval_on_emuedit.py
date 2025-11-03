import os
os.environ["VLLM_USE_V1"] = "0"

import sys
sys.path.append('../')
sys.path.append('../VIEScore/src/')
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
from constants import openai_key

from viescore.metric import VIEScore
from utils import plot_overview
from utils import pad_tensor_list, smart_resize
from utils import extract_bboxes_and_keypoints, visualize_bboxes_and_keypoints
from utils import update_bbox_in_text

from benchmarks import init_emuedit

from vllm_processing_emu3 import (
    CachedPrefixConstrainedLogitsProcessor,
    Emu3Processor,
)

from vllm import LLM, SamplingParams

from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from emu3.mllm.processing_emu3 import Emu3Processor

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList

from huggingface_hub import login, hf_hub_download

import contextlib
import gc

import torch
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
# from vllm.utils import is_cpu


EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"

tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True, do_resize=False)
image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval()


def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    # if not is_cpu():
    #     torch.cuda.empty_cache()


def eval_automatic(OUR_MODEL_ARGS):
    baseline = "OurModel"
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

    
    OUR_MODEL_ARGS.save_dir = f'{SAVE_DIR}/results/{baseline}/{OUR_MODEL_ARGS.model_path.split("/")[-1]}_{OUR_MODEL_ARGS.revision}/{BENCHMARK}'

    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer, decode_mode=OUR_MODEL_ARGS.mode, vllm=True)

    if OUR_MODEL_ARGS.model_location != 'local':
        with open("./dummy_hf_token.txt", "r") as file:
            dummy_huggingface_token = file.read().strip()
        login(dummy_huggingface_token)

    llm = LLM(
        model=OUR_MODEL_ARGS.model_path,
        revision=OUR_MODEL_ARGS.revision,
        tokenizer=EMU_HUB,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        swap_space=4,
    )

    images_dict = {"All_Sizes": [img for img_list in image_dict.values() for img in img_list]}
    instructions_dict = {"All_Sizes": [instr for instr_list in instruction_dict.values() for instr in instr_list]}
    task_types_dict = {"All_Sizes": [task_type for task_type_list in task_type_dict.values() for task_type in task_type_list]}
    idxs_dict = {"All_Sizes": [idx for idx_list in idx_dict.values() for idx in idx_list]}

    os.makedirs(f'{SAVE_DIR}/results/{baseline}/{OUR_MODEL_ARGS.model_path.split("/")[-1]}_{OUR_MODEL_ARGS.revision}/{BENCHMARK}', exist_ok=True)

    project_name = f"Baseline_{baseline}_on_{BENCHMARK}_Auto"
    run_name = f"Baseline_{baseline}_on_{BENCHMARK}_{OUR_MODEL_ARGS.model_path.split('/')[-1]}_{OUR_MODEL_ARGS.revision}"

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
        task_types = task_types_dict[img_size]
        idxs = idxs_dict[img_size]

        for batch_idx, i in tqdm(enumerate(range(0, len(input_images), BATCH_SIZE)), desc="Processing batches"):
            print(f"Processing batch {batch_idx+1} of {(len(input_images) // BATCH_SIZE)+1} batches")

            if i + BATCH_SIZE > len(input_images):
                batch_input_images = input_images[i:]
                batch_edit_instructions = edit_instructions[i:]
                batch_task_types = task_types[i:]
                batch_idxs = idxs[i:]
            else:
                batch_input_images = input_images[i:i + BATCH_SIZE]
                batch_edit_instructions = edit_instructions[i:i + BATCH_SIZE]
                batch_task_types = task_types[i:i + BATCH_SIZE]
                batch_idxs = idxs[i:i + BATCH_SIZE]

            
            with torch.inference_mode():
                batch_input_images = [Image.fromarray(np.array(img).astype("uint8")) for img in batch_input_images]
                input_ids_list = []
                attention_mask_list = []
                h_list = []
                w_list = []
                batch_edit_instructions_modified = []
                list_prompt_token_ids = []
                list_sampling_params = []
                for img, instr in zip(batch_input_images, batch_edit_instructions):
                    img = smart_resize(img, OUR_MODEL_ARGS.image_area)

                    # if OUR_MODEL_ARGS.reasoning_input == "True":
                    #     if eval_split == "val_external":
                    #         reasoning, _ = update_bbox_in_text(smart_resize(img, 512*512), reasoning)
                    #     instr = instr + ' <|start thinking|> ' + reasoning + ' <|end thinking|>'
                    # else:
                    #     instr = instr

                    batch_edit_instructions_modified.append(instr)
                    inputs = processor(
                        text=instr, 
                        image=img,
                        mode=OUR_MODEL_ARGS.mode,
                        image_area=OUR_MODEL_ARGS.image_area,
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
                        n=OUR_MODEL_ARGS.k_sample,
                        temperature=1.0,
                        top_k=OUR_MODEL_ARGS.top_k,
                        top_p=1.0,
                        max_tokens=OUR_MODEL_ARGS.max_new_tokens,
                        stop_token_ids=[tokenizer.eos_token_id],
                        detokenize=False,
                        logits_processors=logits_processor if OUR_MODEL_ARGS.use_logit_processor else None,
                    )

                    list_prompt_token_ids.append(inputs["input_ids"])
                    list_sampling_params.append(sampling_params)


                inputs = [
                    {"prompt_token_ids": prompt_token_ids}
                    for prompt_token_ids in list_prompt_token_ids
                ]
                start = time.time()
                responses = llm.generate(inputs, sampling_params=list_sampling_params)
                # Corrected code to flatten responses
                flattened_responses = [sample.token_ids for resp in responses for sample in resp.outputs]
                print(len(responses), len(flattened_responses))
                print(responses, flattened_responses)
                decoded_outputs = processor.batch_decode(
                    flattened_responses,
                    skip_special_tokens=False
                )

                batch_edited_images = []
                batch_generated_cot = []
                batch_grounding_images = []

                for idx_i, mm_list in enumerate(decoded_outputs):    
                    input_image = batch_input_images[idx_i]           

                    edited_image = None
                    generated_text = ''

                    for idx, im in enumerate(mm_list):
                        if not isinstance(im, Image.Image):
                            generated_text += im
                            continue
                        else:
                            edited_image = im
                            break

                    parsed_bboxes, parsed_kps = extract_bboxes_and_keypoints(generated_text)
                    grounding_img = visualize_bboxes_and_keypoints(parsed_bboxes, parsed_kps, input_image)
                    if edited_image == None:
                        edited_image = Image.new("RGB", (256, 256), (255, 255, 255))

                    batch_edited_images.append(edited_image)
                    batch_generated_cot.append(generated_text)
                    batch_grounding_images.append(grounding_img)


            print("Here at 320")
            batch_overview_paths = []
            batch_input_image_paths = []
            batch_edited_image_paths = []
            # batch_gt_edited_image_paths = []
            batch_grounding_image_paths = [] # Specific to OurModel

            for batch_sample_idx in tqdm(range(len(batch_input_images)), desc="Processing batches for saving images"):
                current_uniq_id = uniq_id + batch_sample_idx # Use a consistent ID for paths within this sample

                overview_path = f'{SAVE_DIR}/results/{baseline}/{OUR_MODEL_ARGS.model_path.split("/")[-1]}_{OUR_MODEL_ARGS.revision}/{BENCHMARK}/overview_{current_uniq_id}.png'
                input_image_path = f'{SAVE_DIR}/results/{baseline}/{OUR_MODEL_ARGS.model_path.split("/")[-1]}_{OUR_MODEL_ARGS.revision}/{BENCHMARK}/original_{current_uniq_id}.png'
                edited_image_path = f'{SAVE_DIR}/results/{baseline}/{OUR_MODEL_ARGS.model_path.split("/")[-1]}_{OUR_MODEL_ARGS.revision}/{BENCHMARK}/edited_{current_uniq_id}.png'
                # gt_edited_image_path = f'{SAVE_DIR}/results/{baseline}/{OUR_MODEL_ARGS.model_path.split("/")[-1]}_{OUR_MODEL_ARGS.revision}/{BENCHMARK}/gt_edited_{current_uniq_id}.png'
                grounding_image_path = f'{SAVE_DIR}/results/{baseline}/{OUR_MODEL_ARGS.model_path.split("/")[-1]}_{OUR_MODEL_ARGS.revision}/{BENCHMARK}/grounding_{current_uniq_id}.png'
                
                
                overview = plot_overview(
                    batch_input_images[batch_sample_idx], 
                    batch_edited_images[batch_sample_idx], 
                    batch_edit_instructions_modified[batch_sample_idx]
                )
                overview.save(overview_path)
                smart_resize(batch_input_images[batch_sample_idx], 256*256).save(input_image_path)
                batch_edited_images[batch_sample_idx].save(edited_image_path)
                # smart_resize(batch_gt_edited_images[batch_sample_idx], 256*256).save(gt_edited_image_path)
                batch_grounding_images[batch_sample_idx].save(grounding_image_path)
                
                batch_overview_paths.append(overview_path)
                batch_input_image_paths.append(input_image_path)
                batch_edited_image_paths.append(edited_image_path)
                # batch_gt_edited_image_paths.append(gt_edited_image_path)
                # if baseline == "OurModel":
                batch_grounding_image_paths.append(grounding_image_path)


            print("Here at 357")
            for batch_sample_idx in tqdm(range(len(batch_input_images)), desc="Processing batches for wandb"):
                # Retrieve saved paths and other data for this sample
                overview_path = batch_overview_paths[batch_sample_idx]
                input_image_path = batch_input_image_paths[batch_sample_idx]
                edited_image_path = batch_edited_image_paths[batch_sample_idx]
                # gt_edited_image_path = batch_gt_edited_image_paths[batch_sample_idx]
                # if baseline == "OurModel":
                grounding_image_path = batch_grounding_image_paths[batch_sample_idx]

                wandb_report_row = {
                    "overview": overview_path, 
                    "original_image": input_image_path,
                    "edit_instruction": batch_edit_instructions[batch_sample_idx],
                    # "gt_reasoning": batch_reasoning_super_concise[batch_sample_idx],
                    # "coord": batch_coords[batch_sample_idx] if eval_split == "train" else '',
                    "reasoning": batch_generated_cot[batch_sample_idx],
                    "edited_image": edited_image_path,
                    "grounding_image": grounding_image_path,
                    "task": batch_task_types[batch_sample_idx],
                    "id": batch_idxs[batch_sample_idx],
                    # 'gt_edited_image': gt_edited_image_path,
                    # 'id': batch_omniedit_id[batch_sample_idx],
                    # 'task': batch_task_type[batch_sample_idx],
                }
                wandb_batch_table_rows.append(
                    [
                        wandb.Image(overview_path), # Log path instead of image object
                        wandb.Image(input_image_path), 
                        batch_edit_instructions[batch_sample_idx], 
                        # batch_reasoning_super_concise[batch_sample_idx],
                        # batch_coords[batch_sample_idx] if eval_split == "train" else '',
                        batch_generated_cot[batch_sample_idx],
                        wandb.Image(edited_image_path), 
                        wandb.Image(grounding_image_path) if grounding_image_path else None,
                        batch_task_types[batch_sample_idx],
                        batch_idxs[batch_sample_idx],
                        # wandb.Image(gt_edited_image_path),
                        # wandb_report_row['id'],
                        # wandb_report_row['task'],
                    ]
                )

                wandb_report = wandb_report._append(wandb_report_row, ignore_index=True)
                    
                # Make sure uniq_id is incremented correctly after processing all samples in the batch
                if batch_sample_idx == len(batch_input_images) - 1:
                    uniq_id += len(batch_input_images)

            columns = ["overview", "original_image", "edit_instruction", "reasoning", "edited_image", "grounding_image", "task", "id"]
            # columns = ["overview", "original_image", "edit_instruction", "gt_cot", "coord", "reasoning", "edited_image",
            #             "id", "task"]

            wandb_batch_table_data = wandb.Table(data=wandb_batch_table_rows, columns=columns)
            run.log({EVAL_TABLE: wandb_batch_table_data}, commit=True)
            wandb_batch_table_rows = [] # Reset for the next batch

            # Clear memory after each batch
            torch.cuda.empty_cache()     # Clear CUDA cache
            gc.collect()
            del llm
            cleanup()


    json_path = f'{SAVE_DIR}/eval_report_{baseline}_{OUR_MODEL_ARGS.model_path.split("/")[-1]}_{OUR_MODEL_ARGS.revision}_{BENCHMARK}.json'
    wandb_report.to_json(json_path, orient='records', lines=True)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/network/scratch/a/ankur.sikarwar/image_editing_baselines_automatic", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size to use for evaluation")
    parser.add_argument("--mode", type=str, default="E", help="Mode to use for evaluation")
    # parser.add_argument("--split", type=str, default="train", help="Split to use for evaluation")
    # parser.add_argument("--size", type=str, default="500", help="Size to use for evaluation")
    parser.add_argument("--model_paths", nargs="+", help="Model paths to use for evaluation")
    parser.add_argument("--reasoning_input", type=str, default="False", help="Whether to use reasoning input")
    parser.add_argument("--revision", nargs="+", help="Revision to use for evaluation")
    parser.add_argument("--use_revision", type=str, default="False", help="Whether to use revision")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    SAVE_DIR = args.save_dir


    for model_path in tqdm(args.model_paths, desc="Evaluating models"):
        print(f"Evaluating {model_path}")
        if args.use_revision == "True":
            for revision in args.revision:
                print(f"Revision: {revision}")

                OUR_MODEL_ARGS = Namespace(
                    mode=args.mode,
                    model_path=model_path, 
                    revision=revision, 
                    cot_keys=["source_image_grounding_information", "conditioning_information"], 
                    model_location="HF", 
                    split=None, 
                    save_dir=None, 
                    max_new_tokens=3000, 
                    image_area=65636, 
                    use_logit_processor=True,
                    k_sample=1,
                    top_k=2048, 
                    device="cuda:0", 
                    reasoning_input=args.reasoning_input
                )

                print(f"Evaluating {model_path}")
                # print(f"Split: {args.split}")
                # print(f"Size: {args.size}")
                print("OUR_MODEL_ARGS: ", OUR_MODEL_ARGS)

                eval_automatic(OUR_MODEL_ARGS=OUR_MODEL_ARGS)

        elif args.use_revision == "False":
            OUR_MODEL_ARGS = Namespace(
                mode=args.mode,
                model_path=model_path, 
                revision="main", 
                cot_keys=["source_image_grounding_information", "conditioning_information"], 
                model_location="HF", 
                split=None, 
                save_dir=None, 
                max_new_tokens=3000, 
                image_area=65636, 
                use_logit_processor=True,
                k_sample=1,
                top_k=2048, 
                device="cuda:0", 
                reasoning_input=args.reasoning_input
            )

            print(f"Evaluating {model_path}")
            # print(f"Split: {args.split}")

            eval_automatic(OUR_MODEL_ARGS=OUR_MODEL_ARGS)
