# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import os
import os.path as osp
import pathlib

import transformers as tf
import torch

import re
from typing import List, Optional
from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM
from emu3.train_image_editing.my_datasets import Emu3FeatureDataset

import wandb

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")

@dataclass
class DataArguments:
    data_paths: List[str] = field(default_factory=list)
    #pre-tokenize validation set for faster validation
    validation_paths: List[str] = field(default_factory=list)
    validation_size: Optional[int] = field(default=50)
    coefficients: List[float] = field(default_factory=list)
    # null_prompt_prob: float = field(default=0.05)
    # apply_loss_on_only_vision: bool = field(default=False)
    # apply_loss_on_only_text: bool = field(default=False)
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)
    cot_keys: List[str] = field(default_factory=list) 
    random_seed: int = field(default=42)

@dataclass
class TrainingArguments(tf.TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)
    run_name: Optional[str] = field(default="Emu3-SFT")
    evaluation_strategy: Optional[str] = field(default="epoch")


def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)


def convert_to_dict(obj):
    return {k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v)) for k, v in vars(obj).items()}

def train():
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize model config and update with training args
    model_config = Emu3Config.from_pretrained(model_args.model_name_or_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate

    # Setup wandb environment
    os.environ["WANDB_DIR"] = training_args.output_dir
    os.environ["WANDB_PROJECT"] = "Emu3-Base-SFT"

    # Initialize model
    try:
        model = Emu3ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
            torch_dtype=torch.bfloat16 if training_args.bf16 else None
        )
        model.config.use_cache = False
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {str(e)}")

    # Initialize tokenizer
    try:
        tokenizer = Emu3Tokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}")

    # Initialize datasets
    train_dataset = Emu3FeatureDataset(data_args, validation=False, tokenizer=tokenizer)
    val_dataset = Emu3FeatureDataset(data_args, validation=True, tokenizer=tokenizer)
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    run_id_path = f"{training_args.output_dir}/wandb_run_id.txt"

    # Initialize wandb only on main process
    if training_args.local_rank in [-1, 0]:
        print('training_args: ', training_args)
        checkpoint_paths = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        
        if checkpoint_paths:
            # Extract the numeric suffix from each checkpoint name and sort numerically
            def extract_checkpoint_number(path):
                match = re.search(r"checkpoint-(\d+)", path.name)
                return int(match.group(1)) if match else -1  # Default to -1 if no match
            newest_checkpoint = max(checkpoint_paths, key=extract_checkpoint_number)
            if newest_checkpoint and newest_checkpoint.exists():
                with open(run_id_path, "r") as f:
                    run_id = f.read().strip()
                wandb.init(
                    project=os.environ["WANDB_PROJECT"],
                    name=training_args.run_name,
                    dir=os.environ["WANDB_DIR"],
                    resume="must",
                    id=run_id
                )
                wandb.config.update({"model/num_parameters": 266240}, allow_val_change=True)
        else:
            wandb.init(
                project=os.environ["WANDB_PROJECT"],
                name=training_args.run_name,
                dir=os.environ["WANDB_DIR"]
            )
            run_id = wandb.run.id
            with open(run_id_path, "w") as f:
                f.write(run_id)
            wandb.config.update({
                "model_args": convert_to_dict(model_args),
                "data_args": convert_to_dict(data_args),
                "training_args": convert_to_dict(training_args),
            })
    else:
        # For non-master processes, disable wandb logging.
        wandb.init(mode="disabled")
    
    # Initialize trainer
    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=Emu3FeatureDataset.collate_fn
    )
    
    # Find newest checkpoint if exists
    checkpoint_paths = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoint_paths:
        # Extract the numeric suffix from each checkpoint name and sort numerically
        def extract_checkpoint_number(path):
            match = re.search(r"checkpoint-(\d+)", path.name)
            return int(match.group(1)) if match else -1  # Default to -1 if no match

        # Find the checkpoint with the highest number
        newest_checkpoint = max(checkpoint_paths, key=extract_checkpoint_number)
        print(f"Resuming from checkpoint: {newest_checkpoint}")
        if newest_checkpoint and newest_checkpoint.exists():
            trainer.train(resume_from_checkpoint=newest_checkpoint)
        else:
            print("*********No valid checkpoint found. Starting training from scratch.*********")
            trainer.train()
    else:
        print("No checkpoint found. Starting training from scratch.")
        trainer.train()

    # Save final model state
    trainer.save_state()
    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()