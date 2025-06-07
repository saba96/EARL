"""
This code is heavily based on the nano-aha-moment repository:
https://github.com/McGill-NLP/nano-aha-moment
"""

import fnmatch
import gc
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import _jsonnet
import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from deepspeed import DeepSpeedEngine
from deepspeed.runtime.utils import see_memory_usage
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    Emu3Config,
    Emu3ForCausalLM,
    HfArgumentParser,
    PreTrainedModel,
)
from utils import (
    clean_up_checkpoints,
    compute_token_log_probs,
    dump_episodes,
    find_last_checkpoint,
    get_api_base_from_github,
    initialize_distributed_training_pg,
    load_model_into_vllm,
)
from vllm import LLM, SamplingParams

import wandb
from evaluation.vllm_processing_emu3 import (
    CachedPrefixConstrainedLogitsProcessor,
    Emu3Processor,
)
from rl.data import (
    Emu3PPOCollator,
    InfiniteReplacementBatchSampler,
    PreTokenizedEmu3Dataset,
)
from rl.reward import ImageEditingRewardFunction

os.environ["VLLM_USE_V1"] = "0"

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


@dataclass
class TrainingArgs:
    """Arguments pertaining to training configuration."""

    # Required parameters (no default values)
    # Dataset and data paths
    train_paths_file: List[str] = field(
        metadata={"help": "Path to the file containing the list of data point paths"},
    )
    val_paths_file: List[str] = field(
        metadata={"help": "Path to the file containing the list of data point paths"},
    )
    base_exp_dir: str = field(
        metadata={"help": "Path to the base directory for the experiment"},
    )

    # Model and tokenizer paths
    model_path: str = field(
        metadata={"help": "Path to the model"},
    )
    tokenizer_path: str = field(
        metadata={"help": "Path to the tokenizer"},
    )
    assistant_prefill: str = field(
        metadata={"help": "Text to prefill for assistant responses"},
    )

    # Training hyperparameters
    learning_rate: float = field(metadata={"help": "Learning rate for training"})
    num_iterations: int = field(metadata={"help": "Total number of training iterations"})
    kl_coeff: float = field(metadata={"help": "KL coefficient for PPO"})
    temperature: float = field(metadata={"help": "Temperature for sampling"})
    per_device_batch_size: int = field(metadata={"help": "Batch size for each GPU device during training"})

    # Generation parameters
    episodes_per_iteration: int = field(
        metadata={"help": "Number of episodes to collect per iteration for training"},
    )
    generations_per_sample: int = field(
        metadata={"help": "Number of responses to generate for each input prompt"},
    )
    max_response_tokens: int = field(
        metadata={"help": "Maximum number of tokens to generate in each response"},
    )
    top_p: float = field(metadata={"help": "Nucleus sampling parameter (1.0 = disabled)"})
    top_k: int = field(metadata={"help": "Top-k sampling parameter (-1 = disabled)"})
    model_context_size: int = field(
        metadata={"help": "Maximum context size (sequence length) for the model"},
    )

    # vLLM configuration
    vllm_gpu_memory_utilization: float = field(
        metadata={"help": "vLLM GPU memory utilization ratio (0.0 to 1.0)"},
    )

    random_seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    coefficients: List[float] = field(default_factory=list)
    val_coefficients: List[float] = field(default_factory=list)

    model_revision: str = field(
        default=None,
        metadata={"help": "Model revision to use"},
    )

    # Optional parameters (with default values)
    # Token length constraints
    max_prompt_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length of the prompt"},
    )
    max_completion_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length of the completion"},
    )

    # Emu3 specific settings
    emu3_use_logit_processor: bool = field(
        default=True,
        metadata={"help": "Whether to use logit processor for Emu3"},
    )
    separate_language_and_vision_vocabs: bool = field(
        default=True,
        metadata={"help": "Whether to use separate language and vision vocabs"},
    )
    train_text_tokens_only: bool = field(
        default=False,
        metadata={"help": "Whether to train on text tokens only"},
    )
    text_only_cycle_steps: int = field(
        default=10,
        metadata={"help": "Number of steps in each text-only training cycle"},
    )
    text_only_steps: int = field(
        default=10,
        metadata={"help": "Number of steps to train on text in each cycle"},
    )
    train_vision_tokens_only: bool = field(
        default=False,
        metadata={"help": "Whether to train on vision tokens only"},
    )
    vision_only_cycle_steps: int = field(
        default=10,
        metadata={"help": "Number of steps in each vision-only training cycle"},
    )
    vision_only_steps: int = field(
        default=10,
        metadata={"help": "Number of steps to train on vision in each cycle"},
    )

    # Reward function configuration
    reward_function_api_base: Optional[str] = field(
        default=None,
        metadata={"help": "Base URL for the OpenAI API"},
    )
    reward_function_api_model_name: str = field(
        default="Qwen/Qwen2.5-VL-72B-Instruct",
        metadata={"help": "Name of the OpenAI API model"},
    )
    reward_compute_viescore: bool = field(
        default=True,
        metadata={"help": "Whether to compute VIEScore"},
    )
    reward_compute_ground_score: bool = field(
        default=True,
        metadata={"help": "Whether to compute GroundScore"},
    )
    reward_final_reward: str = field(
        default="viescore",
        metadata={
            "help": "How to compute final reward",
            "choices": ["viescore", "ground_score", "sqrt(viescore_pqxground_score)"],
        },
    )

    # Logging and checkpointing
    log_episodes_every_n_iterations: int = field(
        default=20,
        metadata={"help": "Log episodes every n iterations"},
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the run for logging and checkpointing"},
    )
    keep_checkpoints_every_n_iterations: Optional[int] = field(
        default=None,
        metadata={"help": "Number of iterations to keep checkpoints"},
    )
    save_checkpoints_every_n_iterations: Optional[int] = field(
        default=None,
        metadata={"help": "Number of iterations to save checkpoints"},
    )
    push_to_hub_every_n_iterations: Optional[int] = field(
        default=None,
        metadata={"help": "Number of iterations to push to hub"},
    )
    push_to_hub_repo_id: Optional[str] = field(
        default=None,
        metadata={"help": "Repository ID to push to hub"},
    )

    def __post_init__(self):
        if str(self.reward_function_api_base).lower() == "none":
            self.reward_function_api_base = None
        if str(self.run_name).lower() == "none":
            self.run_name = None


class Emu3RLProcessor:
    def __init__(
        self,
        args: TrainingArgs,
        tokenizer: AutoTokenizer,
        device: torch.device,
        hf_repo_id: str = "BAAI/Emu3-VisionTokenizer",
    ):
        self.use_logit_processor = args.emu3_use_logit_processor
        self.tokenizer = tokenizer

        self.image_processor = AutoImageProcessor.from_pretrained(hf_repo_id, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(hf_repo_id, trust_remote_code=True).cpu().eval()
        self.processor = Emu3Processor(
            image_processor=self.image_processor, vision_tokenizer=self.image_tokenizer, tokenizer=self.tokenizer
        )

        # Compile regex patterns for better performance
        self.image_token_pattern = re.compile(
            rf"({re.escape(self.tokenizer.img_token)}.*?{re.escape(self.tokenizer.eoi_token)})"
        )
        self.vision_token_pattern = re.compile(r"<\|visual token (\d+)\|>")
        self.eol_pattern = re.compile(re.escape(self.tokenizer.eol_token))
        self.device = device
        self.image_prefix_template = "{H}*{W}"
        self.sampling_params_dict = create_sampling_params_dict(args)

        vision_token_template = self.processor.visual_template[0]
        self.vision_start_index, self.vision_end_index = self.tokenizer.encode(
            [
                vision_token_template.format(token_id=0),
                vision_token_template.format(token_id=self.image_tokenizer.config.codebook_size - 1),
            ]
        )

    def get_vision_start_index(self):
        return self.vision_start_index

    @torch.no_grad()
    def decode_to_images(self, completions: List[str]) -> List[Optional[Image.Image]]:
        self.image_tokenizer.to(self.device)

        out_images = []

        for compl in completions:
            match = self.image_token_pattern.search(compl)
            if not match:
                out_images.append(None)
                continue

            image_str = match.group(1)
            try:
                # Split image string into rows using compiled pattern
                image_rows = self.eol_pattern.split(image_str)

                # Process each row to extract vision tokens
                vision_ids_2d = []
                for row in image_rows:
                    token_ids = self.vision_token_pattern.findall(row)
                    if token_ids:
                        row_tokens = [int(token_id) for token_id in token_ids]
                        vision_ids_2d.append(row_tokens)

                image = torch.tensor(vision_ids_2d, dtype=torch.long, device=self.image_tokenizer.device)
                image = self.image_tokenizer.decode(image[None]).float()
                image = self.image_processor.postprocess(image)["pixel_values"][0]
                out_images.append(image)
            except Exception as e:
                logger.warning(f"Failed to decode image from completion: {str(e)}")
                out_images.append(None)

        self.image_tokenizer.cpu()

        assert len(out_images) == len(completions)
        return out_images

    def decode_to_images_from_token_ids(self, image_token_ids: List[np.ndarray]) -> List[Optional[Image.Image]]:
        self.image_tokenizer.to(self.device)

        out_images = []

        for vision_ids_2d in image_token_ids:
            image = torch.tensor(vision_ids_2d, dtype=torch.long, device=self.image_tokenizer.device)
            image = self.image_tokenizer.decode(image[None]).float()
            image = self.image_processor.postprocess(image)["pixel_values"][0]
            out_images.append(image)

        self.image_tokenizer.cpu()

        assert len(out_images) == len(image_token_ids)
        return out_images

    def prepare_initial_inputs_for_vllm(
        self,
        samples: List[Dict[str, Any]],
        stop_token_ids: Optional[List[int]] = None,
        stop: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Prepare inputs for episode generation.
        """
        inputs = []
        sampling_params = []

        if stop_token_ids is None and stop is None:
            stop_token_ids = [self.tokenizer.eos_token_id]

        for sample in samples:
            prompt_ids = sample["prompt_ids"].tolist()

            logits_processors = None
            if self.use_logit_processor:
                h, w = sample["original_image"].shape[:2]
                constrained_fn = self.processor.build_prefix_constrained_fn(np.array([h]), np.array([w]))
                logits_processors = [CachedPrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1)]

            inputs.append({"prompt_token_ids": prompt_ids})
            sampling_params.append(
                SamplingParams(
                    **self.sampling_params_dict,
                    logits_processors=logits_processors,
                    stop_token_ids=stop_token_ids,
                    include_stop_str_in_output=True,
                    stop=stop,
                    detokenize=False,
                )
            )

        return (inputs,), {"sampling_params": sampling_params}

    def compute_image_mask(self, token_ids: List[int]) -> np.ndarray:
        vision_token_range = range(self.vision_start_index, self.vision_end_index + 1)
        return np.array([tok_id in vision_token_range for tok_id in token_ids], dtype=np.int32)


def create_datasets(args, tokenizer, rank, world_size):
    """
    Create training and validation datasets.

    Args:
        args: Training arguments
        tokenizer: Tokenizer object
        rank: Process rank for distributed training
        world_size: Total number of processes

    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    # Initialize arguments (args) and paths
    random.seed(args.random_seed)

    filelist_train = []
    filelist_val = []

    dataset_sizes = {}
    dataset_coefficients = {}
    dataset_coefficients_val = {}

    # Handle datasets (both training and validation sets)
    logger.info(f"args.coefficients: {args.coefficients}")

    if args.coefficients == []:
        args.coefficients = [None] * len(args.data_paths)
        logger.info("****Important: Upsampling all to size of the largest dataset****")

    # Process training datasets
    logger.info(f"args.train_paths_file: {args.train_paths_file}")
    logger.info(f"args.coefficients: {args.coefficients}")
    logger.info(f"args.val_paths_file: {args.val_paths_file}")
    for path, coefficient in zip(args.train_paths_file, args.coefficients):
        with open(path) as f:
            d = json.load(f)
            prefix = str(Path(path).parent.parent)
            files = d["path_list"]
            # files = list(set(files))
            dataset_sizes[prefix] = len(files)
            dataset_coefficients[prefix] = coefficient

    # If no coefficients are provided, set them based on the largest dataset
    if all(coef is None for coef in dataset_coefficients.values()):
        max_size = max(dataset_sizes.values())
        dataset_coefficients = {prefix: float(max_size) / size for prefix, size in dataset_sizes.items()}
    else:
        dataset_coefficients = {
            prefix: coef if coef is not None else 1.0 for prefix, coef in dataset_coefficients.items()
        }

    for path, coefficient in zip(args.val_paths_file, args.val_coefficients):
        with open(path) as f:
            d = json.load(f)
            prefix = str(Path(path).parent.parent)
            files = d["path_list"]
            # files = list(set(files))
            dataset_sizes[prefix] = len(files)
            dataset_coefficients_val[prefix] = coefficient

    # Initialize filelists for both training and validation
    filelist_train = []
    filelist_val = []

    for path in args.train_paths_file:
        is_validation = False  # It's explicitly not validation
        prefix = None
        with open(path) as f:
            d = json.load(f)
            prefix = str(Path(path).parent.parent)
            files = d["path_list"]
            # files = list(set(files))

        coef = dataset_coefficients.get(prefix, 1.0)
        sampled_files = []

        if coef >= 1:
            sampled_files.extend([os.path.join(prefix, f) for f in files] * int(coef))
            remainder = coef - int(coef)
            if remainder > 0:
                sample_size = int(len(files) * remainder)
                sampled_files.extend(random.sample([os.path.join(prefix, f) for f in files], sample_size))
        elif coef < 1:
            sample_size = int(len(files) * coef)
            sampled_files = random.sample([os.path.join(prefix, f) for f in files], sample_size)
        else:
            sampled_files = []
        # sampled_files = os.path.join(prefix, *sampled_files)
        filelist_train.extend(sampled_files)
        logger.info(
            f"Training Dataset: {prefix}, Original Size: {dataset_sizes[prefix]}, Sampled Size: {len(sampled_files)}"
        )

    # Separate processing for validation paths
    for path in args.val_paths_file:
        is_validation = True  # It's explicitly for validation
        prefix = None
        with open(path) as f:
            d = json.load(f)
            prefix = str(Path(path).parent.parent)
            files = d["path_list"]
            # files = [os.path.join(prefix, file) for file in files]
            # files = list(set(files))

        coef = dataset_coefficients_val.get(prefix, 1.0)
        sampled_files = []

        if coef >= 1:
            sampled_files.extend([os.path.join(prefix, f) for f in files] * int(coef))
            remainder = coef - int(coef)
            if remainder > 0:
                sample_size = int(len(files) * remainder)
                sampled_files.extend(random.sample([os.path.join(prefix, f) for f in files], sample_size))
        elif coef < 1:
            sample_size = int(len(files) * coef)
            sampled_files = random.sample([os.path.join(prefix, f) for f in files], sample_size)
        else:
            sampled_files = []
        # sampled_files = os.path.join(prefix, *sampled_files)
        filelist_val.extend(sampled_files)
        logger.info(
            f"Validation Dataset: {prefix}, Original Size: {dataset_sizes[prefix]}, Sampled Size: {len(sampled_files)}"
        )

    # Shuffle both training and validation sets after sampling
    random.shuffle(filelist_train)
    random.shuffle(filelist_val)

    # Final dataset paths for training and validation
    train_data_paths = filelist_train
    validation_data_paths = filelist_val

    # Log the sizes
    logger.info(f"Training set size: {len(train_data_paths)}")
    logger.info(f"Validation set size: {len(validation_data_paths)}")

    # Ensure no duplicate paths in training data
    # orig_train_len = len(train_data_paths)
    # unique_train_data_paths = list(set(train_data_paths))
    # if len(unique_train_data_paths) != orig_train_len:
    #     logger.warning(
    #         f"Found {orig_train_len - len(unique_train_data_paths)} duplicate paths in training data. "
    #         f"Using {len(unique_train_data_paths)} unique paths."
    #     )
    #     unique_train_data_paths.sort()
    #     random.Random(42).shuffle(unique_train_data_paths)
    #     train_data_paths = unique_train_data_paths

    # Define the training and validation datasets
    train_dataset = PreTokenizedEmu3Dataset(
        data_paths=train_data_paths,
        data_parallel_rank=rank,
        data_parallel_world_size=world_size,
        tokenizer=tokenizer,
        assistant_prefill=args.assistant_prefill,
    )

    logger.info(f"Training set size per each rank: {len(train_dataset)}")

    validation_dataset = PreTokenizedEmu3Dataset(
        data_paths=validation_data_paths,
        data_parallel_rank=rank,
        data_parallel_world_size=world_size,
        tokenizer=tokenizer,
        assistant_prefill=args.assistant_prefill,
    )

    return train_dataset, validation_dataset


def create_sampling_params_dict(args: TrainingArgs) -> Dict[str, Any]:
    sampling_params_dict = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_response_tokens,
        "n": args.generations_per_sample,
    }
    return sampling_params_dict


def create_training_episodes(
    samples: List[Dict[str, Any]],
    processor: Emu3RLProcessor,
    inference_engine: LLM,
    reward_function: ImageEditingRewardFunction,
    tokenizer: AutoTokenizer,
    args: TrainingArgs,
    iteration: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    #########################################################
    # Generate completions
    #########################################################

    gen_args, gen_kwargs = processor.prepare_initial_inputs_for_vllm(samples)
    outputs = inference_engine.generate(*gen_args, **gen_kwargs)
    all_prompt_ids = [np.array(out.prompt_token_ids) for out in outputs for _ in out.outputs]
    all_completion_ids = [list(g.token_ids) for out in outputs for g in out.outputs]
    all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]
    all_completions = tokenizer.batch_decode(all_completion_ids, skip_special_tokens=False)
    all_samples = [smpl for smpl in samples for _ in range(args.generations_per_sample)]

    assert len(all_completion_ids) == len(all_finish_reasons) == len(all_samples) == len(all_prompt_ids)
    assert len(all_completion_ids) == len(samples) * args.generations_per_sample

    inference_engine.sleep(1)

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    #########################################################
    # Compute rewards
    #########################################################

    all_rewards: List[Dict[str, float]] = reward_function(
        completions=all_completions, finish_reasons=all_finish_reasons, samples=all_samples
    )
    assert len(all_rewards) == len(all_completions) == len(all_finish_reasons) == len(all_samples)

    #########################################################
    # Process rewards and create episodes
    #########################################################

    groups = [
        list(range(i, i + args.generations_per_sample))
        for i in range(0, len(all_completions), args.generations_per_sample)
    ]  # example: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    all_advantages, all_completion_image_masks = [], []

    metrics = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
        "has_let_s_think": ["let's think step".lower() in c.lower() for c in all_completions],
        **{f"reward_metrics/{k}": [] for k in all_rewards[0].keys() if k != "final_reward"},
    }

    for group_indices in groups:
        grp_completion_ids = [all_completion_ids[i] for i in group_indices]
        grp_finish_reasons = [all_finish_reasons[i] for i in group_indices]
        grp_rewards = [all_rewards[i] for i in group_indices]

        #########################################################
        # Advantage computation
        #########################################################

        rewards = np.array([r["final_reward"] for r in grp_rewards])
        per_seq_advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        per_token_advantages = [
            adv * np.ones(len(compl)) for adv, compl in zip(per_seq_advantages, grp_completion_ids)
        ]

        all_advantages.extend(per_token_advantages)
        all_completion_image_masks.extend([processor.compute_image_mask(ids) for ids in grp_completion_ids])

        metrics["rewards"].extend(rewards)
        metrics["non_stop_rate"].extend([fr != "stop" for fr in grp_finish_reasons])
        metrics["response_lengths"].extend([len(ids) for ids in grp_completion_ids])
        for reward in grp_rewards:
            for k, v in reward.items():
                if k == "final_reward":
                    continue
                metrics[f"reward_metrics/{k}"].append(v)

    assert (
        len(all_completion_ids)
        == len(all_completion_image_masks)
        == len(all_advantages)
        == len(all_prompt_ids)
        == len(all_samples)
    )

    episodes = {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "completion_image_masks": all_completion_image_masks,
        "advantages": all_advantages,
        "samples": all_samples,
        "rewards": all_rewards,
    }

    return episodes, metrics


def compute_pg_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    processor: Emu3RLProcessor,
    batch: Dict[str, torch.Tensor],
    total_response_len: torch.Tensor,
    args: TrainingArgs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    completion_mask = batch["completion_mask"].float()  # [batch_size, seq_len]
    advantages = batch.pop("advantages")  # [batch_size, seq_len]
    ref_logps = batch.pop("ref_log_probs")  # [batch_size, seq_len]

    logps = compute_token_log_probs(
        model=policy_model,
        inputs=batch,
        temperature=args.temperature,
        separate_language_and_vision_vocabs=args.separate_language_and_vision_vocabs,
        vision_start_index=processor.get_vision_start_index(),
    )  # [batch_size, seq_len]

    ref_logratio = ref_logps - logps
    kl_penalty = torch.exp(ref_logratio) - 1 - ref_logratio  # [batch_size, seq_len]
    kl_penalty = kl_penalty * completion_mask  # [batch_size, seq_len]

    with torch.no_grad():
        kl_penalty_2 = ((logps - ref_logps) * completion_mask).sum() / total_response_len

    entropy = -logps.sum() / completion_mask.sum()  # scalar

    policy_loss = -logps * advantages  # [batch_size, seq_len]
    policy_loss = policy_loss * completion_mask  # [batch_size, seq_len]

    loss = (policy_loss + args.kl_coeff * kl_penalty).sum() / total_response_len  # scalar

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len,
        "kl_penalty": kl_penalty.sum().item() / total_response_len,
        "kl_penalty_2": kl_penalty_2.item(),
        "avg_ref_logps": ref_logps.sum() / total_response_len,
        "avg_logps": logps.sum().item() / total_response_len,
        "entropy": entropy.item() / total_response_len,
    }

    return loss, metrics


def main():
    parser = HfArgumentParser(TrainingArgs)

    rank = int(os.environ.get("RANK", "0"))
    nproc = int(os.environ.get("WORLD_SIZE", "1"))

    if len(sys.argv) >= 2 and (sys.argv[1].endswith(".json") or sys.argv[1].endswith(".jsonnet")):
        # Parse config file to get arguments
        config_file = os.path.abspath(sys.argv[1])
        if config_file.endswith(".jsonnet"):
            # Parse Jsonnet file with environment variables
            ext_vars = {k: v for k, v in os.environ.items() if k.startswith("APP__")}
            json_str = _jsonnet.evaluate_file(config_file, ext_vars=ext_vars)
            args_dict = json.loads(json_str)
            args = parser.parse_dict(args_dict)[0]
        else:
            args = parser.parse_json_file(config_file)[0]
        if args.run_name is None:
            args.run_name = os.path.splitext(os.path.basename(config_file))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]
        if not args.run_name:
            raise ValueError("Run name is required")

    if int(os.environ.get("RANK", "0")) == 0:
        logger.info(f"Args: {args}")

    initialize_distributed_training_pg(rank, nproc)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    curr_cuda_device = torch.device("cuda")
    logger.info(f"Rank: {rank}, World size: {world_size}, Device: cuda:{torch.cuda.current_device()}")

    # Disable logging for non-main processes to avoid duplicate logs
    if dist.get_rank() != 0:
        logger.setLevel(logging.ERROR)

    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        logger.info("HF Hub is offline, will not push to hub")
        args.push_to_hub_every_n_iterations = None

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    processor = Emu3RLProcessor(args, tokenizer, device=curr_cuda_device)

    if args.reward_function_api_base is None:
        reward_api_base = lambda: f"http://{get_api_base_from_github()}:4877/v1"
    else:
        if os.path.exists(args.reward_function_api_base):
            reward_api_base = lambda: f"http://{open(args.reward_function_api_base, 'r').read().strip()}:4877/v1"
        else:
            reward_api_base = args.reward_function_api_base

    reward_function = ImageEditingRewardFunction(
        api_base=reward_api_base,
        api_model_name=args.reward_function_api_model_name,
        processor=processor,
        enable_logging=dist.get_rank() == 0,
        args=args,
    )

    collator = Emu3PPOCollator(tokenizer, args.max_prompt_len, args.max_completion_len)
    train_dataset, _ = create_datasets(args, tokenizer, rank, world_size)

    # Load checkpoint if it exists
    base_exp_dir = Path(args.base_exp_dir)
    exp_dir = base_exp_dir / args.run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Logs and Checkpoints will be saved to: {exp_dir}")

    begin_iter = 0
    ckpt_path, ckpt_iter = find_last_checkpoint(exp_dir)
    if ckpt_path is not None:
        logger.info(f"Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
        begin_iter = ckpt_iter + 1

    local_episodes_per_iteration = args.episodes_per_iteration // world_size
    samples_per_iteration = local_episodes_per_iteration // args.generations_per_sample
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=InfiniteReplacementBatchSampler(
            len(train_dataset),
            batch_size=samples_per_iteration,
            generator=torch.Generator().manual_seed(42),
            skip_iterations=max(0, begin_iter - 1),
        ),
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=lambda x: x,
    )
    train_iter = iter(train_dataloader)
    _ = next(train_iter)  # Kick off the dataloader to warm up

    # DeepSpeed configuration
    train_batch_size = args.episodes_per_iteration
    grad_accum_steps = train_batch_size // args.per_device_batch_size // world_size

    deepspeed_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2, "overlap_comm": False},
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": args.per_device_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "torch_adam": True,
                "fused": True,
            },
        },
    }
    ref_deepspeed_config = {
        "bf16": {"enabled": True},
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": args.per_device_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,  # No effect
    }

    dist.barrier(device_ids=[torch.cuda.current_device()])

    ############################################
    # Initialize Models
    ############################################

    see_memory_usage("Before initializing models", force=dist.get_rank() == 0)

    model_class = Emu3ForCausalLM
    config_class = Emu3Config

    # Disable dropout
    logger.info(f"Loading policy and reference from `{args.model_path}`")
    DROPOUT_CONFIG_KEYS = [
        "dropout",
        "attention_dropout",
        "classifier_dropout",
        "hidden_dropout",
        "activation_dropout",
        "resid_pdrop",
        "embd_pdrop",
        "attn_pdrop",
    ]
    model_kwargs = {}
    model_config = config_class.from_pretrained(
        args.model_path, revision="main" if args.model_revision is None else args.model_revision
    )
    for key in DROPOUT_CONFIG_KEYS:
        if hasattr(model_config, key):
            model_kwargs[key] = 0.0
    if len(model_kwargs) > 0:
        logger.info(f"Disabled dropout keys: {model_kwargs}")

    policy_model = model_class.from_pretrained(
        args.model_path,
        revision="main" if args.model_revision is None else args.model_revision,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
        **model_kwargs,
    )
    reference_model = model_class.from_pretrained(
        args.model_path,
        revision="main" if args.model_revision is None else args.model_revision,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
        **model_kwargs,
    )
    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    see_memory_usage("Before initializing DeepSpeed engines", force=dist.get_rank() == 0)

    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=deepspeed_config,
        model_parameters=policy_model.parameters(),
    )

    reference_model, *_ = deepspeed.initialize(
        model=reference_model,
        config=ref_deepspeed_config,
    )

    reference_model.module.cpu()

    ############################################
    # Initialize vLLM (Inference) engine
    ############################################

    see_memory_usage("Before initializing vLLM", force=dist.get_rank() == 0)
    if dist.get_rank() != 0:
        # Disable root vllm logger for non-main ranks
        vllm_logger = logging.getLogger("vllm")
        vllm_logger.setLevel(logging.ERROR)

    inference_engine = LLM(
        model=args.model_path,
        revision=args.model_revision,
        tokenizer=args.tokenizer_path,
        skip_tokenizer_init=True,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enable_prefix_caching=True,
        swap_space=4,
        dtype=torch.bfloat16,
        max_model_len=args.model_context_size,
        enable_sleep_mode=True,
        trust_remote_code=True,
        device=f"cuda:{torch.cuda.current_device()}",
        tensor_parallel_size=1,
    )

    see_memory_usage("After initializing vLLM", force=dist.get_rank() == 0)

    if dist.get_rank() == 0:
        wandb.init(name=args.run_name, config=vars(args), resume="allow")
        parent_dir = Path(__file__).parent.absolute()
        wandb.run.log_code(parent_dir)

    if ckpt_path is not None:
        logger.info(f"Loading checkpoint {ckpt_path} at iteration {ckpt_iter}")
        out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
        if out is None:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}")
        load_model_into_vllm(policy_model, inference_engine)

    for iteration in trange(begin_iter, args.num_iterations):
        logger.info(f"Iteration {iteration}/{args.num_iterations}")

        metrics = {}

        #########################################################
        # Generate Episodes
        #########################################################
        gen_time = time.time()

        episodes, episode_metrics = create_training_episodes(
            samples=next(train_iter),
            processor=processor,
            inference_engine=inference_engine,
            reward_function=reward_function,
            tokenizer=tokenizer,
            args=args,
            iteration=iteration,
        )

        logger.info(
            f"Time taken to generate {len(episodes['prompt_ids'])} responses: {time.time() - gen_time} seconds"
        )

        t0 = time.time()
        do_log = iteration % args.log_episodes_every_n_iterations == 0 or iteration == 0
        episode_table = dump_episodes(
            episodes=episodes,
            episodes_metrics=episode_metrics,
            exp_dir=exp_dir,
            tokenizer=tokenizer,
            iteration=iteration,
            processor=processor,
            do_log=do_log,
            rank=dist.get_rank(),
        )
        metrics.update({k: np.mean(v) for k, v in episode_metrics.items()})
        del episode_metrics
        logger.info(f"Time taken to dump episodes: {time.time() - t0} seconds")

        #########################################################
        # Training
        #########################################################

        # Prepare training batch
        start_time = time.time()
        model_inputs = collator(**episodes, device=curr_cuda_device)
        logger.info(f"Time taken to prepare training batch: {time.time() - start_time} seconds")

        if args.train_text_tokens_only:
            # Determine if we should train on image generation this step
            num_steps_in_cycle = args.text_only_cycle_steps
            step_in_cycle = iteration % num_steps_in_cycle
            train_on_text = step_in_cycle < args.text_only_steps  # First text_only_steps steps of each cycle

            completion_image_mask = model_inputs["completion_image_mask"]
            completion_mask = model_inputs["completion_mask"]
            logger.info(f"total active tokens before masking: {completion_mask.sum()}")

            if train_on_text:
                # For text-only: train on non-image tokens
                completion_mask = completion_mask * (~(completion_image_mask.bool())).long()
                logger.info(f"Training on text only (step {step_in_cycle}/{num_steps_in_cycle})")
            else:
                # For image generation: only train on image tokens
                completion_mask = completion_mask * completion_image_mask.long()
                logger.info(f"Training on image generation (step {step_in_cycle}/{num_steps_in_cycle})")

            logger.info(f"total active tokens after masking: {completion_mask.sum()}")
            model_inputs["completion_mask"] = completion_mask

        if args.train_vision_tokens_only:
            # Determine if we should train on image generation this step
            num_steps_in_cycle = args.vision_only_cycle_steps
            step_in_cycle = iteration % num_steps_in_cycle
            train_on_vision = step_in_cycle < args.vision_only_steps  # First vision_only_steps steps of each cycle

            completion_image_mask = model_inputs["completion_image_mask"]
            completion_mask = model_inputs["completion_mask"]
            logger.info(f"total active tokens before masking: {completion_mask.sum()}")

            if train_on_vision:
                # For vision-only: train on image tokens
                completion_mask = completion_mask * completion_image_mask.long()
                logger.info(f"Training on vision only (step {step_in_cycle}/{num_steps_in_cycle})")
            else:
                # For text generation: only train on text tokens
                pass  # We train on all tokens

            logger.info(f"total active tokens after masking: {completion_mask.sum()}")
            model_inputs["completion_mask"] = completion_mask

        # Compute reference logprobs
        start_time = time.time()
        reference_model.module.to(curr_cuda_device)
        reference_model.eval()
        logger.info(f"Time taken to move reference model to GPU: {time.time() - start_time} seconds")

        ref_log_probs = []
        with torch.no_grad():
            for i in trange(
                0,
                len(model_inputs["prompt_ids"]),
                args.per_device_batch_size,
                desc="Computing reference logprobs",
                disable=dist.get_rank() != 0,
            ):
                batch = {k: v[i : i + args.per_device_batch_size] for k, v in model_inputs.items()}
                ref_log_probs.append(
                    compute_token_log_probs(
                        model=reference_model,
                        inputs=batch,
                        temperature=args.temperature,
                        separate_language_and_vision_vocabs=args.separate_language_and_vision_vocabs,
                        vision_start_index=processor.get_vision_start_index(),
                    )
                )
            ref_log_probs = torch.cat(ref_log_probs)
            model_inputs["ref_log_probs"] = ref_log_probs
            del ref_log_probs

        reference_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        # Log advantage statistics
        close_to_zero_advantages = (
            (model_inputs["advantages"] < 1e-6) * model_inputs["completion_mask"]
        ).sum() / model_inputs["completion_mask"].sum()
        metrics["train/close_to_zero_advantages"] = close_to_zero_advantages.item()

        # Calculate losses and update model
        policy_model.train()
        total_response_len = model_inputs["completion_mask"].sum()

        see_memory_usage("Before training", force=dist.get_rank() == 0)

        train_time = time.time()

        for i in trange(
            0,
            len(model_inputs["prompt_ids"]),
            args.per_device_batch_size,
            desc="Gradient Accumulation",
            disable=dist.get_rank() != 0,
        ):
            batch = {k: v[i : i + args.per_device_batch_size] for k, v in model_inputs.items()}

            # Compute policy gradient loss
            loss, loss_metrics = compute_pg_loss(
                policy_model=policy_model,
                processor=processor,
                batch=batch,
                total_response_len=total_response_len,
                args=args,
            )

            # Track metrics
            metrics.setdefault("loss", []).append(loss.item())
            grad_norm = policy_model.get_global_grad_norm()
            if grad_norm is not None:
                grad_norm = grad_norm.item()
            metrics.setdefault("grad_norm", []).append(grad_norm)
            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

            # Backpropagation
            policy_model.backward(loss, scale_wrt_gas=False)
            del loss, loss_metrics

            # Optimization step
            policy_model.step()

        logger.info(f"Time taken to train: {time.time() - train_time} seconds")

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        #########################################################
        # Log metrics
        #########################################################

        if dist.get_rank() == 0:
            train_metrics = {k: np.mean(v) for k, v in metrics.items() if not np.any(np.equal(v, None))}
            train_metrics["learning_rate"] = policy_model.get_lr()[0]
            logs = {
                "iteration": iteration,
                f"episodes/iter_{iteration:06d}": episode_table,
                "active_tokens": total_response_len.item(),
                **{f"train/{k}": v for k, v in train_metrics.items()},
            }
            wandb.log(logs)

            selected_patterns = [
                "train/kl_penalty",
                "train/response_lengths",
                "train/rewards",
                "train/reward_metrics/*",
            ]
            key_metrics = {}
            for pattern in selected_patterns:
                matching_keys = fnmatch.filter(logs.keys(), pattern)
                for k in matching_keys:
                    key_metrics[k] = float(logs[k])
            logger.info(f"KEY METRICS: {key_metrics}")

        #########################################################
        # Save checkpoint
        #########################################################

        if (
            args.save_checkpoints_every_n_iterations is not None
            and (iteration % args.save_checkpoints_every_n_iterations == 0 or iteration == args.num_iterations - 1)
            and iteration != 0
        ):
            logger.info(f"Saving checkpoint at iteration {iteration}")
            checkpoint_dir = exp_dir / "checkpoints" / f"ckpt_{iteration:06d}"

            logger.info("Saving HF model and tokenizer")
            if dist.get_rank() == 0:
                Emu3ForCausalLM.save_pretrained
                policy_model.module.save_pretrained(str(checkpoint_dir / "hf_model"))
                tokenizer.save_pretrained(str(checkpoint_dir / "hf_model"))
            dist.barrier(device_ids=[torch.cuda.current_device()])

            policy_model.save_checkpoint(str(checkpoint_dir / "deepspeed"))

            if dist.get_rank() == 0:
                clean_up_checkpoints(
                    exp_dir, keep_every_n_steps=args.keep_checkpoints_every_n_iterations, exclude=[checkpoint_dir]
                )
            dist.barrier(device_ids=[torch.cuda.current_device()])

        if (
            args.push_to_hub_every_n_iterations is not None
            and (iteration % args.push_to_hub_every_n_iterations == 0 or iteration == args.num_iterations - 1)
            and iteration != 0
            and args.push_to_hub_repo_id is not None
        ):
            logger.info(f"Pushing checkpoint to hub at iteration {iteration}")
            repo_id = args.push_to_hub_repo_id
            if dist.get_rank() == 0:
                policy_model.module.push_to_hub(
                    repo_id=repo_id,
                    revision=f"ckpt_{iteration:06d}",
                    commit_message=f"Push model checkpoint at iteration {iteration}",
                    max_shard_size="3GB",
                )
                tokenizer.push_to_hub(
                    repo_id=repo_id,
                    revision=f"ckpt_{iteration:06d}",
                    commit_message=f"Push tokenizer checkpoint at iteration {iteration}",
                )
            dist.barrier(device_ids=[torch.cuda.current_device()])

        #########################################################
        # Update inference engine weights
        #########################################################

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        see_memory_usage("Before waking up inference engine", force=dist.get_rank() == 0)

        inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)


if __name__ == "__main__":
    main()
