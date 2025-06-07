import asyncio
import logging
import os
import re
import shutil
import socket
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import openai
import torch
import torch.distributed as dist
from datasets import Dataset
from deepspeed import DeepSpeedEngine
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

import wandb

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class OpenAIApiInference:
    def __init__(
        self,
        api_base: str,
        api_key: str = "NONE",
        model_name: str = "default",
        max_workers: int = 10,
        timeout: int = 1000,
    ):
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_workers)

        # Initialize client with optional api_base parameter
        client_kwargs = {
            "base_url": api_base,
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": 3,  # OpenAI client has built-in retries
        }

        self.client = AsyncOpenAI(**client_kwargs)

    @retry(
        retry=retry_if_exception_type(
            exception_types=(
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.APITimeoutError,
            )
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.1, max=0.5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def call_openai_api(self, query_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Call the OpenAI API with retry logic."""
        async with self.semaphore:
            logger.debug(f"Making API call for query {query_id}")
            try:
                response = await self.client.chat.completions.create(model=self.model_name, **query)
                return {
                    "query_id": query_id,
                    "response": response,
                }
            except Exception as e:
                logger.warning(f"API call failed for query {query_id}: {str(e)}")
                raise

    async def call_chat_async(
        self, queries: List[Dict[str, Any]], tqdm_desc: str = "Performing API calls", tqdm_enable: bool = True
    ) -> List[ChatCompletion]:
        tasks = []
        for i, query in enumerate(queries):
            task = asyncio.create_task(self.call_openai_api(query_id=i, query=query))
            tasks.append(task)

        results = {}
        for task in tqdm_asyncio.as_completed(tasks, desc=tqdm_desc, disable=not tqdm_enable):
            res = await task
            results[res["query_id"]] = res["response"]

        # Put results in order of queries
        results = [results[i] for i in range(len(queries))]
        return results

    def call_chat(
        self, queries: List[Dict[str, Any]], tqdm_desc: str = "Performing API calls", tqdm_enable: bool = True
    ) -> List[ChatCompletion]:
        return asyncio.run(self.call_chat_async(queries, tqdm_desc, tqdm_enable))


def get_api_base_from_github(repo_url: str = "https://github.com/kazemnejad/mila-api-base") -> Optional[str]:
    """
    Get the API base URL from the api.txt file in the GitHub repository.

    Args:
        repo_url (str): The GitHub repository URL (e.g., 'https://github.com/username/repo' or 'git@github.com:username/repo.git')

    Returns:
        Optional[str]: The API base URL from api.txt or None if not found
    """
    # Handle SSH format
    if repo_url.startswith("git@github.com:"):
        repo_url = repo_url.replace("git@github.com:", "https://github.com/")

    # Handle HTTPS format
    if not repo_url.startswith("https://github.com/"):
        return None

    # Remove .git suffix if present
    repo_url = repo_url.rstrip(".git")

    # Extract username and repo name
    match = re.match(r"https://github.com/([^/]+)/([^/]+)", repo_url)
    if not match:
        return None

    username, repo = match.groups()

    # Construct raw content URL for api.txt
    raw_url = f"https://raw.githubusercontent.com/{username}/{repo}/main/api.txt"

    try:
        import requests

        response = requests.get(raw_url)
        if response.status_code == 200:
            api_base = response.text.strip()
            print(f"API base: {api_base}")
            api_base = api_base.replace(".server.mila.quebec", "")
            return api_base
        return None
    except Exception:
        return None


@torch.compile(dynamic=True)
def log_softmax_and_gather(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Copied from https://github.com/allenai/open-instruct/blob/main/open_instruct/model_utils.py#L425

    torch compiled version of the common `log_softmax -> gather` operation.

    The compiled version of this opration avoids the (significant) memory overhead of
    allocating a new (batch_size, seq_len, vocab_size) tensor to store the logprobs.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size) containing the logits
        index: Tensor of shape (batch_size, seq_len) containing the indices to gather

    Returns:
        Tensor of shape (batch_size, seq_len) containing the log probabilities for the
        specified indices

    See https://github.com/allenai/open-instruct/pull/584
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def compute_token_log_probs(
    model,
    inputs: Dict[str, torch.Tensor],
    temperature: float = 1.0,
    vision_start_index: Optional[int] = None,
    separate_language_and_vision_vocabs: bool = False,
) -> torch.Tensor:
    inputs = {k: v for k, v in inputs.items()}
    _ = inputs.pop("advantages", None)
    prompt_ids, prompt_mask = inputs.pop("prompt_ids"), inputs.pop("prompt_mask")
    completion_ids, completion_mask = inputs.pop("completion_ids"), inputs.pop("completion_mask")
    completion_image_masks = inputs.pop("completion_image_mask")

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        logits_to_keep=logits_to_keep + 1,
        return_dict=True,
        use_cache=False,
        **inputs,
    ).logits

    logits = logits[:, :-1, :] / temperature  # Shape: [batch_size, seq_len, vocab_size]

    if separate_language_and_vision_vocabs:
        lang_logits = logits[:, :, :vision_start_index]
        vision_logits = logits[:, :, vision_start_index:]

        lang_completion_ids = torch.clamp(completion_ids, max=vision_start_index - 1)
        vision_completion_ids = torch.clamp(completion_ids - vision_start_index, min=0)

        lang_log_probs = log_softmax_and_gather(lang_logits, lang_completion_ids)
        vision_log_probs = log_softmax_and_gather(vision_logits, vision_completion_ids)

        log_probs = torch.where(
            completion_image_masks.bool(), vision_log_probs, lang_log_probs
        )  # Shape: [batch_size, seq_len]
    else:
        log_probs = log_softmax_and_gather(logits, completion_ids)  # [batch_size, seq_len]

    log_probs = log_probs * completion_mask  # Shape: [batch_size, seq_len]

    return log_probs


def dump_episodes(
    episodes: Dict[str, Any],
    episodes_metrics: Dict[str, Any],
    exp_dir: Path,
    tokenizer: AutoTokenizer,
    iteration: int,
    processor,
    is_eval: bool = False,
    do_log: bool = True,
    max_samples: Optional[int] = None,
    rank: int = 0,
) -> wandb.Table:
    def tidy_up_vision_tokens(token_ids: List[int]) -> str:
        image_mask = processor.compute_image_mask(token_ids)
        repl_token_id = processor.get_vision_start_index()
        repl_token = tokenizer.convert_ids_to_tokens(repl_token_id)
        token_ids = [repl_token_id if is_image else tok_id for tok_id, is_image in zip(token_ids, image_mask)]
        text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        text = re.sub(rf"(?:{re.escape(repl_token)})+", "\n[...vTokens...]", text)
        return text

    samples = episodes["samples"]
    prompt_ids = episodes["prompt_ids"]
    completion_ids = episodes["completion_ids"]
    advantages = episodes["advantages"]
    rewards = episodes["rewards"]
    response_lengths = episodes_metrics["response_lengths"]

    if not is_eval and rank == 0:
        print(f"########## Example 1 (Reward: {rewards[0]}, Response Length: {response_lengths[0]})")
        print(f"#### Prompt:\n`{tidy_up_vision_tokens(prompt_ids[0])}`")
        print(f"#### Completion:\n`{tidy_up_vision_tokens(completion_ids[0])}`\n\n")

        print(f"########## Example 2 (Reward: {rewards[1]}, Response Length: {response_lengths[1]})")
        print(f"#### Prompt:\n`{tidy_up_vision_tokens(prompt_ids[1])}`")
        print(f"#### Completion:\n`{tidy_up_vision_tokens(completion_ids[1])}`\n\n")

    if not do_log:
        return None

    if is_eval:
        episodes_dir = exp_dir / "eval_episodes"
    else:
        episodes_dir = exp_dir / "episodes"
    episodes_dir = episodes_dir / f"iter_{iteration:06d}" / f"rank_{rank}"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    if max_samples is None:
        indices = range(len(prompt_ids))
    else:
        max_samples = min(max_samples, len(prompt_ids))
        indices = np.random.choice(len(prompt_ids), size=max_samples, replace=False).tolist()

    columns = [
        "prompt",
        "completion",
        "orig_image",
        "gt_edited",
        "pred_edited",
        "advantage",
        "reward",
        "response_length",
    ]
    reward_metrics_column = [f"reward/{k}" for k in sorted(rewards[0].keys()) if k != "final_reward"]
    columns += reward_metrics_column

    original_images = processor.decode_to_images_from_token_ids([s["original_image"] for s in samples])
    gt_edited_images = processor.decode_to_images_from_token_ids([s["edited_image"] for s in samples])
    pred_edited_images = processor.decode_to_images(tokenizer.batch_decode(completion_ids, skip_special_tokens=False))

    table = wandb.Table(columns=columns)

    for idx in indices:
        prompt = tidy_up_vision_tokens(prompt_ids[idx])
        completion = tidy_up_vision_tokens(completion_ids[idx])
        orig_image = original_images[idx]
        gt_edited = gt_edited_images[idx]
        pred_edited = pred_edited_images[idx]
        adv = advantages[idx][0]
        reward_detailed = rewards[idx]
        response_length = response_lengths[idx]

        table.add_data(
            prompt,
            completion,
            wandb.Image(orig_image),
            wandb.Image(gt_edited),
            wandb.Image(pred_edited) if pred_edited is not None else None,
            adv,
            reward_detailed["final_reward"],
            response_length,
            *[reward_detailed[k.replace("reward/", "")] for k in reward_metrics_column],
        )

        data = {
            "prompt": prompt,
            "completion": completion,
            "orig_image": np.array(orig_image),
            "gt_edited": np.array(gt_edited),
            "pred_edited": np.array(pred_edited) if pred_edited is not None else None,
            "advantage": adv,
            "reward": reward_detailed["final_reward"],
            "response_length": response_length,
            **{k: reward_detailed[k.replace("reward/", "")] for k in reward_metrics_column},
        }

        np.savez(episodes_dir / f"eps_{idx:06d}.npz", **data)

    return table


def find_last_checkpoint(exp_dir: Path) -> Tuple[Optional[Path], Optional[int]]:
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("ckpt_*"))
    # Filter out directories that don't have a deepspeed subdirectory
    checkpoints = [ckpt for ckpt in checkpoints if (ckpt / "deepspeed").exists()]
    if not checkpoints:
        return None, None
    ckpt_path = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1]))
    ckpt_iter = int(ckpt_path.stem.split("_")[-1])
    return ckpt_path, ckpt_iter


def clean_up_checkpoints(
    exp_dir: Path, keep_every_n_steps: Optional[int] = None, exclude: Optional[List[Path]] = None
) -> None:
    if exclude is None:
        exclude = []

    checkpoint_dir = exp_dir / "checkpoints"
    for ckpt in checkpoint_dir.glob("ckpt_*"):
        if keep_every_n_steps is None or ckpt in exclude:
            continue

        ckpt_iter = int(ckpt.stem.split("_")[-1])
        if ckpt_iter % keep_every_n_steps == 0:
            # Remove non-hf_model files and dirs
            removed_files_and_dirs = []
            for file in ckpt.iterdir():
                if file.name not in ["hf_model"]:
                    try:
                        removed_files_and_dirs.append(file.name)
                        if file.is_dir():
                            shutil.rmtree(file, ignore_errors=True)
                    except Exception as e:
                        logger.warning(f"Error removing {file}: {e}")
            if len(removed_files_and_dirs) > 0:
                logger.info(f"Removed non-hf_model files and dirs: of checkpoint {ckpt.name}")

            continue

        logger.info(f"Removing checkpoint {ckpt}")
        shutil.rmtree(ckpt)


def load_model_into_vllm(model: Union[DeepSpeedEngine, PreTrainedModel], llm) -> None:
    """
    Load weights from a HuggingFace model (either wrapped in DeepSpeed or not) into a vLLM inference engine.

    This function transfers the weights from a training model to a vLLM inference engine,
    allowing for efficient inference using the updated model weights.

    Args:
        model (Union[DeepSpeedEngine, PreTrainedModel]): The source model to copy weights from.
            Can be either a DeepSpeed-wrapped model or a regular HuggingFace PreTrainedModel.
        vllm (LLM): The target vLLM inference engine to load the weights into.
            Must be already initialized and ready to accept new weights.

    Returns:
        None
    """
    state_dict = model.module.state_dict() if isinstance(model, DeepSpeedEngine) else model.state_dict()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())


def initialize_distributed_training_pg(rank: int, world_size: int):
    """
    Initialize the distributed training process group using NCCL backend.

    This function sets up the distributed training environment by:
    1. Reading environment variables for rank, world size, and master node configuration
    2. Setting the CUDA device for the current process
    3. Initializing the NCCL process group with the specified configuration
    4. Synchronizing all processes using a barrier

    Environment Variables Used:
        APP__MASTER_ADDR: Address of the master node (default: "localhost")
        APP__MASTER_TRAINING_PORT: Port for the master node (default: "8473")

    The function logs initialization information on rank 0 and ensures all processes
    are synchronized before proceeding.

    Args:
        rank: The global rank of the process
        world_size: Total number of processes

    Note:
        This function assumes CUDA is available and sets the CUDA device based on
        the local rank. It uses NCCL as the backend for distributed communication.
    """
    local_rank = rank

    master_addr = os.environ.get("APP__MASTER_ADDR", "localhost")
    master_training_port = os.environ.get("APP__MASTER_TRAINING_PORT", "8473")

    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        logger.info(
            f"{'#' * 80}\n"
            f"# Initializing the training NCCL PG with\n"
            f"# world_size={world_size} current_rank={rank} \n"
            f"{'#' * 80}"
        )

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_training_port}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=1800),
    )
    dist.barrier(device_ids=[local_rank])
    logger.info(f"Rank{rank}: training NCCL PG initialized.")
