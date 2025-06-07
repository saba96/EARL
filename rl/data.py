from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from tqdm import trange


class InfiniteReplacementBatchSampler(Sampler):
    """
    Yields batches of random indices forever (uniform w/ replacement).
    """

    def __init__(self, data_source_len: int, batch_size: int, generator=None, skip_iterations: int = 0):
        self.n = data_source_len
        self.batch_size = batch_size
        self.generator = generator or torch.Generator()
        self.np_gr = np.random.RandomState(self.generator.initial_seed())
        self.skip_iterations = skip_iterations

    def __iter__(self):
        for _ in trange(self.skip_iterations, desc="Skipping iterations"):
            _ = self.np_gr.choice(self.n, size=self.batch_size, replace=False)

        while True:
            # Generate a batch of random indices
            indices = self.np_gr.choice(self.n, size=self.batch_size, replace=False).tolist()
            yield indices

    def __len__(self):
        return 2**63  # any big number; never actually used


class PreTokenizedEmu3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_paths: List[str],
        data_parallel_rank: int,
        data_parallel_world_size: int,
        tokenizer: AutoTokenizer,
        assistant_prefill: str,
        image_prefix_template: str = "{H}*{W}",
        vision_token_template: str = "<|visual token {token_id:0>6d}|>",
    ):
        self.data_paths = data_paths[data_parallel_rank::data_parallel_world_size]
        self.tokenizer = tokenizer
        self.assistant_prefill = assistant_prefill
        self.image_prefix_template = image_prefix_template
        self.vision_token_template = vision_token_template

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int) -> np.ndarray:
        data_path = self.data_paths[index]
        data = torch.load(data_path, weights_only=False)

        instruction: str = data["instruction"]
        original_image: np.ndarray = data["original_image"]

        prompt = (
            self.tokenizer.bos_token + self.encode_image_tokens(original_image) + instruction + self.assistant_prefill
        )

        edited_image_str = self.encode_image_tokens(data["edited_image"])
        edited_image_str_ids = self.tokenizer.encode(edited_image_str)[1:]

        return {
            "prompt_ids": np.array(self.tokenizer.encode(prompt)),
            "instruction": instruction,
            "original_image": original_image,
            "edited_image": data["edited_image"],
            "edited_image_str": edited_image_str,
            "edited_image_str_ids": edited_image_str_ids,
        }

    def encode_image_tokens(self, image_tokens: np.ndarray) -> str:
        """Convert image tokens to string representation for the model."""
        image_token_str = [
            [self.vision_token_template.format(token_id=token_id) for token_id in token_row]
            for token_row in image_tokens
        ]
        image_rows_str = ["".join(token_row) for token_row in image_token_str]

        return (
            self.tokenizer.boi_token
            + self.image_prefix_template.format(H=image_tokens.shape[0], W=image_tokens.shape[1])
            + self.tokenizer.img_token
            + self.tokenizer.eol_token.join(image_rows_str)
            + self.tokenizer.eol_token
            + self.tokenizer.eof_token
            + self.tokenizer.eoi_token
        )


class Emu3PPOCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_prompt_len: int = None,
        max_completion_len: int = None,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_completion_len = max_completion_len

    def __call__(
        self,
        prompt_ids: List[Sequence[int]],
        completion_ids: List[Sequence[int]],
        advantages: List[Sequence[float]],
        completion_image_masks: List[Sequence[int]],
        device: torch.device,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # We apply left padding to the query and right padding to the response
        max_prompt_len = self.max_prompt_len or max(len(p) for p in prompt_ids)
        max_completion_len = self.max_completion_len or max(len(c) for c in completion_ids)

        def to_list(x):
            if isinstance(x, (np.ndarray, torch.Tensor)):
                return x.tolist()
            return x

        inputs = {
            "prompt_ids": [],
            "prompt_mask": [],
            "completion_ids": [],
            "completion_mask": [],
            "completion_image_mask": [],
            "advantages": [],
        }

        pad_token_id = 0  # Doesn't matter, will be masked

        for idx in range(len(prompt_ids)):
            pmt = to_list(prompt_ids[idx])
            comp = to_list(completion_ids[idx])
            adv = to_list(advantages[idx])
            comp_img_mask = to_list(completion_image_masks[idx])

            num_left_padding = max_prompt_len - len(pmt)
            num_right_padding = max_completion_len - len(comp)

            pmt_mask = [0] * num_left_padding + [1] * len(pmt)
            pmt = [pad_token_id] * num_left_padding + pmt
            assert len(pmt) == len(pmt_mask) == max_prompt_len

            comp_mask = [1] * len(comp) + [0] * num_right_padding
            comp = comp + [pad_token_id] * num_right_padding
            comp_img_mask = comp_img_mask + [0] * num_right_padding
            adv = adv + [0.0] * num_right_padding

            assert len(comp) == len(comp_mask) == len(comp_img_mask) == len(adv) == max_completion_len

            inputs["prompt_ids"].append(pmt)
            inputs["prompt_mask"].append(pmt_mask)
            inputs["completion_ids"].append(comp)
            inputs["completion_mask"].append(comp_mask)
            inputs["completion_image_mask"].append(comp_img_mask)
            inputs["advantages"].append(adv)

        # Convert to tensors
        return {
            k: torch.tensor(v, dtype=torch.long if k != "advantages" else torch.float, device=device)
            for k, v in inputs.items()
        }
