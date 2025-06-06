# minor modification from:
#   https://github.com/baaivision/Emu3/blob/main/emu3/mllm/processing_emu3.py
#   https://github.com/baaivision/Emu3/blob/main/emu3/mllm/utils_emu3.py
# Copied from:
# https://github.com/FlagOpen/FlagScale/blob/main/flagscale/inference/processing_emu3.py


import torch


from transformers.utils import logging


logger = logging.get_logger(__name__)

class Emu3PrefixConstrainedLogitsHelperFreeForm:

    def __init__(
        self,
        height,
        width,
        img_token,
        eoi_token,
        eos_token,
        eol_token,
        eof_token,
        pad_token,
        visual_tokens,
        text_tokens
    ):
        self.height = height
        self.width = width
        self.img_token = img_token
        self.eoi_token = eoi_token
        self.eos_token = eos_token
        self.eol_token = eol_token
        self.eof_token = eof_token
        self.pad_token = pad_token
        self.visual_tokens = visual_tokens
        self.text_tokens = text_tokens

        self.offset_cache = {}

    def __call__(self, batch_id, input_ids):
        # assert isinstance(input_ids, torch.Tensor)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        if len(torch.nonzero(input_ids == self.img_token, as_tuple=True)[0])==0:
            return self.text_tokens
        positions = torch.nonzero(input_ids == self.img_token, as_tuple=True)[0]
        if len(positions) == 1:
            start_position = positions[0].item()
        else:
            raise ValueError(f"Expected at only one img_token occurrences, but found {len(positions)}.")
        height = self.height[batch_id] if self.height.shape[0] > 1 else self.height[0]
        width = self.width[batch_id] if self.width.shape[0] > 1 else self.width[0]
        offset = len(input_ids) - start_position
        if offset % (width + 1) == 0:
            return (self.eol_token, )
        elif offset == (width + 1) * height + 1:
            return (self.eof_token, )
        elif offset == (width + 1) * height + 2:
            return (self.eoi_token, )
        elif offset == (width + 1) * height + 3:
            return (self.eos_token, )
        elif offset > (width + 1) * height + 3:
            return (self.pad_token, )
        else:
            return self.visual_tokens
