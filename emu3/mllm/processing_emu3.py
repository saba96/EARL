# coding=utf-8
# Copyright 2024 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Processor class for Emu3. """

import re
from typing import List, Optional, Sequence
from functools import partial

from PIL import Image
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput
from transformers.utils import logging


from .utils import Emu3PrefixConstrainedLogitsHelperFreeForm


logger = logging.get_logger(__name__)

class Emu3Processor(ProcessorMixin):
    r"""
    Constructs an Emu3 processor which wraps an Emu3 image processor and an Emu3 vision vq model and an Emu3 tokenizer into a single processor.

    [`Emu3Processor`] offers all the functionalities of [`Emu3VisionVQModel`] and [`Emu3Tokenizer`]. See the
    [`~Emu3Processor.__call__`], [`~Emu3Processor.decode`], [`~Emu3Processor.vision_encode`], [`~Emu3Processor.vision_decode`]
    for more information.

    Args:
        image_processor ([`Emu3VisionVQImageProcessor`]):
            The image processor is a required input.
        vision_tokenizer ([`Emu3VisionVQModel`]):
            The vision tokenizer is a required input.
        tokenizer ([`Emu3Tokenizer`]):
            The tokenizer is a required input.
        prefix_template(`str`, *optional*):
            The prefix template for image tokens
        visual_template(`Tuple[str, ...]`, *optional*):
            The visual token template for image tokens
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["vision_tokenizer", "prefix_template", "visual_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        vision_tokenizer=None,
        tokenizer=None,
        chat_template="You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:",
        prefix_template="{H}*{W}",
        visual_template=("<|visual token {token_id:0>6d}|>", r"<\|visual token (\d+)\|>"),
        decode_mode = 'G',
        vllm=False,
        **kwargs,
    ):
        assert vision_tokenizer is not None, "image tokenizer can not be None"

        self.vision_tokenizer = vision_tokenizer
        self.prefix_template = prefix_template
        self.visual_template = visual_template
        self.vllm = vllm

        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.decode_mode = decode_mode
        self.height = None
        self.width = None

    @torch.no_grad()
    def __call__(
        self,
        text: Optional[TextInput | PreTokenizedInput] = None,
        image: Optional[Image.Image | List[Image.Image]] = None,
        *,
        mode: str = "G",
        ratio: str = "1:1",
        image_area: int = 518400,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Emu3Tokenizer's [`~Emu3Tokenizer.__call__`] to encode the text.
        To prepare the image(s), this method forwards the `image` argument to
        Emu3VisionVQImageProcessor's [`~Emu3VisionVQImageProcessor.__call__`] and Emu3VisionVQModel's [`~EmuVideoVQModel.encode`]
        if `image` is not `None`. Please refer to the doctsring of the above two methods for more information.

        Args:
            text (`str` or `List[str]`):
                The sequence or a batch of sequence to be encoded. A sequence is a string.
            image (`PIL.Image.Image` or `List[PIL.Image.Image]`, *optional*):
                The image or a batch of images to be prepared. An image is a PIL image.
            mode (`str`, *optional*, in `G`, `U` or `E`):
                task mode, `G` for generation, `U` for understanding, and `E` for image editing
            ratio (`str`, *optional*):
                the image width-height ratio for generation
            image_area (`int`, *optional*):
                image area used to calcualte the generated image height and width
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **image_size** -- List of image size of input images or generated images.
        """
        assert mode in ('G', 'C', 'U', 'E', 'CE', 'mixed'), "mode must be 'G', 'C', 'U', 'E', 'CE' or 'mixed'."
        if isinstance(text, str):
            text = [text]

        if not isinstance(text[0], str):
            raise ValueError("`text` must be string or list of string")

        image_inputs = None

        if mode == 'G':
            if image is not None:
                raise ValueError("You have to specify only `text` in generation mode")
            if len(text) > 1:
                raise ValueError("`text` can only be `str` in generation mode")
        elif mode == 'U':
            if image is None:
                raise ValueError("Invalid input image. Please provide exactly one PIL.Image.Image per text.")

            if not isinstance(image, Sequence) and not isinstance(image, Image.Image):
                raise ValueError("Invalid input image. Please provide PIL.Image.Image or List[PIL.Image.Image].")

            if isinstance(image, Sequence) and not isinstance(image[0], Image.Image):
                raise ValueError("Invalid input image. Please provide PIL.Image.Image or List[PIL.Image.Image].")

            image_inputs = self.image_processor(image, return_tensors="pt")["pixel_values"]
            image_inputs = image_inputs.to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
            image_tokens = self.vision_tokenizer.encode(image_inputs)

            if len(text) != len(image_tokens):
                raise ValueError("number of image must match number of text prompt")
        elif mode == 'E' or mode == 'C' or mode == 'CE' or mode == 'mixed':
            if image is None:
                raise ValueError("You have to specify `image` in the image editing mode")
            else:
                image_inputs = self.image_processor(image, return_tensors="pt")["pixel_values"]
                image_inputs = image_inputs.to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
                image_tokens = self.vision_tokenizer.encode(image_inputs)

            if len(text) > 1:
                raise ValueError("`text` can only be `str` in the image editing mode")

        prompt_list, size_list = [], []
        for idx, text_prompt in enumerate(text):
            prompt = self.tokenizer.bos_token
            if self.vllm:
                h, w = image_tokens[idx].shape
                # h_i, w_i = self.calculate_generate_size(f'{w}:{h}', image_area, self.vision_tokenizer.spatial_scale_factor)
                imgstr = self.to_imgstr(image_tokens[idx])
                prompt = (
                    self.tokenizer.boi_token +
                    self.prefix_template.format(H=h, W=w) +
                    self.tokenizer.img_token + 
                    imgstr +
                    self.tokenizer.eol_token +
                    self.tokenizer.eof_token +
                    self.tokenizer.eoi_token +
                    text_prompt +
                    self.get_assistant_prefill(h, w, mode)
                )
                prompt = self.tokenizer.bos_token + prompt
                sample = self.tokenizer.encode(
                    prompt,
                    add_special_tokens=False
                )
                size_list.append([h, w])
                return {"input_ids": sample, "image_size": size_list}
            else:
                assert False, "vllm is not supported"            
    @torch.no_grad()
    def batch_decode(self, *args, **kwargs):
        docs = self.tokenizer.batch_decode(*args, **kwargs)
        return [self.multimodal_decode(d) for d in docs]

    @torch.no_grad()
    def decode(self, *args, **kwargs):
        doc = self.tokenizer.decode(*args, **kwargs)
        return self.multimodal_decode(doc)

    @torch.no_grad()
    def vision_encode(self, *args, **kwargs):
        return self.vision_tokenizer.encode(*args, **kwargs)

    @torch.no_grad()
    def vision_decode(self, *args, **kwargs):
        return self.vision_tokenizer.decode(*args, **kwargs)

    @torch.no_grad()
    def multimodal_decode(self, doc):
        multimodal_output = []
        pattern = rf'({re.escape(self.tokenizer.img_token)}.*?{re.escape(self.tokenizer.eoi_token)})'
        chunks = re.split(pattern, doc)
        i = 0
        for c in chunks:
            if len(c) == 0:
                continue
            if self.tokenizer.img_token in c:
                image = []
                image_rows = re.split(re.escape(self.tokenizer.eol_token), c)
                try:
                    for r in image_rows:
                        token_ids = re.findall(self.visual_template[1], r)
                        if len(token_ids) > 0:
                            row_token = [int(m) for m in token_ids]
                            image.append(row_token)
                    i += 1
                    image = torch.tensor(image, dtype=torch.long, device=self.vision_tokenizer.device)
                    image = self.vision_tokenizer.decode(image[None]).float()
                    image = self.image_processor.postprocess(image)["pixel_values"][0]
                    multimodal_output.append(image)
                except Exception as e:
                    print(f"Error processing image at index {i}: {e}")
                    multimodal_output.append("*****Invalid image*****")
            else:
                multimodal_output.append(c)
        return multimodal_output

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def to_imgstr(self, image_tokens):
        image_tokens = image_tokens.cpu().numpy().tolist()
        image_token_str = [
            [
                self.visual_template[0].format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr

    def calculate_generate_size(self, ratio, image_area, spatial_scale_factor):
        w, h = map(int, ratio.split(":"))
        current_area = h * w
        target_ratio = (image_area / current_area) ** 0.5

        th = int(round(h * target_ratio / spatial_scale_factor))
        tw = int(round(w * target_ratio / spatial_scale_factor))
        return th, tw

    def build_const_helper(self):
        (
            img_token,
            eoi_token,
            eos_token,
            eol_token,
            eof_token,
            pad_token,
            vis_start,
            vis_end,
            boi_token
        ) = self.tokenizer.encode([
            self.tokenizer.img_token,
            self.tokenizer.eoi_token,
            self.tokenizer.eos_token,
            self.tokenizer.eol_token,
            self.tokenizer.eof_token,
            self.tokenizer.pad_token,
            self.visual_template[0].format(token_id=0),
            self.visual_template[0].format(token_id=self.vision_tokenizer.config.codebook_size - 1),
            self.tokenizer.boi_token,
        ])

        for token in [img_token, eoi_token, eos_token, eol_token, eof_token, pad_token, self.tokenizer.bos_token_id]:
            assert token not in range(vis_start, vis_end + 1), f"token {token} is in visual tokens"
        text_tokens = list(range(0, vis_start))
        if not self.vllm:
            assert False, "vllm is not supported"
        else:
            return partial(
                Emu3PrefixConstrainedLogitsHelperFreeForm,
                img_token=img_token,
                eoi_token=eoi_token,
                eos_token=eos_token,
                eol_token=eol_token,
                eof_token=eof_token,
                pad_token=pad_token,
                visual_tokens=list(range(vis_start, vis_end + 1)),
                text_tokens=text_tokens
            )

   
    def build_prefix_constrained_fn(self, height, width):
        self.height = height
        self.width = width
        self.const_helper = self.build_const_helper()
        helper = self.const_helper(height=height, width=width)
        return helper

    def get_assistant_prefill(self, h, w, mode: str) -> str:
        """Get the appropriate assistant prefill based on the mode."""
        if mode == "E":
            # return self.tokenizer.boi_token
            return (self.tokenizer.boi_token + self.prefix_template.format(H=h, W=w))
        elif mode == "CE":
            return " Let's think step by step. <|start thinking|>"
        elif mode == "mixed":
            return ""
        else:
            raise ValueError(f"Invalid mode: {mode}")


