import base64
import logging
import math
import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_delay, wait_fixed

from rl import viescore
from rl.utils import OpenAIApiInference

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class ImageEditingRewardFunction:
    """
    A class for implementing reward functions in reinforcement learning tasks.

    This class provides functionality for evaluating and scoring model outputs in reinforcement learning scenarios.
    It works with image-based completions and processes them to calculate reward scores based on the specific
    implementation of the __call__ method.

    Methods:
        __call__: Evaluates a batch of completions and returns corresponding reward scores
    """

    def __init__(
        self,
        api_base: str,
        api_model_name: str,
        processor,
        ensure_api_running: bool = True,
        enable_logging: bool = True,
        args=None,
    ):
        """
        Initialize the RewardFunction with configuration and processor.

        Args:
            api_base (str): Base URL for the OpenAI API
            api_model_name (str): Name of the OpenAI API model
            processor (Any): Processor object for decoding and processing model outputs
        """
        self.processor = processor
        self.api_base = api_base
        self.api_model_name = api_model_name
        self.enable_logging = enable_logging
        self.ensure_api_running = ensure_api_running
        self.args = args
        if not enable_logging:
            logger.setLevel(logging.ERROR)

    def __call__(
        self,
        *,
        completions: List[str] = None,
        finish_reasons: List[str] = None,
        samples: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, float]]:
        """
        Evaluate a batch of completions and return their reward scores.
        Each completion is the complete response of the model to an editing instruction.

        This method takes model completions, their finish reasons, and corresponding samples,
        then decodes them to images and calculates rewards based on the specific implementation.

        Args:
            completions (List[str]): List of completions
            finish_reasons (List[str]): List of finish reasons for each completion; could be 'stop' or 'length'
            samples (List[Dict[str, Any]]): List of sample dictionaries containing the original data point like
                original image and the editing instruction

        Returns:
            List[Dict[str, float]]: List of reward scores for each completion. It should at least
                contain the key 'final_reward'. Other keys can be added to the dictionary for logging purposes.
        """
        instructions: List[str] = [sample["instruction"] for sample in samples]
        original_images: List[Image.Image] = self.processor.decode_to_images_from_token_ids(
            [sample["original_image"] for sample in samples]
        )
        gt_edited_images: List[Image.Image] = self.processor.decode_to_images_from_token_ids(
            [sample["edited_image"] for sample in samples]
        )
        edited_images: List[Optional[Image.Image]] = self.processor.decode_to_images(completions)

        assert self.args is not None

        if self.ensure_api_running:
            self._ensure_api_is_running()

        openai_api_inference = OpenAIApiInference(self._get_api_base(), model_name=self.api_model_name, max_workers=64)

        rewards = defaultdict(dict)

        #########################################################
        # Compute VIEScore
        #########################################################

        if self.args.reward_compute_viescore:
            viescores = self.compute_viescore(
                instructions=instructions,
                original_images=original_images,
                edited_images=edited_images,
                openai_api_inference=openai_api_inference,
            )
            for idx in range(len(viescores)):
                rewards[idx].update({f"viescore__{key}": viescores[idx][key] for key in viescores[idx].keys()})

        if self.args.reward_compute_ground_score:
            ground_scores = self.compute_ground_score(
                instructions=instructions,
                gt_edited_images=gt_edited_images,
                edited_images=edited_images,
                openai_api_inference=openai_api_inference,
            )
            for idx in range(len(ground_scores)):
                rewards[idx].update(
                    {f"ground_score__{key}": ground_scores[idx][key] for key in ground_scores[idx].keys()}
                )

        #########################################################
        # TODO: add other reward functions here
        #########################################################

        # Compute final reward
        for idx in range(len(rewards)):
            if finish_reasons[idx] == "stop":
                if self.args.reward_final_reward == "viescore":
                    rewards[idx]["final_reward"] = rewards[idx]["viescore__overall"]
                elif self.args.reward_final_reward == "ground_score":
                    rewards[idx]["final_reward"] = rewards[idx]["ground_score__overall"]
                elif self.args.reward_final_reward == "sqrt(viescore_pqxground_score)":
                    rewards[idx]["final_reward"] = math.sqrt(
                        rewards[idx]["viescore__perceptual_quality"] * rewards[idx]["ground_score__overall"]
                    )
                else:
                    raise ValueError(f"Invalid final reward formula: {self.args.reward_final_reward}")
            else:
                rewards[idx]["final_reward"] = 0.0

        # Reorder rewards to match the original instructions order
        rewards = [rewards[i] for i in range(len(rewards))]

        return rewards

    def compute_viescore(
        self,
        *,
        instructions: List[str],
        original_images: List[Image.Image],
        edited_images: List[Optional[Image.Image]],
        openai_api_inference: OpenAIApiInference,
    ) -> List[Dict[str, float]]:
        SEMANTIC_CONSISTENCY_PROMPT_TEMPLATE = "\n".join(
            [
                viescore._context_no_delimit,
                viescore._prompts_0shot_two_image_edit_rule,
                viescore._prompts_0shot_tie_rule_SC,
            ]
        )

        PERCEPTUAL_QUALITY_PROMPT_TEMPLATE = "\n".join(
            [
                viescore._context_no_delimit,
                viescore._prompts_0shot_rule_PQ,
            ]
        )

        def create_sc_query(instruction, original_image, edited_image):
            sc_prompt = SEMANTIC_CONSISTENCY_PROMPT_TEMPLATE.replace("<instruction>", instruction)
            return {
                "messages": [
                    {"role": "user", "content": self._prepare_content(sc_prompt, [original_image, edited_image])}
                ],
                "max_tokens": 1400,
            }

        def create_pq_query(edited_image):
            pq_prompt = PERCEPTUAL_QUALITY_PROMPT_TEMPLATE
            return {
                "messages": [{"role": "user", "content": self._prepare_content(pq_prompt, [edited_image])}],
                "max_tokens": 1400,
            }

        rewards = {}
        max_tries = 5
        tries = 0

        while len(rewards) < len(instructions) and tries < max_tries:
            queries = []
            indices = []
            for idx, (inst, orig_img, edited_img) in enumerate(zip(instructions, original_images, edited_images)):
                if idx in rewards:
                    continue
                if edited_img is None:
                    rewards[idx] = {
                        "edit_success": 0.0,
                        "overedit": 0.0,
                        "naturalness": 0.0,
                        "artifacts": 0.0,
                        "semantic_consistency": 0.0,
                        "perceptual_quality": 0.0,
                        "overall": 0.0,
                    }
                    continue

                queries.append(create_sc_query(inst, orig_img, edited_img))
                queries.append(create_pq_query(edited_img))
                indices.append(idx)

            assert len(queries) == len(indices) * 2

            responses = openai_api_inference.call_chat(
                queries, tqdm_desc="Computing VIEScore", tqdm_enable=self.enable_logging
            )
            sc_responses = [resp for i, resp in enumerate(responses) if i % 2 == 0]
            pq_responses = [resp for i, resp in enumerate(responses) if i % 2 == 1]

            for idx, sc_resp, pq_resp in zip(indices, sc_responses, pq_responses):
                sc_resp = sc_resp.choices[0].message.content
                pq_resp = pq_resp.choices[0].message.content
                sc_dict = viescore.mllm_output_to_dict(sc_resp, give_up_parsing=False)
                pq_dict = viescore.mllm_output_to_dict(pq_resp, give_up_parsing=False)

                if sc_dict is False or pq_dict is False:
                    continue

                assert idx not in rewards

                sc_score = min(sc_dict["score"])
                pq_score = min(pq_dict["score"])

                rewards[idx] = {
                    "edit_success": sc_dict["score"][0],
                    "overedit": sc_dict["score"][1],
                    "naturalness": pq_dict["score"][0],
                    "artifacts": pq_dict["score"][1],
                    "semantic_consistency": sc_score,
                    "perceptual_quality": pq_score,
                    "overall": math.sqrt(sc_score * pq_score),
                }

            tries += 1
            if len(rewards) < len(instructions):
                logger.warning(
                    f"Failed to compute VIEScore for {len(instructions) - len(rewards)} instructions, retrying..."
                )

        assert len(rewards) == len(instructions)

        # Reorder rewards to match the original instructions order
        rewards = [rewards[i] for i in range(len(instructions))]

        return rewards

    def compute_ground_score(
        self,
        *,
        instructions: List[str],
        gt_edited_images: List[Image.Image],
        edited_images: List[Optional[Image.Image]],
        openai_api_inference: OpenAIApiInference,
    ) -> List[Dict[str, float]]:

        def create_gs_query(instruction, gt_edited_image, edited_image):
            """
            Build a chat-completion request that asks the model to compare the
            predicted image (`edited_image`) against the ground-truth edited
            image (`gt_edited_image`) under the provided editing instruction.
            """
            user_prompt = GROUND_SCORE_USER_PROMPT.format(eps_instruction=instruction)
            return {
                "messages": [
                    {"role": "system", "content": GROUND_SCORE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": self._prepare_content(user_prompt, [gt_edited_image, edited_image]),
                    },
                ],
                "max_tokens": 1300,
                "temperature": 0.3,
            }

        def parse_score(response_text: str) -> Optional[int]:
            match = re.search(r"<score>(\d+)</score>", response_text)
            if match:
                score = int(match.group(1))
                if 0 <= score <= 10:
                    return score
                logger.info(f"Invalid Ground score: {score}")
            else:
                logger.info(f"No score found in response: `{response_text}`")
            return None

        rewards: Dict[int, Dict[str, float]] = {}
        max_tries = 5
        tries = 0

        while len(rewards) < len(instructions) and tries < max_tries:
            queries: List[Dict[str, Any]] = []
            indices: List[int] = []

            for idx, (inst, gt_ed_img, ed_img) in enumerate(zip(instructions, gt_edited_images, edited_images)):
                if idx in rewards:
                    continue

                # If no edited image exists, assign a zero score.
                if ed_img is None:
                    rewards[idx] = {"overall": 0.0}
                    continue

                queries.append(create_gs_query(inst, gt_ed_img, ed_img))
                indices.append(idx)

            if not queries:
                break  # All items are already scored.

            assert len(queries) == len(indices)

            responses = openai_api_inference.call_chat(
                queries, tqdm_desc="Computing GroundScore", tqdm_enable=self.enable_logging
            )

            for idx, resp in zip(indices, responses):
                resp_text = resp.choices[0].message.content
                gs = parse_score(resp_text)

                if gs is None:
                    # Ill-formed response; retry in the next round.
                    continue

                assert idx not in rewards
                rewards[idx] = {"overall": float(gs)}

            tries += 1
            if len(rewards) < len(instructions):
                logger.warning(
                    f"Failed to compute GroundScore for "
                    f"{len(instructions) - len(rewards)} instructions, retrying..."
                )

        assert len(rewards) == len(instructions)

        # Preserve the original order.
        ordered_rewards = [rewards[i] for i in range(len(instructions))]
        return ordered_rewards

    def _prepare_content(self, text_prompt: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
        prompt_content = [{"type": "text", "text": text_prompt}]
        for image in images:
            visual_dict = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_image(image)}"},
            }
            prompt_content.append(visual_dict)

        return prompt_content

    @retry(
        stop=stop_after_delay(120 * 60),  # 2 hour timeout
        wait=wait_fixed(5),  # 5 seconds between retries
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _ensure_api_is_running(self):
        api_base = self._get_api_base()
        logger.info(f"Ensuring API is running at {api_base}")
        response = requests.get(f"{api_base}/models")
        if response.status_code == 200:
            logger.info(f"API is running at {api_base}")
            return response
        else:
            logger.warning(f"API returned status code {response.status_code}: {response.text}")
            raise Exception(f"API returned status code {response.status_code}")

    def _get_api_base(self):
        if isinstance(self.api_base, str):
            return self.api_base
        elif callable(self.api_base):
            return self.api_base()
        else:
            raise ValueError(f"Invalid API base: {self.api_base}")


def encode_pil_image(pil_image):
    # Create an in-memory binary stream
    image_stream = BytesIO()

    # Save the PIL image to the binary stream in JPEG format (you can change the format if needed)
    pil_image.save(image_stream, format="JPEG")

    # Get the binary data from the stream and encode it as base64
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode("utf-8")

    return base64_image


GROUND_SCORE_SYSTEM_PROMPT = """You are a **semantic visual evaluator**.

Your task is to **verify whether a predicted image semantically matches a ground-truth image**, assuming both were generated in response to the **same editing instruction**.
Note that All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You are given:

- **Instruction** - a textual description of the intended image edit  
- **Image A (the first image)** - the **ground-truth** edited image (what the correct edit looks like)  
- **Image B (the second image)** - the **predicted** edited image (generated by a model)

Your job is **not** to assess the ground truth image or whether the ground truth follow the instruction, but, you should always assume the ground truth image is the correct edit, and then your jobs is to score how much the predicted image is semantically similar to the ground truth image.

### What to check

1. **Interpret the instruction** and identify what kind of semantic change could be expected.
2. **Compare the two images** (A and B):
   - The predicted image is from a weak model, so first it has to be a valid image with proper generated objects.
   - Does the changes made in the predicted image match the changes made in the ground truth image?
   - Are there differences from the ground truth image that would change the **meaning or intent** of the instruction?

Scoring (semantic alignment)

Return a single integer score from 0 to 10, formatted as:

<score>8</score>

### Scoring guide
- 10 - Both images match the instruction perfectly and express it in highly similar ways.
- 7-9 - Both fulfill the instruction well, with only minor stylistic or layout differences.
- 4-6 - Some semantic alignment, but key visual details differ in meaning or interpretation, or the predicted image has some artifacts (not clear objects, etc.)
- 1-3 - The predicted image misunderstands or only loosely follows the instruction compared to ground truth.
- 0 - The predicted image has no meaningful connection to the instruction (while the ground truth does) or the predicted image is a completely degenerate image.

### Output format

Start with an concise explanation that compares semantic fidelity between Image A and Image B.
Then, on a new line, output the score in this exact format:

<score>X</score>

(Replace X with your integer score.)"""

GROUND_SCORE_USER_PROMPT = """
### Instruction
{eps_instruction}

### Ground Truth Edited Image (Image A)
The first image is the ground truth edited image.

### Predicted Edited Image (Image B)
The second image is the predicted edited image generated by the model.

Compare **Image A** (ground truth) and **Image B** (predicted image), using only the instruction to guide your judgment. You are not given the original image.

First, reason and briefly explain your reasoning.
Then, provide a score from 0 to 10 indicating how well the **predicted image (Image B)** semantically matches the **ground truth (Image A)** with respect to the **instruction**.

Output the final score in this format on a new line:

<score>X</score>
"""
