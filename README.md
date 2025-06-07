# Emu3 Image Editing

<!-- Official code for the paper *The Promise of RL for Autoregressive Image Editing (EARL)*. The code, models, and data will be released soon. -->

This README provides an overview of how to prepare and tokenize data for EARL image editing using either a Hugging Face dataset or local data. It also covers SFT, RL training and evaluation steps.

## Table of Contents
- [Prerequisites - Installation](#prerequisites---installation)
- [Tokenization](#tokenization)
  <!-- - [Multi-GPU Tokenization](#accelerate-you-can-tokenize-with-multi-gpu) -->
- [SFT Training](#sft-training)
- [RL Training](#rl-training)
- [Evaluation](#evaluation)
- [Try the model quickly: vLLM inference](#try-the-model-quickly-vllm-inference)

## Prerequisites - Installation

```bash
pip install -r requirements.txt

# make sure to install the correct version of flash-attn and deepspeed
pip install flash-attn==2.5.7 --no-build-isolation
pip install deepspeed==0.15.4
```

## Tokenization

For convenience, we provide pre-tokenized training data on Hugging Face 🤗. The dataset will be available at: [URL to be released soon]
 We have tokenization code compatible with both Hugging Face datasets and local files. Follow the steps below to tokenize your data.

To tokenize your data, use the following command. Make sure the images are in PIL format, if not you can slightly modify the code to read the image in the provided format. You can specify which key in your dataset corresponds to each required component (original image, edited image, and text instruction). You **need** to provide list of which keys are part of CoT. **Please see ./scripts/tokenize.sh**:

```bash
python ./emu3/train_image_editing/prepare_data_from_hugging.py --dataset-name "$data_name" --output-path "$output_dir" --original-image-key "$original_image_key" --edit-instruction "$edit_instruction" --CoT-keys "${CoT_keys[@]}" --edited-image "$edited_image" --image-area $image_area --random-seed 42
```

Parameters:
- `--dataset-name`: Path to your dataset, which can be either:
  - A Hugging Face dataset name (e.g., "username/dataset-name")
  - A local path to parquet files containing your dataset
- `--output-path`: Directory where the tokenized data will be saved
- `--original-image-key`: The key/column name in your dataset that contains the original images
- `--edit-instruction`: The key/column name containing the text instructions for editing
- `--CoT-keys`: Array of keys/column names that are part of the Chain of Thought (CoT) process
- `--edited-image`: The key/column name containing the edited/result images
- `--image-area`: The target area for the images (in pixels)
- `--random-seed`: Random seed for reproducibility
- `--source`: Source HF or local.
  - Default: `HF`

For multi-GPU tokenization, use the provided script:
```bash
. ./scripts/tokenize.sh
```

## SFT Training

The trained models will be available at on Hugging Face 🤗: [URL to be released soon]

For training choose between deepspeed stage zero 3 or deepspeed stage zero 3 offload based on your available resources. Use:

```bash
. ./scripts/ie_sft.sh
. ./scripts/ie__sft_offload.sh
```

Ensure that the `DATAPATH` variable points to your **tokenized data** list from the previous step in the bash file, and adjust `EXP_NAME` and `--output_dir`.

```bash
DATAPATH='/path/to/your/train.json'
```

**Note**: Adjust ``--cot_keys`` to mention which keys of CoT is intended to use for training if you want to train with reasoning. See example below.

```bash
--cot_keys 'reasoning_verbose'
```

The `--coefficients` argument is used to control the coefficient of sampling for each dataset.

If provided, each dataset is sampled based on its coefficient.
If not provided, the script automatically oversamples smaller datasets to match the size of the largest dataset.
**Important Note:** If you want automatic upsampling, remove the --coefficients argument when running the script.

## RL Training

The RL training is designed to be user-friendly with clear configuration options and automatic checkpointing. Check out [`rl/README.md`](./rl/README.md) for detailed instructions!

## Evaluation

The evaluation code and scripts will be released soon. This will include:
- Evaluation metrics and visualization tools
- Support for multiple datasets (magicBrush, VisMin, OmniEdit, Aurora-Ag, I2EBench)

## Try the model quickly: vLLM inference

1. Install vLLM
```bash
pip install vllm
```

2. Patch vLLM to support Emu3
```
vim /path/to/venv/lib/python3.10/site-packages/vllm/model_executor/models/registry.py
```

Add the following line to the file line 166
```python
_MULTIMODAL_MODELS = {    
    # add this line
    "Emu3ForCausalLM": ("llama", "LlamaForCausalLM"), 
    # end of adding
    "AriaForConditionalGeneration": ("aria", "AriaForConditionalGeneration"), # already exists
    ...
}
```

3. Run inference
```bash
. ./scripts/batch_eval.sh
```

Notes:
- vLLM support infinite batch size, so technically you can pass entire validation set at once.

