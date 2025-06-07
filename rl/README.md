# RL for Image Editing

**Table of Contents:**
- [Setup the environment](#setup-the-environment)
- [Running RL Training Jobs](#running-rl-training-jobs)
  - [Step 1: Launching the Reward Server (Qwen2.5-VL-72B-Instruct vLLM)](#step-1-launching-the-reward-server-qwen25-vl-72b-instruct-vllm)
  - [Step 2: Starting RL Training](#step-2-starting-rl-training)
- [Code Structure](#code-structure)
- [Config File Structure](#config-file-structure)

This repository contains code for reinforcement learning experiments in image editing. The following guide will help you set up and run experiments. You will need at least 4 GPUs to run the reward model (Qwen2.5-VL-72B-Instruct) and at least 4 GPUs to run the RL training.

## Setup the environment
### Step 1
To get started, follow these steps:

```bash
# Clone the repository
git clone https://github.com/saba96/EARL.git
cd EARL
```
### Step 2

Create a python (>=3.10) environment with torch 2.6 and cuda 12.4

```bash
python -m venv .rl_venv
source .rl_venv/bin/activate
# Install torch first
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Install flash-attn
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### Step 3

Install the remaining dependencies

```bash
pip install -r rl/requirements.txt
```

### Step 4

Patch vLLM to support Emu3

```bash
nano rl_venv/lib/python3.10/site-packages/vllm/model_executor/models/registry.py
```
> *Replace `python3.10` with the version of python you are using*

Then, add the following line to the file line 166

```python
_MULTIMODAL_MODELS = {    
    # -----------add this line-----------
    "Emu3ForCausalLM": ("llama", "LlamaForCausalLM"), 
    # -----------end of editing----------
    "AriaForConditionalGeneration": ("aria", "AriaForConditionalGeneration"), # already exists
    ...
}
```
### Step 5

Download the training dataset and set up the environment variables

```bash
./rl/scripts/setup_datasets.sh
```


## Running RL Training Jobs

The training process involves two main steps. The first step should ideally be done once and then the second step can be run multiple times:

### Step 1: Launching the Reward Server (Qwen2.5-VL-72B-Instruct vLLM)

The reward server can be shared across 5-6 RL training jobs.

```bash
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

torchrun --nproc_per_node=$NUM_GPUS rl/test_distributed_env.py # Optional

# Launch vllm server
export VLLM_USE_V1=0

vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --port 4877 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --tensor-parallel-size $NUM_GPUS \
    --max-model-len 4096 \
    --max-num-seqs 1024 --limit-mm-per-prompt image=2
```

### Step 2: Starting RL Training

```bash
#! /bin/bash

CONFIG_NAME="<config_name>"

export APP__DATA_DIR=RL_DATASETS/data
export APP__BASE_EXP_DIR=RL_EXPERIMENTS
export APP__REWARD_FUNCTION_API_BASE=http://<reward_server_ip>:4877

export APP__RUN_NAME=$CONFIG_NAME
export WANDB_RUN_ID=$CONFIG_NAME

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

torchrun --nproc_per_node=$NUM_GPUS rl/test_distributed_env.py # Optional

export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m rl.simple_launch --nproc $NUM_GPUS \
    rl/nano_aha_moment.py \
    "rl/configs/${CONFIG_NAME}.jsonnet"
```

The `run_name` parameter creates a unique identifier for your run, which generates a unique wandb page for tracking metrics and establishes a unique checkpoint path for saving/loading model states. Using the same run_name allows you to resume training from where you left off.

Configuration files are located in `rl/configs/`.
Here are the available configurations:

- `sft_s__rl_s.jsonnet`: SFT (S) => RL (S)
- `sft_s__rl_c.jsonnet`: SFT (S) => RL (C)
- `sft_s__rl_sc.jsonnet`: SFT (S) => RL (S+C)
- `sft_sc__rl_c.jsonnet`: SFT (S+C) => RL (C)
- `sft_sc__rl_sc.jsonnet`: SFT (S+C) => RL (S+C)
- `sft_sc2stage__rl_c.jsonnet`: SFT (S+C) Two Stage => RL (C)
- `sft_sc2stage__rl_sc.jsonnet`: SFT (S+C) Two Stage => RL (S+C)
- `sft_think_s__rl_s.jsonnet`: SFT Think (S) => RL (S)
- `sft_think_s__rl_c.jsonnet`: SFT Think (S) => RL (C)
- `sft_think_s__rl_sc.jsonnet`: SFT Think (S) => RL (S+C)

The configuration names follow the pattern `sft_X__rl_Y.jsonnet` where:
- `X` indicates the SFT checkpoint type used as the initial model for RL training
- `Y` indicates the RL training configuration
- `S` = Simple edits dataset
- `C` = Complex edits dataset
- `S+C` = Combined simple and complex edits
- `Think` = Model trained with reasoning/thinking capabilities
- `Two Stage` = Two-stage training process

For example, `sft_s__rl_c.jsonnet` means:
1. RL start from a model that was SFT-trained on simple edits (`sft_s`)
2. The RL training is on complex edits (`rl_c`)

## Code Structure

The code inside `rl/` is organized as follows:

- `configs/`: Contains configuration files (`.jsonnet` files) for different RL runs
- `scripts/`: Contains scripts for setting up the environment and running the RL training jobs
- `nano_aha_moment.py`: The main RL training loop (heavily based on [github.com/McGill-NLP/nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment))
- `reward.py`: Contains the reward implementation
- `viescore.py`: Prompts for computing VIEScore
- `data.py`: Contains the data loading logic
- `utils.py`: Utility functions

## Config File Structure

Configuration files are written in [Jsonnet](https://jsonnet.org/) and located in `rl/configs/`. Jsonnet extends JSON with features like variables, conditionals, and file imports, making configurations more maintainable and reusable. Since Jsonnet is a superset of JSON, all JSON files are valid Jsonnet files.

Here are the fields in the config file:

```jsonnet
{
    // Dataset and Data Paths
    // Paths to training and validation data, using environment variables for flexibility
    train_paths_file: [], // List of training data paths
    coefficients: [], // Coefficients for training data
    val_paths_file: [
        std.extVar("APP__DATA_DIR") + "/validation_22march_256/feature/list/tokenized_datalist.json",
    ],
    val_coefficients: [1.0],
    base_exp_dir: std.extVar("APP__BASE_EXP_DIR"),

    // Model and Tokenizer Configuration
    // Paths to pre-trained model and tokenizer, and the assistant's response style
    model_path: "kazemnejad/Emu3-Base-SFT-got-Apr13_post_stochastic_256_lr1e-5-checkpoint-10800",
    tokenizer_path: "BAAI/Emu3-Gen",
    assistant_prefill: " Let's think step by step. <|start thinking|>", // Forced reasoning mode

    // Training Hyperparameters
    // Core parameters controlling the training process
    learning_rate: 3e-6,  // Learning rate for AdamW optimizer
    num_iterations: 2000,    // Total training iterations
    kl_coeff: 0.0003,        // KL divergence penalty coefficient
    temperature: 1.0,       // Sampling temperature
    per_device_batch_size: 8, // Samples per GPU

    // Generation Parameters
    // Settings controlling response generation
    episodes_per_iteration: 128,      // Complete interactions per iteration
    generations_per_sample: 8,       // Different responses per prompt
    max_response_tokens: 2048,       // Max tokens per response
    top_p: 1.0,                      // Nucleus sampling (1.0 = disabled)
    top_k: -1,                       // Top-k sampling (-1 = disabled)
    model_context_size: 4096,        // Max sequence length

    // vLLM Configuration
    // GPU memory management for vLLM
    vllm_gpu_memory_utilization: 0.3, // GPU memory ratio (0.0 to 1.0)

    // Emu3 Specific Settings
    // Special configurations for Emu3 model
    emu3_use_logit_processor: true,           // Use Emu3's logit processor
    separate_language_and_vision_vocabs: true, // Separate vocab spaces

    // Reward Function Configuration
    // Settings for reward computation
    reward_function_api_base: std.extVar("APP__REWARD_FUNCTION_API_BASE"),
    reward_function_api_model_name: "Qwen/Qwen2.5-VL-72B-Instruct",
    reward_compute_viescore: true,           // Compute VIEScore
    reward_compute_ground_score: false,       // Compute GroundScore
    reward_final_reward: "viescore",         // Final reward computation method
    // Options: "viescore", "ground_score", "sqrt(viescore_pqxground_score)"

    // Logging and Checkpointing
    // Experiment tracking and model saving
    run_name: std.extVar("APP__RUN_NAME"),
    log_episodes_every_n_iterations: 5,      // Log episodes frequency
    keep_checkpoints_every_n_iterations: 100, // Keep checkpoints frequency
    save_checkpoints_every_n_iterations: 100, // Save checkpoints frequency
    push_to_hub_every_n_iterations: 100,      // Push to HF Hub frequency
    push_to_hub_repo_id: 'imged_rl__' + $.run_name, // HF Hub repo ID
}
```

Note: Environment variables (prefixed with `APP__`) are used for flexible configuration across different environments. These should be set before running the training script.

