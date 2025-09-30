#!/bin/bash
#SBATCH --job-name=qwen2.5vl-72b-vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=450G
#SBATCH --cpus-per-task=48

# Load required modules
module load cuda/12.2 arrow/17 python/3.11 opencv
source $HOME/venvs/earl/bin/activate

# Set environment variables
export HF_HOME=$HOME/path/to/hf_home
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=DEBUG
export NCCL_P2P_DISABLE=1

# Save hostname to file
PROJECT_DIR=$HOME/path/to/earl
mkdir -p $PROJECT_DIR
hostname > $PROJECT_DIR/qwen_node.txt

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
torchrun --nproc_per_node=$NUM_GPUS rl/test_distributed_env.py

# Launch vllm server
export VLLM_USE_V1=0

# export VLLM_PORT=$((4000 + RANDOM % 1000))
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --port 4877 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --tensor-parallel-size $NUM_GPUS \
    --max-model-len 4096 \
    --max-num-seqs 1024 --limit-mm-per-prompt image=2