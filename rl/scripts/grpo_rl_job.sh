#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=450G
#SBATCH --cpus-per-task=48


# Parse command line arguments
REWARD_API_BASE="NONE"
RUN_NAME="NONE"
CONFIG_FILE="rl/configs/sft_s__rl_sc.jsonnet"  # change this to the desired config file
# or sft_s__rl_c.jsonnet 
# or sft_s__rl_s.jsonnet
# or sft_think_s__rl_sc.jsonnet
# or sft_think_s__rl_s.jsonnet
# or sft_think_s__rl_c.jsonnet


while [[ $# -gt 0 ]]; do
    case $1 in
    --reward_server)
        if [ -f "$2" ]; then
            REWARD_API_BASE="$2"
        elif [ "$2" = "auto" ] || [ "$2" = "AUTO" ]; then
            REWARD_API_BASE="$HOME/path/to/earl/qwen_node.txt"
        else
            REWARD_API_BASE="http://$2:4877/v1"
        fi
        shift 2
        ;;
    --run_name)
        RUN_NAME="$2"
        shift 2
        ;;
    --config)
        CONFIG_FILE="$2"
        shift 2
        ;;
    *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
done

# Load required modules
module load cuda/12.2 arrow/17 python/3.11 opencv httpproxy
source $HOME/venvs/earl/bin/activate

# Set environment variables
export HF_HOME=$HOME/scratch/hf
export HF_HUB_OFFLINE=1

PROJECT_DIR=$HOME/path/to/earl
export APP__DATA_DIR=$PROJECT_DIR/data
export APP__BASE_EXP_DIR=$PROJECT_DIR/experiments

export APP__REWARD_FUNCTION_API_BASE=$REWARD_API_BASE

export APP__RUN_NAME=$RUN_NAME
export WANDB_RUN_ID=$RUN_NAME

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Change to the repository directory
# cd "$REPO_DIR"

torchrun --nproc_per_node=$NUM_GPUS rl/test_distributed_env.py

export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m rl.simple_launch --nproc $NUM_GPUS \
    rl/nano_aha_moment.py \
    $CONFIG_FILE