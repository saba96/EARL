
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}

echo WORLD_SIZE is $WORLD_SIZE
echo RANK is $RANK
echo MASTER_ADDR is $MASTER_ADDR
echo MASTER_PORT is $MASTER_PORT
echo NGPUS is $NGPUS

# Important: Adjust this, path to tokenized training data
DATAPATH=(
    "path/to/tokenized/data/list/tokenized_datalist.json" # path to tokenized training data
)

# Important: Adjust this, path to tokenized validation data
VALPATH=(
    "path/to/tokenized/validation/datalist/tokenized_datalist.json" # path to tokenized validation data
)
# Total number of samples to be sampled uniformly from all dataset, if you have 2 validation datasets and validation_size is set to 10 it samples 5 from each, seed is fixed.
VALIDATION_SIZE=1000

# COT_KEYS=(
#     'reasoning_verbose'
# )

# COEF=(
#     1
# )

# adjust based on resolution: 256*256=65536
IMAGE_AREA=65536
# adjust based on resolution and thinking length: you can print max_length for few steps in my_datasets.py in training: each image (256/8)*((256/8)=1024, we have two images so 2048 tokens for images only
MAX_POS_EMB=3000
# Adjust this to your desired output directory
OUTPUT_DIR="/path/to/output/directory"


# TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
unique_run_identifier_date="SFT_Experiment_name"
if [ ${#COT_KEYS[@]} -eq 0 ]; then
    COT_LABEL="without_cot"
else
    COT_LABEL=$(IFS=_; echo "${COT_KEYS[*]}")
fi

EXP_NAME="Emu3-Base-SFT-${COT_LABEL}-${unique_run_identifier_date}"
echo "Experiment Name: $EXP_NAME"

torchrun \
    --nproc_per_node=4 \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ./emu3/train_image_editing/train_image_editing.py \
    --model_name_or_path BAAI/Emu3-Stage1 \
    --deepspeed ./scripts/zero3_offload.json \
    --data_paths "${DATAPATH[@]}" \
    --validation_paths "${VALPATH[@]}" \
    --validation_size $VALIDATION_SIZE \
    --image_area $IMAGE_AREA \
    --max_position_embeddings $MAX_POS_EMB \
    --output_dir "${OUTPUT_DIR}/${EXP_NAME}" \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy steps \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 50 \
    --learning_rate 1e-4 \
    --min_learning_rate 1e-5 \
    --weight_decay 0.1 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --warmup_steps 30 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name ${EXP_NAME} \
    # --cot_keys "${COT_KEYS[@]}" \
    # --coefficients "${COEF[@]}"