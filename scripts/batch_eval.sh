
SPLIT="dev" # Can be train, dev, or validation based on your dataset
RANDOM_SEED=42
IMAGE_AREA=65636
VAL_SIZE=700
BS=100
K_SAMPLE=1
SOURCE="HF" #Adjust this to HF or local for data
MODEL_LOCATION="HF"  #Adjust this to local if it's saved locally
MODEL_PATH="Image-editing/imged_rl_grpo_sft.s_rl.sc" # Adjust this to local path if it's saved locally
HF_REVISION="ckpt_001999" # Adjust this to the specific revision if needed or main for the main branch
MODE="E" # Adjust this 'CE' for CoT editing and 'E' without CoT, for CE mode make sure the model is trained with CoT
NEW_TOKENS=11000
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
FORMATTED_MODEL_NAME=$(echo "$MODEL_PATH" | sed 's/\//_/g' | sed 's/-/_/g')
SAVE_DIR_PREFIX="path/to/save/dir"
SAVE_DIR="${SAVE_DIR_PREFIX}/${FORMATTED_MODEL_NAME}_${RANDOM_SEED}_${TIMESTAMP}_${MODE}_${SPLIT}"

# Define the CoT keys for logging ground truth reasoning
# CoT_keys=(
#     # "reasoning_verbose"
# )

original_image_key="src_img" # Key for the original image in the dataset
edit_instruction="edited_prompt_list" # Key for the edit instruction in the dataset
edited_image="edited_img" # Key for the edited image in the dataset
edit_type_key="task" # Key for the edit type in the dataset, if applicable


# Define the dataset paths it should be changed to name of data on HF for local path
VALPATH=(
    "TIGER-Lab/OmniEdit-Filtered-1.2M"
    # "path/to/raw/validation/data" or "dataset/name" on HF
)
python ./evaluation/batch_inference.py \
    --save_dir "$SAVE_DIR" \
    --random_seed $RANDOM_SEED \
    --image_area $IMAGE_AREA \
    --sample_size $VAL_SIZE \
    --dataset_names "${VALPATH[@]}" \
    --source "$SOURCE" \
    --model_location "$MODEL_LOCATION" \
    --max_new_tokens $NEW_TOKENS\
    --mode "$MODE"\
    --split "$SPLIT"\
    --batch_size $BS\
    --k_sample $K_SAMPLE \
    --use_logit_processor \
    --original-image-key "$original_image_key" --edit-instruction "$edit_instruction" --edited-image "$edited_image" \
    --model_path "$MODEL_PATH" \
    --revision $HF_REVISION \
    --edit-type-key "$edit_type_key" \
    # --cot_keys "${CoT_keys[@]}" \