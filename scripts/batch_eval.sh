
SPLIT="validation" # Can be train or validation
RANDOM_SEED=42
IMAGE_AREA=65636
VAL_SIZE=1000
BS=100
K_SAMPLE=1
SOURCE="local" #Adjust this to HF or local for data
MODEL_LOCATION="HF"  #Adjust this to local if it's saved locally
MODEL_PATH="name/model_on_HF" # Adjust this to local path if it's saved locally
MODE="E" # Adjust this 'CE' for CoT editing and 'E' without CoT
NEW_TOKENS=11000
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
FORMATTED_MODEL_NAME=$(echo "$MODEL_PATH" | sed 's/\//_/g' | sed 's/-/_/g')
SAVE_DIR_PREFIX="path/to/save/dir"
SAVE_DIR="${SAVE_DIR_PREFIX}/${FORMATTED_MODEL_NAME}_${RANDOM_SEED}_${TIMESTAMP}_${MODE}_${SPLIT}"

# Define the CoT keys for logging ground truth reasoning
# CoT_keys=(
#     # "reasoning_verbose"
# )

original_image_key="source_image"
edit_instruction="edit_instruction"
edited_image="edited_image"


# Define the dataset paths it should be changed to name of data on HF for local path
VALPATH=(
    # "path/to/raw/validation/data" or "dataset/name" on HF
)
python ./evaluation/batch_inference.py \
    --model_path "$MODEL_PATH" \
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
    # --cot_keys "${CoT_keys[@]}" \