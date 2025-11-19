MODEL_PATHS=(
    "mair-lab/sft-simple.rl-simple-n-complex"
)

USE_REVISION="False"
REVISION=(
    ""
)

REASONING_INPUT="False"
BATCH_SIZE=1100
MODE="E"
SAVE_DIR="./results"

python our_eval_on_aurora.py \
    --save_dir $SAVE_DIR \
    --batch_size $BATCH_SIZE \
    --mode $MODE \
    --model_paths "${MODEL_PATHS[@]}" \
    --reasoning_input $REASONING_INPUT \
    --revision "${REVISION[@]}" \
    --use_revision $USE_REVISION

python our_eval_on_emuedit.py \
    --save_dir $SAVE_DIR \
    --batch_size $BATCH_SIZE \
    --mode $MODE \
    --model_paths "${MODEL_PATHS[@]}" \
    --reasoning_input $REASONING_INPUT \
    --revision "${REVISION[@]}" \
    --use_revision $USE_REVISION

python our_eval_on_i2ebench.py \
    --save_dir $SAVE_DIR \
    --batch_size $BATCH_SIZE \
    --mode $MODE \
    --model_paths "${MODEL_PATHS[@]}" \
    --reasoning_input $REASONING_INPUT \
    --revision "${REVISION[@]}" \
    --use_revision $USE_REVISION

python our_eval_on_magicbrush.py \
    --save_dir $SAVE_DIR \
    --batch_size $BATCH_SIZE \
    --mode $MODE \
    --model_paths "${MODEL_PATHS[@]}" \
    --reasoning_input $REASONING_INPUT \
    --revision "${REVISION[@]}" \
    --use_revision $USE_REVISION

python our_eval_on_omniedit.py \
    --save_dir $SAVE_DIR \
    --batch_size $BATCH_SIZE \
    --mode $MODE \
    --model_paths "${MODEL_PATHS[@]}" \
    --reasoning_input $REASONING_INPUT \
    --revision "${REVISION[@]}" \
    --use_revision $USE_REVISION

python our_eval_on_vismin.py \
    --save_dir $SAVE_DIR \
    --batch_size $BATCH_SIZE \
    --mode $MODE \
    --model_paths "${MODEL_PATHS[@]}" \
    --reasoning_input $REASONING_INPUT \
    --revision "${REVISION[@]}" \
    --use_revision $USE_REVISION
