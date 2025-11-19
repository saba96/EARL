SAVE_DIR="./results"

BASELINE=(
    "InstructPix2Pix"
    "MagicBrush"
    "Aurora"
    "Omnigen"
)

python baselines_eval_on_aurora.py \
    --save_dir $SAVE_DIR \
    --batch_size 40 \
    --baseline "${BASELINE[@]}"

python baselines_eval_on_emuedit.py \
    --save_dir $SAVE_DIR \
    --batch_size 50 \
    --baseline "${BASELINE[@]}"

python baselines_eval_on_i2ebench.py \
    --save_dir $SAVE_DIR \
    --batch_size 50 \
    --baseline "${BASELINE[@]}"

python baselines_eval_on_magicbrush.py \
    --save_dir $SAVE_DIR \
    --batch_size 40 \
    --baseline "${BASELINE[@]}"

python baselines_eval_on_omniedit.py \
    --save_dir $SAVE_DIR \
    --batch_size 40 \
    --baseline "${BASELINE[@]}"

python baselines_eval_on_vismin.py \
    --save_dir $SAVE_DIR \
    --batch_size 40 \
    --baseline "${BASELINE[@]}"
