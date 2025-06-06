data_name="dataset name on HF"
output_dir="dir to save tokenize data"
original_image_key="source_image"
edit_instruction="edit_instruction"
CoT_keys=("source_image_grounding_information" "conditioning_information" "thinking")
edited_image="edited_image"
image_area=65536
split="train"


CUDA_VISIBLE_DEVICES=0 python ./emu3/train_image_editing/prepare_data.py --dataset-name "$data_name" --output-path "$output_dir" --original-image-key "$original_image_key" --edit-instruction "$edit_instruction" --CoT-keys "${CoT_keys[@]}" --edited-image "$edited_image" --image-area $image_area --random-seed 42 --split $split &
CUDA_VISIBLE_DEVICES=1 python ./emu3/train_image_editing/prepare_data.py --dataset-name "$data_name" --output-path "$output_dir" --original-image-key "$original_image_key" --edit-instruction "$edit_instruction" --CoT-keys "${CoT_keys[@]}" --edited-image "$edited_image" --image-area $image_area --random-seed 75 --split $split &
CUDA_VISIBLE_DEVICES=2 python ./emu3/train_image_editing/prepare_data.py --dataset-name "$data_name" --output-path "$output_dir" --original-image-key "$original_image_key" --edit-instruction "$edit_instruction" --CoT-keys "${CoT_keys[@]}" --edited-image "$edited_image" --image-area $image_area --random-seed 96 --split $split &
CUDA_VISIBLE_DEVICES=3 python ./emu3/train_image_editing/prepare_data.py --dataset-name "$data_name" --output-path "$output_dir" --original-image-key "$original_image_key" --edit-instruction "$edit_instruction" --CoT-keys "${CoT_keys[@]}" --edited-image "$edited_image" --image-area $image_area --random-seed 97 --split $split &

wait

