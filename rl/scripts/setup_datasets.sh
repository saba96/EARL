#!/bin/bash
echo "Downloading and extracting the dataset; Can take up to an hour..."

DATA_DIR=RL_DATASETS
mkdir -p $DATA_DIR
python rl/dl_data.py --base_dir $DATA_DIR --url https://huggingface.co/datasets/mair-lab/omniedit-got-tokenized-256/resolve/main/validation_22march_256.tar.gz
python rl/dl_data.py --base_dir $DATA_DIR --url https://huggingface.co/datasets/mair-lab/omniedit-got-tokenized-256/resolve/main/train_22march_256.tar.gz
python rl/dl_data.py --base_dir $DATA_DIR --url https://huggingface.co/datasets/rabiulawal/emu3_tokenized_rl_100k/resolve/main/emu3_tokenized_rl_100k.tar.gz

ln -snf -T $DATA_DIR/data/validation_22march_256/list $DATA_DIR/data/validation_22march_256/feature/list
ln -snf -T $DATA_DIR/data/train_22march_256/list $DATA_DIR/data/train_22march_256/feature/list 

echo "Setup completed successfully!"