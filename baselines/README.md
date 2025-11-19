# Evaluation Protocol

## Prerequisites

Ensure the following credential files exist inside the `baselines` directory:
- HuggingFace authentication token (`dummy_hf_token.txt`)
- OpenAI API key (`api_key.txt`)

## Step 1: Dataset Preparation

Navigate to the following directory and download the following benchmark datasets:

```bash
cd baselines
mkdir -p benchmarks
```

Download the I2EBench evaluation dataset:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='sikarwarank/i2bench_eval', filename='i2ebench-dataset.parquet', repo_type='dataset', local_dir='./benchmarks')"
```

Download the MagicBrush test split:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='sikarwarank/magicbrush_eval', filename='magicbrush-test-00000.parquet', repo_type='dataset', local_dir='./benchmarks')"
```

Other benchmark datasets will be automatically pulled from their respective HF repos when following evaluation scripts are executed.

## Step 2: Model Evaluation

Configure the model path in the evaluation script to specify the model variant for assessment. Update the `MODEL_PATH` variable in `scripts/our_eval.sh` accordingly.

Execute the evaluation pipeline for our model:
```bash
bash scripts/our_eval.sh
```

Execute the evaluation pipeline for other baselines:
```bash
bash scripts/baseline_eval.sh
```

**Note:** All evaluation results will be saved to the `./results` directory.

## Step 3: VIEScore Computation

Calculate the VIEScore metrics using the generated evaluation results:

```bash
python gen_viescore.py --json_path {PATH_TO_RESULTS_JSON}
```

Replace `{PATH_TO_RESULTS_JSON}` with the actual path to the corresponding JSON file generated in Step 2.

Output: The computed VIEScore statistics will be written to {PATH_TO_RESULTS_JSON}_viescore_stats.json in the same directory as the input file.

## Directory Structure

```
baselines/
├── benchmarks/          # Downloaded evaluation datasets
├── results/             # Evaluation outputs and metrics
└── scripts/
    └── our_eval.sh      # Main evaluation script
```

## Citation

If you use these evaluation protocols in your research, please cite the relevant papers for the Vismin, I2EBench, MagicBrush, OmniEdit, EmuEdit and Aurora datasets.
