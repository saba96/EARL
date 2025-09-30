{
    // Dataset and data paths
    val_base_dir: std.extVar("APP_DATA_DIR")",

    train_paths_file: [
        "earl-datasets/magic_brush/train/list/tokenized_datalist.json",
        "earl-datasets/aurora_ag/train/list/tokenized_datalist.json",
        "earl-datasets/aurora_kubric/train/list/tokenized_datalist.json",
        "earl-datasets/ss2/train/list/tokenized_datalist.json",
        "earl-datasets/vismin/train/list/tokenized_datalist.json",
        "earl-datasets/human-edit/train/list/tokenized_datalist.json"
    ],

    val_paths_file: [
        // "earl-datasets/omniedit/validation/list/tokenized_datalist.json",
        // "earl-datasets/magic_brush/validation/list/tokenized_datalist.json",
        // "earl-datasets/aurora_ag/validation/list/tokenized_datalist.json",
        // "earl-datasets/aurora_kubric/validation/list/tokenized_datalist.json",
        // "earl-datasets/ss2/validation/list/tokenized_datalist.json",
        // "earl-datasets/vismin/validation/list/tokenized_datalist.json",
        // "earl-datasets/human_edit/validation/list/tokenized_datalist.json"
    ],
    // each dataset should've 50K sample approx, so we upscale or downscale
    coefficients: [
        6
        6
        0.7
        1.55
        3.7
        11 
    ],
    val_coefficients: [
        
    ],

    // Model and tokenizer paths
    model_path: "mair-lab/sft-simple",
    // tokenizer_path: "/replace/with/your/model/path",
    tokenizer_path: "BAAI/Emu3-Stage1",

    // Assistant prefill prompt
    assistant_prefill: "#image start\\n", // Forced no-reasoning mode
    //assistant_prefill: "Let's think step by step. <|start thinking|>\\n", // Forced reasoning mode

    // Training hyperparameters
    learning_rate: 3e-6,
    num_iterations: 3000,
    kl_coeff: 0.0003,
    temperature: 1.0,
    per_device_batch_size: 4,
    // Generation parameters
    episodes_per_iteration: 128,
    generations_per_sample: 8,
    max_response_tokens: 2048,
    top_p: 1.0,
    top_k: -1,

    // VLM configuration
    vlm_spm_memory_utilization: 0.3,
    model_context_size: 4096,

    // Emu3 specific settings
    emu3_use_logits_processor: true,
    separate_language_and_vision_vocabs: true,
    train_text_tokens_only: false,
    // Only used when train_text_tokens_only is true:
    // text_only_cycle_steps: 10,
    // text_only_steps: 7,

    // Reward function configuration
    reward_function_api_base: std.extVar("APP_REWARD_FUNCTION_API_BASE"),
    reward_function_api_model_name: "Qwen/Qwen2.5-VL-72B-Instruct",
    reward_compute_viescore: true,
    reward_compute_ground_score: false,
    // "viescore", "ground_score", or "sqrt(viescore_pxground_score)"
    reward_final_reward: "viescore",

    // Logging and checkpointing
    run_name: std.extVar("APP_RUN_NAME"),
    log_episodes_every_n_iterations: 5,
    keep_checkpoints_every_n_iterations: 100,
    save_checkpoints_every_n_iterations: 100,
    push_to_hub_every_n_iterations: 100,
    push_to_hub_repo_id: "imged_rl__" + $.run_name,
}