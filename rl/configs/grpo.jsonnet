{
    // Dataset and data paths
    train_paths_file: [], 
    coefficients: [],
    val_paths_file: [
        std.extVar("APP__DATA_DIR") + "/validation_22march_256/feature/list/tokenized_datalist.json",
    ],
    val_coefficients: [1.0],
    base_exp_dir: std.extVar("APP__BASE_EXP_DIR"),

    // Model and tokenizer paths
    model_path: null,
    tokenizer_path: "BAAI/Emu3-Gen",

    // Assistant prefill prompt
    assistant_prefill: "<|image start|>",

    // Training hyperparameters
    learning_rate: 3e-6,
    num_iterations: 2000,
    kl_coeff: 0.0003,
    temperature: 1.0,
    per_device_batch_size: 8, # reduce this if it doesn't fit in the GPU memory

    // Generation parameters
    episodes_per_iteration: 128,
    generations_per_sample: 8,
    max_response_tokens: 2048,
    top_p: 1.0,
    top_k: -1,

    // vLLM configuration
    vllm_gpu_memory_utilization: 0.3,
    model_context_size: 3000,

    // Emu3 specific settings
    emu3_use_logit_processor: true,
    separate_language_and_vision_vocabs: true,

    // Reward function configuration
    reward_function_api_base: std.extVar("APP__REWARD_FUNCTION_API_BASE"),
    reward_function_api_model_name: "Qwen/Qwen2.5-VL-72B-Instruct",
    reward_compute_viescore: true,
    reward_compute_ground_score: false,
    reward_final_reward: "viescore", 

    // Logging and checkpointing
    run_name: std.extVar("APP__RUN_NAME"),
    log_episodes_every_n_iterations: 5,
    keep_checkpoints_every_n_iterations: 100,
    save_checkpoints_every_n_iterations: 100,
    push_to_hub_every_n_iterations: 100,
    push_to_hub_repo_id: 'imged_rl__' + $.run_name,
} 
