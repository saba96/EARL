{
    // Dataset and data paths
    train_paths_file: [
        std.extVar("APP__DATA_DIR") + "/emu3_tokenized_rl_100k/aurora_ag/train_20250208_256/list/tokenized_datalist.json",
        std.extVar("APP__DATA_DIR") + "/emu3_tokenized_rl_100k/aurora_kubric/train_3may_256/list/tokenized_datalist.json",
        std.extVar("APP__DATA_DIR") + "/emu3_tokenized_rl_100k/human-edit/train_20250208_256/list/tokenized_datalist.json",
        std.extVar("APP__DATA_DIR") + "/emu3_tokenized_rl_100k/magicbrush/train_20250208_256/list/tokenized_datalist.json",
        std.extVar("APP__DATA_DIR") + "/emu3_tokenized_rl_100k/ss2/train_3may_256/list/tokenized_datalist.json",
        std.extVar("APP__DATA_DIR") + "/emu3_tokenized_rl_100k/vismin/train_3may_relcnt_256/list/tokenized_datalist.json",
    ],
    coefficients: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
} 
