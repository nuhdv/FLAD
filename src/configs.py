def get_best_config(ds_name, epochs, seq_len, stride):

    n_dims = {"SWaT": 51, "WADI": 123, "MSL": 55, "SMAP": 25, "SMD": 38, "ASD": 19}
    batch_size = 128
    train_val_pc = 0.25
    gpu = 0

    best_configs = {
        "num_epochs": epochs,
        "lr": 0.00015,
        "dropout": 0.42,
        "num_channels": [64, 128, 256],
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "stride": stride,
        "gpu": gpu,
        "train_val_percentage": train_val_pc,
    }
    return best_configs



