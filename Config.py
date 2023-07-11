import torch

class Config:
    # Time
    time_string = None

    # Paths
    base_path = None
    data_path = None
    save_path = None
    output_path = None

    # Load Models
    load_ckpt = False
    load_name = None

    # Dataset / DataLoader
    train_loader = None
    valid_loader = None
    test_loader = None

    # Training Related
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = None
    optimizer = None

    seed = 3141592
    learning_rate = 3e-4
    epochs = 10000
    batch_size = 128
    early_stop = 50
    valid_ratio = 0.2


if __name__ == "__main__":
    print(Config.seed)