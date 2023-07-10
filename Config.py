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
    learning_rate = 1e-5
    epochs = 30000
    batch_size = 256
    early_stop = 1000
    valid_ratio = 0.2


if __name__ == "__main__":
    print(Config.seed)