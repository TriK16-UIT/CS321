import fire
from datasets import Dataset

def preprocess_dataset(
    input_file: str = "Data/train.parquet",
    output_dir: str = "Checkpoints",
    init_ratio: float = 1/3,
    num_checkpoints: int = 10,
    seed: int = 42
):
    """
    Split a dataset into an initial training set and multiple checkpoint sets.
    
    Args:
        input_file (str): Path to the input parquet file
        output_dir (str): Directory to save the split datasets
        init_ratio (float): Ratio of data to use for initial training
        num_checkpoints (int): Number of checkpoint sets to create
        seed (int): Random seed for shuffling
    """
    # Load and shuffle
    dataset = Dataset.from_parquet(input_file)
    dataset = dataset.shuffle(seed=seed)

    # Split sizes
    n = len(dataset)
    init_size = int(n * init_ratio)
    remaining_size = n - init_size
    checkpoint_size = remaining_size // num_checkpoints

    # Save train_init
    train_init = dataset.select(range(init_size))
    train_init.to_parquet(f"{output_dir}/train-init.parquet")
    print(f"Saved initial training set with {len(train_init)} samples to {output_dir}/train-init.parquet")

    # Save checkpoints
    remaining = dataset.select(range(init_size, n))
    for i in range(num_checkpoints):
        start = i * checkpoint_size
        end = min((i + 1) * checkpoint_size, remaining_size)
        checkpoint = remaining.select(range(start, end))
        checkpoint.to_parquet(f"{output_dir}/train-checkpoint-{i+1}.parquet")
        print(f"Saved checkpoint {i+1} with {len(checkpoint)} samples to {output_dir}/train-checkpoint-{i+1}.parquet")

    # Verify
    split_dataset = Dataset.from_parquet(f"{output_dir}/train-checkpoint-1.parquet")
    print("Split schema (datasets split):")
    print(split_dataset.features)
    
    return {
        "init_size": init_size,
        "checkpoint_size": checkpoint_size,
        "total_samples": n
    }

if __name__ == "__main__":
    fire.Fire(preprocess_dataset)