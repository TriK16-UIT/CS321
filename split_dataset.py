from datasets import Dataset

# Load and shuffle
dataset = Dataset.from_parquet("Data/train.parquet")
dataset = dataset.shuffle(seed=42)

# Split sizes
n = len(dataset)
init_size = n // 3
remaining_size = n - init_size
checkpoint_size = remaining_size // 10

# Save train_init
train_init = dataset.select(range(init_size))
train_init.to_parquet("Checkpoints/train-init.parquet")

# Save checkpoints
remaining = dataset.select(range(init_size, n))
for i in range(10):
    start = i * checkpoint_size
    end = min((i + 1) * checkpoint_size, remaining_size)
    checkpoint = remaining.select(range(start, end))
    checkpoint.to_parquet(f"Checkpoints/train-checkpoint-{i+1}.parquet")

# Verify
split_dataset = Dataset.from_parquet("Checkpoints/train-checkpoint-1.parquet")
print("Split schema (datasets split):")
print(split_dataset.features)