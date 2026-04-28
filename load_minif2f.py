from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Tonic/MiniF2F")

# split into train and test datasets (80% train, 20% test)
split_ds = ds["train"].train_test_split(test_size=0.2)

print(f"dataset is {split_ds}")

print(f"dataset is {split_ds["train"][0]}")
