
# used this guide: https://stackoverflow.com/questions/76001128/splitting-dataset-into-train-test-and-validation-using-huggingface-datasets-fun

# split training set into train, test, validate
# do fine tuning using SFT Trainer
# choose SmolLm2 model
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import DatasetDict

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

#######################
# IMPORTS/DATA INGEST #
#######################

# load numina math lean for training
original_ds = load_dataset("AI-MO/NuminaMath-LEAN")

# scrub dataset/reformat such that SFT is viable
# see here: https://huggingface.co/docs/trl/main/en/sft_trainer#expected-dataset-type-and-format

def preprocess_ds(datum):
    return {
        "prompt": [{"role": "user", "content": datum["problem"]}],
        "completion": [
            {"role": "assistant", "content": datum['formal_ground_truth']}
        ]
    }

# format and remove all columns
formatted_ds = original_ds.map(preprocess_ds, 
                      remove_columns=["uuid", "question_type", "answer", "author", "formal_statement", "ground_truth_type", "formal_proof", "rl_data", "source", "problem_type", "exam"])

#print(next(iter(formatted_ds["train"])))
# #split into 80 - 20 train/test

# due to time constraints, split into a 0.5%/99.5%. So that's roughly 500 questions/datums to deal with for training
# we did this because training with all ~105000 examples was going to take hundreds of hours, or even 80% would have been rough.
# note: this may have been due to inefficient choices... we could see if this works on Unsloth?
#split_ds = formatted_ds["train"].train_test_split(test_size=0.20)
split_ds = formatted_ds["train"].train_test_split(test_size=0.997)
print(split_ds)
# # train set
train_set = formatted_ds["train"]

# create validation set and test sets (10% each, or 50% of 20% each)
# valid_test_set = split_ds["test"].train_test_split(test_size=0.5)
# valid_set = valid_test_set["train"]
# test_set = valid_test_set["test"]

ds_splits = DatasetDict(
    {
        'train': split_ds["train"]
    }
)
# ds_splits = DatasetDict({
#         'train': train_set,
#         'test': test_set,
#         'valid': valid_set
#     })

# # print before and after splits
print(f"Before:\n {original_ds}\n")
print(f"After:\n {ds_splits}\n")


# ##########################
# # SUPERVISED FINE TUNING #
# ##########################

# # Taken from sft_trainer.py in lean-dojo
class SFTTrainerClass:
    def __init__(self, model_name: str, output_dir: str, epochs_per_repo: int,
                 batch_size: int, lr: float):
        self.model_name = model_name
        self.output_dir = output_dir
        self.epochs_per_repo = epochs_per_repo
        self.batch_size = batch_size
        self.lr = lr

    
    def train(self, train_set):
        print("creating config")
        config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs_per_repo,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            packing=False,
            completion_only_loss=True,
            use_cpu=True)

        print("creating trainer")
        trainer = SFTTrainer(
                self.model_name,
                config,
                train_dataset=train_set)

        print("created trainer, about to train")
        trainer.train()
        print("trainer is finished")

# Create the trainer, give it a batch size of 10 and learning rate of 0.02 (used to be like 2e-05)
sf_trainer = SFTTrainerClass("HuggingFaceTB/SmolLM2-135M-Instruct", "outputs", 1, 10, .02)

# train with dataset
sf_trainer.train(ds_splits["train"])

print("finished")
