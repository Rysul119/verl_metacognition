import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_prompt_len", default = 1000)
    parser.add_argument("--local_dir", default="~/data/ethics")
    parser.add_argument("--hdfs_dir", default=None)
    #sys_prompt = "The User is going to provide you with some commonsense ethical or not ethical statements. Just acknowledge the statement."
    args = parser.parse_args()

    data_source = "hendrycks/ethics"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    # filter by statement length 
    def keep_example(ex):
        return len(ex["input"]) <= args.max_prompt_len
        
    dataset = dataset.filter(keep_example)

    train_dataset = dataset["train"]
    holdout_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            statement = example.pop("input")
            label = int(example.pop("label"))
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": statement}],
                "ability": "ethics/commonsense",
                "reward_model": {"style": "rule", "ground_truth": label},
                "extra_info": {"split": split, "index": idx},
                "label": label
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    holdout_dataset = holdout_dataset.map(function=make_map_fn("holdout"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    holdout_dataset.to_parquet(os.path.join(local_dir, "holdout.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)