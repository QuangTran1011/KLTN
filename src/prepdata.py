import json
import string
import sys

import pandas as pd
from loguru import logger
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, "..")

from id_mapper import IDMapper
from SkipGram.dataset import SkipGramDataset

class Args(BaseModel):
    num_negative_samples: int = 2
    window_size: int = 1
    batch_size: int = 256

    user_col: str = "user_id"
    item_col: str = "item_id"
args = Args()


sequences_fp = './data/item_sequence.json'
val_sequences_fp = './data/val_item_sequence.json'
with open(sequences_fp, "r") as f:
    item_sequence = json.load(f)
with open(val_sequences_fp, "r") as f:
    val_item_sequence = json.load(f)
idm = idm = IDMapper().load("./data/idm.json")


# dataset = SkipGramDataset(
#     item_sequence,
#     window_size=args.window_size,
#     negative_samples=args.num_negative_samples,
#     id_to_idx=idm.item_to_index,
#     num_workers=4
# )

# print(dataset[0])

if __name__ == "__main__":
    dataset = SkipGramDataset(
        item_sequence,
        window_size=args.window_size,
        negative_samples=args.num_negative_samples,
        id_to_idx=idm.item_to_index,
        num_workers=2
    )

    print(dataset[0])