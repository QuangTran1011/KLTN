import json
from collections import defaultdict
from copy import deepcopy
from typing import List
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from multiprocessing import Process, Manager, Queue


class SkipGramDataset(Dataset):
    """
    Dataset SkipGram với negative sampling dùng multiprocessing + progress bar
    """

    def __init__(
        self,
        sequences: List[int],
        interacted=defaultdict(set),
        item_freq=defaultdict(int),
        window_size=2,
        negative_samples=5,
        id_to_idx=None,
        seed: int = 42,
        num_processes: int = 4,
    ):
        self.sequences = sequences
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.num_processes = num_processes

        if id_to_idx is None:
            self.id_to_idx = dict()
            self.idx_to_id = dict()
        else:
            self.id_to_idx = id_to_idx
            self.idx_to_id = {v: k for k, v in id_to_idx.items()}

        self.idx_to_seq = []
        self.idx_to_seq_idx = []

        self.interacted = deepcopy(interacted)
        self.item_freq = deepcopy(item_freq)

        np.random.seed(seed)

        for seq_idx, seq in tqdm(
            enumerate(sequences),
            desc="Building interactions",
            total=len(sequences),
            leave=True,
        ):
            self.idx_to_seq.extend([seq_idx] * len(seq))
            self.idx_to_seq_idx.extend(np.arange(len(seq)))

            for item in seq:
                idx = self.id_to_idx.get(item)
                if idx is None:
                    idx = len(self.id_to_idx)
                    self.id_to_idx[item] = idx
                    self.idx_to_id[idx] = item

            seq_idx_set = set([self.id_to_idx[id_] for id_ in seq])
            for idx in seq_idx_set:
                self.interacted[idx].update(seq_idx_set)
                self.item_freq[idx] += 1

        self.vocab_size = len(self.item_freq) if id_to_idx is None else len(self.id_to_idx)

        items, frequencies = zip(*self.item_freq.items())
        self.item_freq_array = np.zeros(self.vocab_size)
        self.item_freq_array[np.array(items)] = frequencies

        self.items = np.arange(self.vocab_size)
        self.sampling_probs = self.item_freq_array**0.75
        self.sampling_probs /= self.sampling_probs.sum()

        self.dataset = self.create_pair_data()

    # ------------------ Negative sampling worker ------------------
    @staticmethod
    def negative_sampling_worker(pairs_part, items, interacted, sampling_probs, negative_samples, queue, return_dict, idx):
        pairs_with_neg = []
        labels_with_neg = []

        for target_item, _ in pairs_part:
            negative_sampling_probs = sampling_probs.copy()
            negative_sampling_probs[list(interacted[target_item])] = 0
            negative_sampling_probs /= negative_sampling_probs.sum()

            negative_items = np.random.choice(
                items,
                size=negative_samples,
                p=negative_sampling_probs,
                replace=False,
            )

            for negative_item in negative_items:
                pairs_with_neg.append((target_item, negative_item))
                labels_with_neg.append(0)

            # gửi update cho progress bar
            queue.put(1)

        return_dict[idx] = (pairs_with_neg, labels_with_neg)

    # ------------------ Tạo pair data với multiprocessing + progress bar ------------------
    def create_pair_data(self):
        pairs = []
        labels = []

        # 1️⃣ Tạo positive pairs
        for idx in tqdm(range(len(self.idx_to_seq)), desc="Building positive pairs"):
            sequence_idx = self.idx_to_seq[idx]
            sequence = self.sequences[sequence_idx]
            sequence = [self.id_to_idx[item] for item in sequence]
            i = self.idx_to_seq_idx[idx]
            target_item = sequence[i]

            start = max(i - self.window_size, 0)
            end = min(i + self.window_size + 1, len(sequence))

            for j in range(start, end):
                if i != j:
                    context_item = sequence[j]
                    pairs.append((target_item, context_item))
                    labels.append(1)

        # 2️⃣ Chia positive pairs
        n = len(pairs)
        pairs_parts = []
        for i in range(self.num_processes):
            start_idx = i * n // self.num_processes
            end_idx = (i + 1) * n // self.num_processes
            pairs_parts.append(pairs[start_idx:end_idx])

        # 3️⃣ Manager dict + Queue cho progress bar
        manager = Manager()
        return_dict = manager.dict()
        queue = Queue()

        processes = []
        for i, part in enumerate(pairs_parts):
            p = Process(target=self.negative_sampling_worker, args=(
                part, self.items, self.interacted, self.sampling_probs,
                self.negative_samples, queue, return_dict, i
            ))
            p.start()
            processes.append(p)

        # 4️⃣ progress bar
        pbar = tqdm(total=n, desc="Building negative pairs")
        finished = 0
        while finished < n:
            queue.get()
            finished += 1
            pbar.update(1)

        for p in processes:
            p.join()

        # 5️⃣ Merge negative samples
        for i in range(self.num_processes):
            part_pairs, part_labels = return_dict[i]
            pairs.extend(part_pairs)
            labels.extend(part_labels)

        # 6️⃣ Convert to tensor & shuffle
        target_items = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
        context_items = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)

        perm = torch.randperm(len(labels))
        target_items = target_items[perm]
        context_items = context_items[perm]
        labels = labels[perm]

        return {
            "target_items": target_items,
            "context_items": context_items,
            "labels": labels,
        }

    # ------------------ Dataset methods ------------------
    def __len__(self):
        return len(self.dataset["target_items"])

    def __getitem__(self, idx):
        return {
            "target_items": self.dataset['target_items'][idx],
            "context_items": self.dataset['context_items'][idx],
            "labels": self.dataset['labels'][idx],
        }

    def collate_fn(self, batch):
        target_items = torch.stack([x['target_items'] for x in batch])
        context_items = torch.stack([x['context_items'] for x in batch])
        labels = torch.stack([x['labels'] for x in batch])
        return {
            "target_items": target_items,
            "context_items": context_items,
            "labels": labels,
        }

    def save_id_mappings(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(
                {
                    "id_to_idx": self.id_to_idx,
                    "idx_to_id": self.idx_to_id,
                },
                f,
            )
