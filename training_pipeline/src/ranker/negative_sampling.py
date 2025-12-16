import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Process, Manager, Queue
from typing import List


def generate_negative_samples_mp(
    df,
    user_col="user_indice",
    item_col="item_indice",
    label_col="rating",
    timestamp_col="timestamp",
    neg_label=0,
    neg_to_pos_ratio=1,
    seed=None,
    features: List[str] = [],
    num_processes: int = 6
):
    """
    Generate negative samples using multiprocessing.
    """

    if seed is not None:
        np.random.seed(seed)

    # ----- Chuẩn bị dữ liệu chung -----
    item_popularity = df[item_col].value_counts()
    items = item_popularity.index.values
    all_items_set = set(items)
    user_item_dict = df.groupby(user_col)[item_col].apply(set).to_dict()
    popularity = item_popularity.values.astype(np.float64)
    sampling_probs = popularity / popularity.sum()
    item_to_index = {item: idx for idx, item in enumerate(items)}

    # ----- Hàm worker -----
    def worker(df_part, return_dict, queue, idx):
        local_results = []

        for _, row in df_part.iterrows():
            user = row[user_col]
            pos_items = user_item_dict[user]
            negative_candidates = all_items_set - pos_items

            if not negative_candidates:
                queue.put(1)
                continue

            num_neg = min(neg_to_pos_ratio, len(negative_candidates))
            negative_candidates_list = list(negative_candidates)
            candidate_indices = [item_to_index[item] for item in negative_candidates_list]
            candidate_probs = sampling_probs[candidate_indices]
            candidate_probs /= candidate_probs.sum()

            sampled_items = np.random.choice(
                negative_candidates_list, size=num_neg, replace=False, p=candidate_probs
            )

            for neg_item in sampled_items:
                new_row = {col: row[col] for col in [user_col, timestamp_col, *features]}
                new_row[item_col] = neg_item
                new_row[label_col] = neg_label
                local_results.append(new_row)

            queue.put(1)

        return_dict[idx] = pd.DataFrame(local_results)

    # ----- Chia dữ liệu -----
    parts = np.array_split(df, num_processes)

    manager = Manager()
    return_dict = manager.dict()
    queue = Queue()

    processes = []
    for i, part in enumerate(parts):
        p = Process(target=worker, args=(part, return_dict, queue, i))
        p.start()
        processes.append(p)

    # ----- progress bar -----
    pbar = tqdm(total=len(df), desc="Generating negative samples")
    finished = 0
    while finished < len(df):
        queue.get()
        finished += 1
        pbar.update(1)

    for p in processes:
        p.join()

    # ----- Gộp kết quả -----
    df_negative = pd.concat(return_dict.values(), ignore_index=True)
    return df_negative
